"""
Multi-token J++ lens: offset-resolved expert Jacobians with LRP.

Combines the offset-resolved J-lens of this repository with the J++ Lens
(Ayonrinde & Lindsey, 2026).  For source layer l, offset Δ and activation
cluster (expert) e:

    J_{l,Δ,e} = E[ ∂_LRP h_final,s+Δ / ∂h_l,s  |  h_l,s ∈ C_{l,e} ]
    J_{l,Δ}   = Σ_e w_{l,Δ,e} J_{l,Δ,e}
    readout_Δ(h_l,s) = top-k over V_sem of  W_U · norm(J_{l,Δ} h_l,s)     → token s+Δ+1

With ``lrp=False`` and count-proportional weights this is the offset J-lens;
with ``lrp=True`` and count weights it is an offset R-lens; with learned
weights (``jspace.expert_weights``) it is the offset J++ lens.

Estimator — exact over dimensions, Hutchinson over positions
-----------------------------------------------------------
As in J++, the forward pass is replicated B times and replica b receives a
one-hot cotangent on output dimension (dim_start + b), so every backward pass
yields B exact Jacobian rows.  To separate offsets, the cotangent is placed
only on target positions t' ≡ r (mod M), M > max(Δ), each multiplied by an
independent random sign σ_{t'}.  For a source s with s+Δ tagged,

    σ_{s+Δ} · ∂(Σ_{t'} σ_{t'} h_final,t'[i]) / ∂h_l,s  =  J(s, s+Δ)[i, :] + noise,

where the noise comes only from other tagged targets s+Δ+kM (k ≥ 1), is zero
mean, and never includes the dominant Δ=0 term.  Sign draws alternate between
two halves, giving a split-half reliability and an unbiased ‖J‖² per (l, Δ).

Positions: the first ``skip_first`` positions (attention sinks) and the final
position are excluded as sources and targets, following J++.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jspace.lrp import LrpRules, lrp_rules
from jspace.model import HookedModel
from jspace.routing import ActivationRouter, fit_router
from jspace.utils import free_device_memory

logger = logging.getLogger(__name__)

_FORMAT = "jspace.expert_offset_lens"
_VERSION = "1.0"


@dataclass(frozen=True)
class ExpertOffsetConfig:
    """
    Hyper-parameters of an expert offset lens fit.

    Attributes
    ----------
    offsets : tuple[int, ...]
        Strictly increasing non-negative offsets Δ.
    layer_indices : tuple[int, ...]
        Source layers (required; d² memory per layer × offset × expert).
    num_experts : int
        Activation clusters per layer (J++ uses 8).  1 disables Jacobian filtering.
    projection_dim : int
        PCA dimensions for routing.
    rows_per_pass : int
        Replicas per forward pass = exact Jacobian rows per backward pass.
    n_sign_draws : int
        Independent position-sign draws per sequence; even, split into two halves.
    skip_first : int
        Leading positions excluded as sources and targets (attention sinks).
    lrp : bool
        Compute Jacobians through the LRP backward graph.
    min_kept_positions : int
        Experts with fewer samples fall back to the pooled Jacobian.
    router_max_points : int
        Maximum activations per layer used to fit a router.
    seed : int
    """

    offsets: Tuple[int, ...] = (0, 1, 2, 3, 4)
    layer_indices: Tuple[int, ...] = ()
    num_experts: int = 8
    projection_dim: int = 64
    rows_per_pass: int = 16
    n_sign_draws: int = 2
    skip_first: int = 16
    lrp: bool = True
    min_kept_positions: int = 50
    router_max_points: int = 20000
    seed: int = 0

    def __post_init__(self) -> None:
        """Normalise sequences to tuples and validate every field."""
        object.__setattr__(self, "offsets", tuple(int(o) for o in self.offsets))
        object.__setattr__(self, "layer_indices", tuple(int(layer) for layer in self.layer_indices))
        if not self.offsets or self.offsets[0] < 0:
            raise ValueError(f"offsets must be non-empty and non-negative, got {self.offsets}")
        if any(b <= a for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError(f"offsets must be strictly increasing, got {self.offsets}")
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError(
                f"layer_indices must be non-empty and unique, got {self.layer_indices}"
            )
        if self.num_experts < 1 or self.projection_dim < 1 or self.rows_per_pass < 1:
            raise ValueError("num_experts, projection_dim and rows_per_pass must be >= 1")
        if self.n_sign_draws < 2 or self.n_sign_draws % 2:
            raise ValueError(
                f"n_sign_draws must be a positive even number, got {self.n_sign_draws}"
            )
        if self.skip_first < 0 or self.min_kept_positions < 1 or self.seed < 0:
            raise ValueError("skip_first and seed must be >= 0, min_kept_positions >= 1")

    @property
    def comb_stride(self) -> int:
        """Comb stride M = max(offsets) + 1."""
        return self.offsets[-1] + 1


def kept_position_mask(seq_len: int, skip_first: int) -> torch.Tensor:
    """(seq_len,) bool: positions in [skip_first, seq_len - 1), the J++ position rule."""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    return mask


class ExpertOffsetLens:
    """
    Offset-resolved expert Jacobians with optional LRP, plus their combination weights.

    Attributes
    ----------
    routers : dict[int, ActivationRouter]
    experts : dict[int, torch.Tensor]
        layer → (K, E, d, d) float32 CPU expert Jacobians (fallbacks already
        replaced by the pooled Jacobian).
    pooled : dict[int, torch.Tensor]
        layer → (K, d, d) float32 CPU count-weighted mean over all experts.
    expert_counts : dict[int, torch.Tensor]
        layer → (K, E) int64 samples per expert.
    fallback : dict[int, torch.Tensor]
        layer → (K, E) bool, experts replaced by the pooled Jacobian.
    weights : dict[int, torch.Tensor]
        layer → (K, E) float32 expert weights of the combined lens; empty until
        :meth:`set_weights` is called.
    reach_sq, split_half_cosine : torch.Tensor
        (n_layers, K) unbiased ‖J_pooled‖_F² and split-half cosine of the pooled estimate.

    Parameters
    ----------
    model : HookedModel
    config : ExpertOffsetConfig
    """

    def __init__(self, model: HookedModel, config: ExpertOffsetConfig) -> None:
        invalid = [layer for layer in config.layer_indices if not 0 <= layer < model.num_layers - 1]
        if invalid:
            raise ValueError(f"layer indices {invalid} outside [0, {model.num_layers - 1})")
        self.model = model
        self.config = config
        self.layer_indices: List[int] = list(config.layer_indices)
        self.offsets: List[int] = list(config.offsets)
        self.d_model: int = int(model.model.config.hidden_size)

        self.routers: Dict[int, ActivationRouter] = {}
        self.experts: Dict[int, torch.Tensor] = {}
        self.pooled: Dict[int, torch.Tensor] = {}
        self.expert_counts: Dict[int, torch.Tensor] = {}
        self.fallback: Dict[int, torch.Tensor] = {}
        self.weights: Dict[int, torch.Tensor] = {}
        self.reach_sq = torch.empty(0)
        self.split_half_cosine = torch.empty(0)
        self.n_sequences = 0
        self._matrix_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        """Model device."""
        return torch.device(self.model.device)

    @property
    def final_layer(self) -> int:
        """Index of the last transformer block (the Jacobian target)."""
        return self.model.num_layers - 1

    # ── fitting ─────────────────────────────────────────────────────────────

    def fit_routers(self, sequences: Sequence[torch.Tensor]) -> "ExpertOffsetLens":
        """
        Fit one activation router per source layer on kept-position activations.

        Parameters
        ----------
        sequences : Sequence[torch.Tensor]
            (1, T) token-id tensors.

        Returns
        -------
        ExpertOffsetLens
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")
        collected: Dict[int, List[torch.Tensor]] = {layer: [] for layer in self.layer_indices}
        n_points = 0
        with torch.no_grad():
            for ids in sequences:
                if n_points >= self.config.router_max_points:
                    break
                kept = kept_position_mask(ids.shape[1], self.config.skip_first)
                if not kept.any():
                    continue
                with self.model.capture_residuals(self.layer_indices) as store:
                    self.model.model(input_ids=ids.to(self.device), use_cache=False)
                for layer in self.layer_indices:
                    collected[layer].append(store[layer][0, kept.to(self.device)].cpu())
                n_points += int(kept.sum())
        for j, layer in enumerate(self.layer_indices):
            points = torch.cat(collected[layer])[: self.config.router_max_points]
            if self.config.num_experts == 1:
                self.routers[layer] = ActivationRouter(
                    mean=torch.zeros(self.d_model),
                    components=torch.eye(self.d_model, 1),
                    centroids=torch.zeros(1, 1),
                )
            else:
                self.routers[layer] = fit_router(
                    points,
                    self.config.num_experts,
                    self.config.projection_dim,
                    seed=self.config.seed + j,
                )
        logger.info(
            "Fitted %d routers on %d positions each",
            len(self.routers),
            min(n_points, self.config.router_max_points),
        )
        return self

    def fit_experts(
        self, sequences: Sequence[torch.Tensor], show_progress: bool = True
    ) -> "ExpertOffsetLens":
        """
        Estimate per-(layer, offset, expert) Jacobians over ``sequences``.

        Parameters
        ----------
        sequences : Sequence[torch.Tensor]
            (1, T) token-id tensors, unpadded.
        show_progress : bool

        Returns
        -------
        ExpertOffsetLens
        """
        if not self.routers:
            raise RuntimeError("fit_routers() first")
        n_layers, n_offsets, n_exp, d = (
            len(self.layer_indices),
            len(self.offsets),
            self.config.num_experts,
            self.d_model,
        )
        logger.info(
            "Expert offset fit: %d layers × %d offsets × %d experts, lrp=%s, d=%d "
            "(CPU accumulators %.2f GB)",
            n_layers,
            n_offsets,
            n_exp,
            self.config.lrp,
            d,
            2 * n_layers * n_offsets * n_exp * d * d * 4 / 1e9,
        )
        sums = torch.zeros((2, n_layers, n_offsets, n_exp, d, d), dtype=torch.float32)
        counts = torch.zeros((2, n_offsets, n_layers, n_exp), dtype=torch.int64)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        n_ok = 0
        for ids in tqdm(
            sequences,
            desc=f"expert offset fit (lrp={self.config.lrp})",
            disable=not show_progress,
            dynamic_ncols=True,
        ):
            try:
                self._accumulate_sequence(ids, sums, counts, generator)
                n_ok += 1
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM on a sequence of length %d — skipped", ids.shape[1])
                free_device_memory(self.device)
        if n_ok == 0:
            raise RuntimeError("No sequences were processed")
        self.n_sequences = n_ok
        self._finalise(sums, counts)
        free_device_memory(self.device)
        return self

    # ── combination and readout ─────────────────────────────────────────────

    def count_weights(self) -> Dict[int, torch.Tensor]:
        """layer → (K, E) count-proportional weights (reproduce the pooled lens absent fallbacks)."""
        return {
            layer: (c.float() / c.sum(-1, keepdim=True).clamp_min(1)).float()
            for layer, c in self.expert_counts.items()
        }

    def set_weights(self, weights: Mapping[int, torch.Tensor]) -> None:
        """
        Set the expert weights of the combined lens.

        Parameters
        ----------
        weights : Mapping[int, torch.Tensor]
            layer → (K, E) weights for every fitted layer.
        """
        missing = set(self.layer_indices) - set(weights)
        if missing:
            raise ValueError(f"weights missing for layers {sorted(missing)}")
        shape = (len(self.offsets), self.config.num_experts)
        for layer, w in weights.items():
            if tuple(w.shape) != shape:
                raise ValueError(f"layer {layer}: weights shape {tuple(w.shape)} != {shape}")
        self.weights = {layer: weights[layer].float().cpu().clone() for layer in self.layer_indices}
        self._matrix_cache = {k: v for k, v in self._matrix_cache.items() if k[1] != "weighted"}

    def matrices(self, layer_idx: int, mode: str = "weighted") -> torch.Tensor:
        """
        The (K, d, d) offset matrices of one layer on the model device.

        Parameters
        ----------
        layer_idx : int
        mode : str
            ``"weighted"`` (Σ_e w_e J_e, requires :meth:`set_weights`) or ``"pooled"``.
        """
        key = (layer_idx, mode)
        if key not in self._matrix_cache:
            if layer_idx not in self.pooled:
                raise KeyError(f"layer {layer_idx} not fitted")
            if mode == "pooled":
                mats = self.pooled[layer_idx]
            elif mode == "weighted":
                if layer_idx not in self.weights:
                    raise RuntimeError("no expert weights set; call set_weights()")
                mats = torch.einsum(
                    "ke,keij->kij", self.weights[layer_idx], self.experts[layer_idx]
                )
            else:
                raise ValueError(f"mode must be 'weighted' or 'pooled', got {mode!r}")
            self._matrix_cache[key] = mats.to(self.device, torch.float32)
        return self._matrix_cache[key]

    def project(self, hidden: torch.Tensor, layer_idx: int, mode: str = "weighted") -> torch.Tensor:
        """(..., d) hidden states → (K, ..., d) transports J_{l,Δ} h."""
        if hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden last dim {hidden.shape[-1]} != d_model {self.d_model}")
        mats = self.matrices(layer_idx, mode)
        return torch.einsum("koi,...i->k...o", mats, hidden.to(mats.device, torch.float32))

    def expert_transports(
        self, hidden: torch.Tensor, layer_idx: int, offset_pos: int
    ) -> torch.Tensor:
        """(P, d) hidden states → (P, E, d) per-expert transports at one offset (for weight fitting)."""
        experts = self.experts[layer_idx][offset_pos].to(self.device, torch.float32)
        return torch.einsum("eoi,pi->peo", experts, hidden.to(self.device, torch.float32))

    def reliability(self) -> torch.Tensor:
        """(n_layers, K) split-half cosine of the pooled estimates."""
        return self.split_half_cosine

    def temporal_reach(self) -> torch.Tensor:
        """(n_layers, K) unbiased Frobenius norm of the pooled estimates."""
        return self.reach_sq.clamp_min(0.0).sqrt()

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save experts (fp16), pooled (fp32), routers, weights and provenance."""
        if not self.pooled:
            raise RuntimeError("Nothing to save: lens is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": _FORMAT,
                "version": _VERSION,
                "config": asdict(self.config),
                "experts": {layer: e.half().clone() for layer, e in self.experts.items()},
                "pooled": {layer: p.clone() for layer, p in self.pooled.items()},
                "expert_counts": self.expert_counts,
                "fallback": self.fallback,
                "weights": self.weights,
                "routers": {layer: r.to_state() for layer, r in self.routers.items()},
                "reach_sq": self.reach_sq,
                "split_half_cosine": self.split_half_cosine,
                "n_sequences": self.n_sequences,
                "model_id": self.model.model_id,
                "d_model": self.d_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "method": "offset-resolved J++ lens (Ayonrinde & Lindsey 2026; Lindsey et al. 2026)",
            },
            path,
        )
        logger.info("Saved expert offset lens → %s", path)

    @classmethod
    def load(cls, path: str | Path, model: HookedModel) -> "ExpertOffsetLens":
        """Load a lens written by :meth:`save`."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != _FORMAT:
            raise ValueError(f"{path} is not an expert offset lens (format={data.get('format')!r})")
        lens = cls(model, ExpertOffsetConfig(**data["config"]))
        if data["d_model"] != lens.d_model:
            raise ValueError(f"lens d_model {data['d_model']} != model {lens.d_model}")
        lens.experts = {int(layer): e.float() for layer, e in data["experts"].items()}
        lens.pooled = {int(layer): p.float() for layer, p in data["pooled"].items()}
        lens.expert_counts = {int(layer): c for layer, c in data["expert_counts"].items()}
        lens.fallback = {int(layer): f for layer, f in data["fallback"].items()}
        lens.weights = {int(layer): w for layer, w in data["weights"].items()}
        lens.routers = {
            int(layer): ActivationRouter.from_state(s, "cpu")
            for layer, s in data["routers"].items()
        }
        lens.reach_sq = data["reach_sq"]
        lens.split_half_cosine = data["split_half_cosine"]
        lens.n_sequences = int(data["n_sequences"])
        return lens

    # ── internals ───────────────────────────────────────────────────────────

    def _accumulate_sequence(
        self,
        ids: torch.Tensor,
        sums: torch.Tensor,
        counts: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """All backward passes for one sequence, accumulated into the CPU sums."""
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected token ids of shape (1, T), got {tuple(ids.shape)}")
        cfg, dev, d = self.config, self.device, self.d_model
        seq_len, n_layers, n_offsets = ids.shape[1], len(self.layer_indices), len(self.offsets)
        kept = kept_position_mask(seq_len, cfg.skip_first).to(dev)
        if not bool(kept.any()):
            return
        stride, n_rows = cfg.comb_stride, cfg.rows_per_pass
        positions = torch.arange(seq_len, device=dev)

        captured, h_final = self._forward_replicated(ids.to(dev), n_rows)
        sources = [captured[layer] for layer in self.layer_indices]
        experts_by_layer = torch.stack(
            [
                self.routers[layer].assign(captured[layer][0].detach().float().cpu()).to(dev)
                for layer in self.layer_indices
            ]
        )  # (L, T)

        signs = torch.randint(0, 2, (cfg.n_sign_draws, seq_len), generator=generator, device=dev)
        signs = signs.float().mul_(2).sub_(1)
        phase = torch.randint(stride, (1,), generator=generator, device=dev)
        offsets_t = torch.tensor(self.offsets, device=dev)

        valid_pairs = kept.unsqueeze(0) & (
            positions.unsqueeze(0) + offsets_t.unsqueeze(1) < seq_len - 1
        )
        expert_onehot = F.one_hot(experts_by_layer, cfg.num_experts).float()  # (L, T, E)
        per_expert = torch.einsum("kt,lte->kle", valid_pairs.float(), expert_onehot).long().cpu()
        counts += per_expert.unsqueeze(0) * (cfg.n_sign_draws // 2)

        n_slices = math.ceil(d / n_rows)
        total_passes = n_slices * cfg.n_sign_draws * stride
        pass_idx = 0
        row_idx = torch.arange(n_rows, device=dev)
        buffer = torch.zeros((2, n_layers, n_offsets, cfg.num_experts, n_rows, d), device=dev)
        for dim_start in range(0, d, n_rows):
            n_b = min(n_rows, d - dim_start)
            buffer.zero_()
            for draw in range(cfg.n_sign_draws):
                for residue in range(stride):
                    tagged = kept & ((positions + phase) % stride == residue)
                    tag_pos = tagged.nonzero().squeeze(1)
                    pass_idx += 1
                    if tag_pos.numel() == 0:
                        continue
                    cotangent = torch.zeros_like(h_final)
                    cotangent[
                        row_idx[:n_b, None], tag_pos[None, :], dim_start + row_idx[:n_b, None]
                    ] = signs[draw, tag_pos].to(h_final.dtype)
                    grads = torch.autograd.grad(
                        h_final,
                        sources,
                        grad_outputs=cotangent,
                        retain_graph=pass_idx < total_passes,
                    )
                    self._add_rows(
                        grads,
                        n_b,
                        draw % 2,
                        tagged,
                        signs[draw],
                        valid_pairs,
                        expert_onehot,
                        buffer,
                    )
            sums[..., dim_start : dim_start + n_b, :] += buffer[..., :n_b, :].cpu()
        del captured, h_final, sources

    def _add_rows(
        self,
        grads: Sequence[torch.Tensor],
        n_b: int,
        half: int,
        tagged: torch.Tensor,
        signs: torch.Tensor,
        valid_pairs: torch.Tensor,
        expert_onehot: torch.Tensor,
        buffer: torch.Tensor,
    ) -> None:
        """Route one backward pass's rows into the per-(half, layer, offset, expert) buffer."""
        stacked = torch.stack([g[:n_b].float() for g in grads])  # (L, B, T, d)
        seq_len = stacked.shape[2]
        for k, delta in enumerate(self.offsets):
            target_tagged = torch.zeros_like(tagged)
            target_tagged[: seq_len - delta] = tagged[delta:]
            src = (target_tagged & valid_pairs[k]).nonzero().squeeze(1)
            if src.numel() == 0:
                continue
            rows = stacked[:, :, src, :] * signs[src + delta].view(1, 1, -1, 1)  # (L, B, P, d)
            buffer[half, :, k, :, :n_b, :] += torch.einsum(
                "lpe,lbpd->lebd", expert_onehot[:, src], rows
            )

    def _forward_replicated(
        self, ids: torch.Tensor, n_rows: int
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Forward B replicas with autograd on (and LRP rules if configured)."""
        captured: Dict[int, torch.Tensor] = {}
        needed = sorted(set(self.layer_indices) | {self.final_layer})

        def make_hook(idx: int):
            def hook(_m: torch.nn.Module, _i: tuple, output: object) -> None:
                captured[idx] = output[0] if isinstance(output, tuple) else output

            return hook

        handles = [self.model.layers[i].register_forward_hook(make_hook(i)) for i in needed]
        rules = LrpRules() if self.config.lrp else LrpRules(False, False, False)
        try:
            with torch.enable_grad(), lrp_rules(self.model.model, rules):
                self.model.model.model(input_ids=ids.expand(n_rows, -1), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        h_final = captured[self.final_layer]
        if not h_final.requires_grad:
            raise RuntimeError("final residual does not require grad; enable parameter grads")
        return captured, h_final

    def _finalise(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        """Turn per-half sums into experts, pooled matrices, reach and reliability."""
        cfg = self.config
        n_layers, n_offsets = len(self.layer_indices), len(self.offsets)
        half_counts = counts.permute(0, 2, 1, 3).float()  # (2, L, K, E)
        inner = torch.zeros(n_layers, n_offsets)
        norm_a = torch.zeros(n_layers, n_offsets)
        norm_b = torch.zeros(n_layers, n_offsets)
        for j, layer in enumerate(self.layer_indices):
            pooled_a = sums[0, j].sum(1) / half_counts[0, j].sum(-1).clamp_min(1).view(-1, 1, 1)
            pooled_b = sums[1, j].sum(1) / half_counts[1, j].sum(-1).clamp_min(1).view(-1, 1, 1)
            inner[j] = (pooled_a * pooled_b).sum((-2, -1))
            norm_a[j] = torch.linalg.matrix_norm(pooled_a)
            norm_b[j] = torch.linalg.matrix_norm(pooled_b)

            total = sums[0, j] + sums[1, j]  # (K, E, d, d)
            n = half_counts[0, j] + half_counts[1, j]  # (K, E)
            pooled = total.sum(1) / n.sum(-1).clamp_min(1).view(-1, 1, 1)
            experts = total / n.clamp_min(1).view(n_offsets, -1, 1, 1)
            fallback = n < cfg.min_kept_positions
            experts[fallback] = pooled.unsqueeze(1).expand_as(experts)[fallback]

            self.pooled[layer] = pooled
            self.experts[layer] = experts
            self.expert_counts[layer] = n.long()
            self.fallback[layer] = fallback
            if bool(fallback.any()):
                logger.warning(
                    "layer %d: %d expert(s) fell back to pooled", layer, int(fallback.sum())
                )
        self.reach_sq = inner
        self.split_half_cosine = inner / (norm_a * norm_b).clamp_min(1e-30)
        self._matrix_cache = {}
