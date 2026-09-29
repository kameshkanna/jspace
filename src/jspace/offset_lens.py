"""
Offset-resolved (multi-token) Jacobian lens.

Extends the J-lens of Lindsey et al. (2026), which averages the Jacobian over
every future target position,

    J_l = E_{t, t'>=t, prompt} [ ∂h_final_{t'} / ∂h_l_t ],

and therefore collapses *when* a residual-stream feature is used.  This module
keeps the offset Δ = t' − t as an explicit axis:

    J_l^(Δ)        = E_{prompt, t} [ ∂h_final_{t+Δ} / ∂h_l_t ]        Δ ∈ offsets
    lens_Δ(h_l_t)  = softmax(W_U · norm(J_l^(Δ) h_l_t))              → token t+Δ+1

so one hidden state decodes to a short latent string (one distribution per
offset) instead of a single next-token distribution.

Estimator — comb-tagged Hutchinson
----------------------------------
Every target position t' receives its own Rademacher probe v_{t'} and one VJP is
taken per projection:

    g_t = Σ_{t'} J(t, t')ᵀ v_{t'}          (one backward pass → all layers, all t)

Independent probes give E[v_{t+Δ} g_tᵀ] = J(t, t+Δ), so every offset comes out of
the same backward passes.  The untagged cross terms add variance, and the Δ=0
term dominates them, so tagging every position buries Δ>0 in Δ=0 noise.  Comb
tagging removes that: projection i tags only targets with t' ≡ r_i (mod M) where
M > max(offsets).  For offset Δ the remaining cross terms lie at offsets Δ + kM ≥ M,
which excludes Δ = 0.  ``tests/test_offset_lens.py`` checks the estimate against
the exact Jacobian.

Split-half reach
----------------
‖Ĵ‖_F of a noisy estimate is biased upward by the estimator variance, which is
largest exactly where the signal is smallest (large Δ).  Projections are paired
(both members share a comb residue) and routed to two independent halves A and B,
so E⟨Ĵ_A, Ĵ_B⟩_F = ‖J‖_F² without bias, and cos(Ĵ_A, Ĵ_B) is a per-(layer, Δ)
reliability score that needs no ground truth.

Packing
-------
Short corpus prompts leave few (t, t+Δ) pairs at large Δ, so by default the
corpus is concatenated into one token stream and cut into ``max_length``
sequences, as in pretraining.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from jspace.model import HookedModel
from jspace.utils import free_device_memory

logger = logging.getLogger(__name__)

_FORMAT = "jspace.offset_lens"
_VERSION = "1.0"


@dataclass(frozen=True)
class OffsetLensConfig:
    """
    Hyper-parameters for fitting offset-resolved J-lens matrices.

    Attributes
    ----------
    offsets : tuple[int, ...]
        Strictly increasing, non-negative target offsets Δ.
    layer_indices : tuple[int, ...] | None
        Source layers to fit.  None fits every layer.
    n_proj : int
        Rademacher projections (backward passes) per sequence.  Must be even:
        projections are paired across the two split halves.
    max_length : int
        Token truncation length, and packed sequence length.
    stride : int | None
        Comb stride M.  Must exceed ``max(offsets)``; None uses ``max(offsets) + 1``.
    seed : int
        Seed for the probe generator.
    pack : bool
        Concatenate corpus prompts into ``max_length`` sequences before fitting.
    pack_separator : str
        Text inserted between packed prompts.
    """

    offsets: Tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)
    layer_indices: Optional[Tuple[int, ...]] = None
    n_proj: int = 32
    max_length: int = 128
    stride: Optional[int] = None
    seed: int = 0
    pack: bool = True
    pack_separator: str = "\n\n"

    def __post_init__(self) -> None:
        """Normalise sequences to tuples and validate every field."""
        object.__setattr__(self, "offsets", tuple(int(o) for o in self.offsets))
        if self.layer_indices is not None:
            object.__setattr__(
                self, "layer_indices", tuple(int(layer) for layer in self.layer_indices)
            )

        if not self.offsets:
            raise ValueError("offsets must be non-empty")
        if self.offsets[0] < 0:
            raise ValueError(f"offsets must be non-negative, got {self.offsets}")
        if any(b <= a for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError(f"offsets must be strictly increasing, got {self.offsets}")
        if self.layer_indices is not None and len(set(self.layer_indices)) != len(
            self.layer_indices
        ):
            raise ValueError(f"layer_indices contains duplicates: {self.layer_indices}")
        if self.n_proj < 2 or self.n_proj % 2:
            raise ValueError(f"n_proj must be a positive even number, got {self.n_proj}")
        if self.max_length <= self.offsets[-1] + 1:
            raise ValueError(
                f"max_length ({self.max_length}) must exceed "
                f"max(offsets) + 1 ({self.offsets[-1] + 1})"
            )
        if self.stride is not None and self.stride <= self.offsets[-1]:
            raise ValueError(
                f"stride ({self.stride}) must exceed max(offsets) ({self.offsets[-1]}) "
                "or Δ=0 cross terms leak into the offset estimates"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def comb_stride(self) -> int:
        """Effective comb stride M."""
        return self.stride if self.stride is not None else self.offsets[-1] + 1


def pack_prompts(
    prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    min_length: int,
    separator: str = "\n\n",
) -> List[torch.Tensor]:
    """
    Concatenate prompts into one token stream and cut it into fixed-length sequences.

    Each sequence is prefixed with the special tokens the tokenizer adds to an
    empty string (e.g. Llama-3's BOS; nothing for Qwen), so every chunk starts
    the way a normal forward pass would.

    Parameters
    ----------
    prompts : Sequence[str]
    tokenizer : PreTrainedTokenizerBase
    max_length : int
        Length of each packed sequence, prefix included.
    min_length : int
        Trailing chunks shorter than this are dropped.
    separator : str
        Text placed after every prompt.

    Returns
    -------
    list[torch.Tensor]
        (1, T) int64 token-id tensors on CPU, T <= max_length.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    prefix = tokenizer("", add_special_tokens=True)["input_ids"]
    body_len = max_length - len(prefix)
    if body_len < 1:
        raise ValueError(
            f"max_length {max_length} leaves no room after {len(prefix)} prefix tokens"
        )
    separator_ids = tokenizer(separator, add_special_tokens=False)["input_ids"]
    bodies = tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
    stream = torch.tensor(
        list(chain.from_iterable(ids + separator_ids for ids in bodies)), dtype=torch.long
    )
    prefix_ids = torch.tensor(prefix, dtype=torch.long)
    return [
        torch.cat([prefix_ids, chunk]).unsqueeze(0)
        for chunk in stream.split(body_len)
        if chunk.numel() + prefix_ids.numel() >= min_length
    ]


def accumulate_offset_outer_products(
    probes: torch.Tensor,
    vjps: torch.Tensor,
    tags: torch.Tensor,
    offsets: Sequence[int],
    sums: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """
    Add the Hutchinson outer products for every offset into running sums, in place.

    For offset Δ, every (projection i, source t) pair whose target t+Δ was tagged
    contributes v_{i,t+Δ} g_{l,i,t}ᵀ, an unbiased sample of J_l(t, t+Δ).

    Parameters
    ----------
    probes : torch.Tensor
        (P, T, d_out) comb-masked Rademacher probes on the final-layer residual.
    vjps : torch.Tensor
        (L, P, T, d_in) vector-Jacobian products at each source layer.
    tags : torch.Tensor
        (P, T) bool — which target positions each projection tagged.
    offsets : Sequence[int]
        Offsets Δ, aligned with ``sums`` dim 1.
    sums : torch.Tensor
        (L, K, d_out, d_in) running sums, updated in place.
    counts : torch.Tensor
        (K,) running sample counts, updated in place.
    """
    n_proj, seq_len, d_out = probes.shape
    if tags.shape != (n_proj, seq_len):
        raise ValueError(f"tags shape {tuple(tags.shape)} != {(n_proj, seq_len)}")
    if vjps.shape[1:3] != (n_proj, seq_len):
        raise ValueError(
            f"vjps shape {tuple(vjps.shape)} inconsistent with probes {tuple(probes.shape)}"
        )
    if sums.shape[:2] != (vjps.shape[0], len(offsets)) or sums.shape[2] != d_out:
        raise ValueError(f"sums shape {tuple(sums.shape)} inconsistent with inputs")

    for k, delta in enumerate(offsets):
        if delta >= seq_len:
            continue
        pair_mask = tags[:, delta:]
        v = probes[:, delta:][pair_mask]
        g = vjps[:, :, : seq_len - delta][:, pair_mask]
        sums[:, k] += v.transpose(0, 1).to(g.dtype) @ g
        counts[k] += v.shape[0]


class OffsetJacobianLens:
    """
    Offset-resolved J-lens: one averaged Jacobian per (source layer, offset Δ).

    Attributes
    ----------
    jacobians : dict[int, torch.Tensor]
        layer → (K, d_model, d_model) float32 stack of J_l^(Δ), on the model device.
    corpus_atoms : dict[int, torch.Tensor]
        layer → (n_corpus, K, d_model) float16 per-offset unit-normalised,
        mean-centred J_l^(Δ) h at the last token of each corpus prompt.
        Dictionary for multi-token concept counting.
    corpus_mean : dict[int, torch.Tensor]
        layer → (K, d_model) float32 corpus mean of J_l^(Δ) h, subtracted from
        atoms and queries before pursuit.
    sample_counts : torch.Tensor
        (K,) number of Hutchinson samples behind each offset estimate.
    reach_sq : torch.Tensor
        (n_layers, K) unbiased split-half estimate of ‖J_l^(Δ)‖_F².
    split_half_cosine : torch.Tensor
        (n_layers, K) cosine between the two half-estimates; ~1 means the
        estimate is signal-dominated, ~0 means noise-dominated.

    Parameters
    ----------
    model : HookedModel
    config : OffsetLensConfig
    """

    def __init__(self, model: HookedModel, config: OffsetLensConfig = OffsetLensConfig()) -> None:
        self.model = model
        self.config = config
        self.layer_indices: List[int] = (
            list(config.layer_indices)
            if config.layer_indices is not None
            else list(range(model.num_layers))
        )
        invalid = [layer for layer in self.layer_indices if not 0 <= layer < model.num_layers]
        if invalid:
            raise ValueError(f"layer indices {invalid} outside [0, {model.num_layers})")
        self.offsets: List[int] = list(config.offsets)
        self.d_model: int = int(model.model.config.hidden_size)

        self.jacobians: Dict[int, torch.Tensor] = {}
        self.corpus_atoms: Dict[int, torch.Tensor] = {}
        self.corpus_mean: Dict[int, torch.Tensor] = {}
        self.sample_counts: torch.Tensor = torch.zeros(len(self.offsets), dtype=torch.int64)
        self.reach_sq: torch.Tensor = torch.empty(0)
        self.split_half_cosine: torch.Tensor = torch.empty(0)
        self.n_prompts: int = 0
        self.n_sequences: int = 0

    @property
    def device(self) -> torch.device:
        """Device the model and fitted matrices live on."""
        return torch.device(self.model.device)

    @property
    def final_layer(self) -> int:
        """Index of the last transformer block (the Jacobian target)."""
        return self.model.num_layers - 1

    # ── fitting ─────────────────────────────────────────────────────────────

    def fit(self, prompts: Sequence[str], show_progress: bool = True) -> "OffsetJacobianLens":
        """
        Fit J_l^(Δ) for every layer and offset, then build the corpus atom dictionary.

        Parameters
        ----------
        prompts : Sequence[str]
            Diverse corpus.  Packed into ``max_length`` sequences when
            ``config.pack`` is set, otherwise one unpadded sequence per prompt.
            Corpus atoms always use each prompt on its own.
        show_progress : bool

        Returns
        -------
        OffsetJacobianLens
            ``self``, fitted.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty")
        if self.config.pack:
            sequences = pack_prompts(
                prompts,
                self.model.tokenizer,
                max_length=self.config.max_length,
                min_length=self.offsets[-1] + 2,
                separator=self.config.pack_separator,
            )
            if not sequences:
                raise ValueError("packing produced no sequences long enough for the largest offset")
            logger.info(
                "Packed %d prompts into %d sequences of <= %d tokens",
                len(prompts),
                len(sequences),
                self.config.max_length,
            )
            encodings: Iterable[Mapping[str, torch.Tensor]] = (
                {"input_ids": ids.to(self.device)} for ids in sequences
            )
            total = len(sequences)
        else:
            encodings = (
                self.model.tokenize([prompt], max_length=self.config.max_length)
                for prompt in prompts
            )
            total = len(prompts)
        self.fit_jacobians(encodings, total=total, show_progress=show_progress)
        self.n_prompts = len(prompts)
        self.build_corpus_atoms(prompts, show_progress=show_progress)
        return self

    def fit_jacobians(
        self,
        encodings: Iterable[Mapping[str, torch.Tensor]],
        total: Optional[int] = None,
        show_progress: bool = True,
    ) -> "OffsetJacobianLens":
        """
        Estimate J_l^(Δ) with the comb-tagged Hutchinson estimator.

        Parameters
        ----------
        encodings : Iterable[Mapping[str, torch.Tensor]]
            Single-sequence model inputs, each with ``input_ids`` of shape (1, T)
            and optionally an all-ones ``attention_mask``.
        total : int | None
            Length of ``encodings`` for the progress bar.
        show_progress : bool

        Returns
        -------
        OffsetJacobianLens
            ``self``, with ``jacobians``, ``sample_counts``, ``reach_sq`` and
            ``split_half_cosine`` populated.
        """
        n_layers, n_offsets, d = len(self.layer_indices), len(self.offsets), self.d_model
        logger.info(
            "Offset J-lens: %d layers × %d offsets %s, n_proj=%d, stride=%d, d_model=%d "
            "(accumulators %.2f GB)",
            n_layers,
            n_offsets,
            self.offsets,
            self.config.n_proj,
            self.config.comb_stride,
            d,
            2 * n_layers * n_offsets * d * d * 4 / 1e9,
        )
        sums = [
            torch.zeros((n_layers, n_offsets, d, d), dtype=torch.float32, device=self.device)
            for _ in range(2)
        ]
        counts = torch.zeros((2, n_offsets), dtype=torch.int64, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        n_ok = 0
        for enc in tqdm(
            encodings,
            total=total,
            desc="offset J-lens: Jacobians",
            disable=not show_progress,
            dynamic_ncols=True,
        ):
            try:
                self._accumulate_sequence(enc, sums, counts, generator)
                n_ok += 1
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM on sequence of length %d — skipped", enc["input_ids"].shape[1])
                free_device_memory(self.device)

        if n_ok == 0:
            raise RuntimeError("No sequences were processed")
        empty = [delta for delta, c in zip(self.offsets, counts[0].tolist()) if c == 0]
        if empty:
            raise RuntimeError(f"No samples for offsets {empty}; sequences are too short")

        half_n = counts.to(torch.float32).view(2, n_offsets, 1, 1)
        inner = torch.empty((n_layers, n_offsets), dtype=torch.float32, device=self.device)
        norm_a = torch.empty_like(inner)
        norm_b = torch.empty_like(inner)
        # One layer at a time bounds peak memory to two (K, d, d) temporaries.
        for i in range(n_layers):
            est_a = sums[0][i] / half_n[0]
            est_b = sums[1][i] / half_n[1]
            inner[i] = (est_a * est_b).sum(dim=(-2, -1))
            norm_a[i] = torch.linalg.matrix_norm(est_a)
            norm_b[i] = torch.linalg.matrix_norm(est_b)
            del est_a, est_b

        total = counts.sum(0)
        jacobians = sums[0].add_(sums[1])
        del sums
        jacobians.div_(total.to(torch.float32).view(1, n_offsets, 1, 1))

        self.jacobians = dict(zip(self.layer_indices, jacobians.unbind(0)))
        self.sample_counts = total.cpu()
        self.reach_sq = inner.cpu()
        self.split_half_cosine = (inner / (norm_a * norm_b).clamp_min(1e-30)).cpu()
        self.n_sequences = n_ok
        free_device_memory(self.device)
        logger.info(
            "Fitted %d sequences; samples per offset: %s",
            n_ok,
            dict(zip(self.offsets, self.sample_counts.tolist())),
        )
        return self

    def build_corpus_atoms(
        self,
        prompts: Sequence[str],
        show_progress: bool = True,
    ) -> "OffsetJacobianLens":
        """
        Project each corpus prompt's last-token state through every J_l^(Δ).

        Parameters
        ----------
        prompts : Sequence[str]
        show_progress : bool

        Returns
        -------
        OffsetJacobianLens
            ``self``, with ``corpus_atoms`` populated.
        """
        if not self.jacobians:
            raise RuntimeError("Fit Jacobians before building corpus atoms")
        last_states = torch.empty(
            (len(prompts), len(self.layer_indices), self.d_model),
            dtype=torch.float32,
            device=self.device,
        )
        kept = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for n, prompt in enumerate(
                tqdm(
                    prompts,
                    desc="offset J-lens: corpus atoms",
                    disable=not show_progress,
                    dynamic_ncols=True,
                )
            ):
                enc = self.model.tokenize([prompt], max_length=self.config.max_length)
                try:
                    with self.model.capture_residuals(self.layer_indices) as store:
                        self.model.model(**enc, use_cache=False)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM on corpus prompt %.40s — skipped", prompt)
                    free_device_memory(self.device)
                    continue
                last_states[n] = torch.stack([store[layer][0, -1] for layer in self.layer_indices])
                kept[n] = True

            last_states = last_states[kept]
            self.corpus_atoms, self.corpus_mean = {}, {}
            for j, layer in enumerate(self.layer_indices):
                projected = torch.einsum("koi,ni->nko", self.jacobians[layer], last_states[:, j])
                # J-space vectors share a dominant common direction (mean pairwise cosine
                # 0.4-0.8 on Qwen2.5-0.5B); without centring, nonnegative pursuit spends one
                # atom on it and then stalls because every atom has a negative dot with the
                # residual.
                mean = projected.mean(0)
                self.corpus_mean[layer] = mean
                self.corpus_atoms[layer] = F.normalize(projected - mean, dim=-1).to(torch.float16)
        free_device_memory(self.device)
        logger.info(
            "Corpus atoms: %d prompts × %d layers", last_states.shape[0], len(self.layer_indices)
        )
        return self

    # ── readout ─────────────────────────────────────────────────────────────

    def project(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply every offset Jacobian of a layer: J_l^(Δ) h.

        Parameters
        ----------
        hidden : torch.Tensor
            (..., d_model) hidden states at ``layer_idx``.
        layer_idx : int

        Returns
        -------
        torch.Tensor
            (K, ..., d_model) float32 on the lens device.
        """
        jacobian = self._jacobian(layer_idx)
        if hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden last dim {hidden.shape[-1]} != d_model {self.d_model}")
        h = hidden.to(device=jacobian.device, dtype=torch.float32)
        return torch.einsum("koi,...i->k...o", jacobian, h)

    def readout_logits(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Offset-resolved readout logits W_U · norm(J_l^(Δ) h).

        Parameters
        ----------
        hidden : torch.Tensor
            (..., d_model) hidden states at ``layer_idx``.
        layer_idx : int

        Returns
        -------
        torch.Tensor
            (K, ..., vocab) detached logits in model dtype.
        """
        return self.model.unembed(self.project(hidden, layer_idx))

    def temporal_reach(self) -> torch.Tensor:
        """
        Unbiased Frobenius norm of every J_l^(Δ), from the split-half inner product.

        Entries whose split-half inner product is negative (noise-dominated) are 0.

        Returns
        -------
        torch.Tensor
            (n_layers, K) float32 on CPU, rows ordered as ``layer_indices``.
        """
        if self.reach_sq.numel() == 0:
            raise RuntimeError("Lens is not fitted")
        return self.reach_sq.clamp_min(0.0).sqrt()

    def reliability(self) -> torch.Tensor:
        """
        Split-half cosine between independent half-estimates of each J_l^(Δ).

        Returns
        -------
        torch.Tensor
            (n_layers, K) float32 on CPU in [-1, 1].
        """
        if self.split_half_cosine.numel() == 0:
            raise RuntimeError("Lens is not fitted")
        return self.split_half_cosine

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path, jacobian_dtype: torch.dtype = torch.float32) -> None:
        """
        Serialise fitted matrices, atoms, and provenance metadata.

        Parameters
        ----------
        path : str | Path
        jacobian_dtype : torch.dtype
            Storage dtype for the Jacobian stacks (bfloat16 halves file size).
        """
        if not self.jacobians:
            raise RuntimeError("Nothing to save: lens is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": _FORMAT,
                "version": _VERSION,
                "config": asdict(self.config),
                "layer_indices": self.layer_indices,
                "offsets": self.offsets,
                "jacobians": {
                    layer: j.to(device="cpu", dtype=jacobian_dtype).clone()
                    for layer, j in self.jacobians.items()
                },
                "corpus_atoms": {layer: a.cpu().clone() for layer, a in self.corpus_atoms.items()},
                "corpus_mean": {layer: m.cpu().clone() for layer, m in self.corpus_mean.items()},
                "sample_counts": self.sample_counts.cpu(),
                "reach_sq": self.reach_sq.cpu(),
                "split_half_cosine": self.split_half_cosine.cpu(),
                "n_prompts": self.n_prompts,
                "n_sequences": self.n_sequences,
                "model_id": self.model.model_id,
                "d_model": self.d_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "https://github.com/kameshkanna/jspace",
                "method": "offset-resolved extension of the J-lens (Lindsey et al., 2026)",
                "license": "MIT",
            },
            path,
        )
        logger.info(
            "Saved offset J-lens — model=%s layers=%d offsets=%s → %s",
            self.model.model_id,
            len(self.jacobians),
            self.offsets,
            path,
        )

    @classmethod
    def load(cls, path: str | Path, model: HookedModel) -> "OffsetJacobianLens":
        """
        Load a lens saved by :meth:`save` and move it to the model's device.

        Parameters
        ----------
        path : str | Path
        model : HookedModel

        Returns
        -------
        OffsetJacobianLens
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != _FORMAT:
            raise ValueError(f"{path} is not an offset J-lens file (format={data.get('format')!r})")
        lens = cls(model, OffsetLensConfig(**data["config"]))
        if data["d_model"] != lens.d_model:
            raise ValueError(f"Lens d_model {data['d_model']} != model hidden size {lens.d_model}")
        if data["model_id"] != model.model_id:
            logger.warning(
                "Lens was fitted on %s but model is %s", data["model_id"], model.model_id
            )
        lens.jacobians = {
            int(layer): j.to(device=lens.device, dtype=torch.float32)
            for layer, j in data["jacobians"].items()
        }
        lens.corpus_atoms = {
            int(layer): a.to(lens.device) for layer, a in data["corpus_atoms"].items()
        }
        lens.corpus_mean = {
            int(layer): m.to(lens.device) for layer, m in data["corpus_mean"].items()
        }
        lens.sample_counts = data["sample_counts"]
        lens.reach_sq = data["reach_sq"]
        lens.split_half_cosine = data["split_half_cosine"]
        lens.n_prompts = int(data["n_prompts"])
        lens.n_sequences = int(data["n_sequences"])
        logger.info(
            "Loaded offset J-lens from %s (%d layers, offsets=%s)",
            path,
            len(lens.jacobians),
            lens.offsets,
        )
        return lens

    # ── internals ───────────────────────────────────────────────────────────

    def _jacobian(self, layer_idx: int) -> torch.Tensor:
        """Return the (K, d, d) Jacobian stack for a fitted layer."""
        if layer_idx not in self.jacobians:
            raise KeyError(
                f"Layer {layer_idx} not in fitted offset J-lens {sorted(self.jacobians)}"
            )
        return self.jacobians[layer_idx]

    def _accumulate_sequence(
        self,
        enc: Mapping[str, torch.Tensor],
        sums: Sequence[torch.Tensor],
        counts: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """
        One forward pass and ``n_proj`` backward passes for a single sequence.

        Even projections accumulate into ``sums[0]`` / ``counts[0]`` and odd ones
        into ``sums[1]`` / ``counts[1]``.
        """
        input_ids = enc["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"expected input_ids of shape (1, T), got {tuple(input_ids.shape)}")
        mask = enc.get("attention_mask")
        if mask is not None and not bool(mask.all()):
            raise ValueError("padded sequences are not supported; tokenize prompts one at a time")
        seq_len = input_ids.shape[1]

        captured = self._forward_with_graph(enc)
        h_final = captured[self.final_layer]
        sources = [captured[layer] for layer in self.layer_indices]
        if not h_final.requires_grad:
            raise RuntimeError("final-layer residual does not require grad; enable parameter grads")

        probes, tags = self._comb_probes(seq_len, generator)
        vjps = torch.empty(
            (len(sources), self.config.n_proj, seq_len, self.d_model),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(self.config.n_proj):
            grads = torch.autograd.grad(
                h_final,
                sources,
                grad_outputs=probes[i].unsqueeze(0).to(h_final.dtype),
                retain_graph=i < self.config.n_proj - 1,
            )
            vjps[:, i] = torch.stack(grads)[:, 0].float()
        del captured, h_final, sources

        for half in (0, 1):
            accumulate_offset_outer_products(
                probes[half::2],
                vjps[:, half::2],
                tags[half::2],
                self.offsets,
                sums[half],
                counts[half],
            )

    def _forward_with_graph(self, enc: Mapping[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Run the model with autograd on and return in-graph block outputs by layer."""
        captured: Dict[int, torch.Tensor] = {}
        needed = sorted(set(self.layer_indices) | {self.final_layer})

        def make_hook(idx: int):
            def hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
                captured[idx] = output[0] if isinstance(output, tuple) else output

            return hook

        handles = [self.model.layers[i].register_forward_hook(make_hook(i)) for i in needed]
        try:
            with torch.enable_grad():
                self.model.model(**enc, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        missing = [i for i in needed if i not in captured]
        if missing:
            raise RuntimeError(f"Hooks did not fire for layers {missing}")
        return captured

    def _comb_probes(
        self,
        seq_len: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw comb-masked Rademacher probes.

        Projections 2j and 2j+1 share a residue, so both split halves sample the
        same (position, offset) pairs with independent probes.

        Returns
        -------
        probes : torch.Tensor
            (n_proj, seq_len, d_model) float32, zero on untagged positions.
        tags : torch.Tensor
            (n_proj, seq_len) bool, True where projection i tags position t.
        """
        n_proj, stride = self.config.n_proj, self.config.comb_stride
        probes = (
            torch.empty((n_proj, seq_len, self.d_model), dtype=torch.float32, device=self.device)
            .bernoulli_(0.5, generator=generator)
            .mul_(2.0)
            .sub_(1.0)
        )
        phase = torch.randint(stride, (1,), generator=generator, device=self.device)
        residue = (torch.arange(n_proj, device=self.device) // 2 + phase) % stride
        tags = (torch.arange(seq_len, device=self.device) % stride).unsqueeze(
            0
        ) == residue.unsqueeze(1)
        probes.mul_(tags.unsqueeze(-1))
        return probes, tags
