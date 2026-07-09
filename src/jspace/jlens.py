"""
Jacobian Lens (J-lens) — paper-faithful implementation.

Paper (Lindsey et al., 2026):
    J_l = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]

J_l is a (d_model x d_model) matrix: the averaged linearised map from
an intermediate hidden state to the final transformer layer output.

Readout (paper formula):
    lens(h_l) = softmax(W_U  norm(J_l @ h_l))

We approximate J_l using the Hutchinson matrix estimator:
for each Rademacher vector v ~ {+1,-1}^d_model,
    grad(h_final_{t'} . v, h_l_t) = J_l^T @ v

so:  J_l = E_v [ outer(v, J_l^T @ v) ]  (Rademacher identity, E[v v^T] = I)

Implementation detail:
    Each layer requires a SEPARATE forward pass so that h_final is computed
    from h_l through the live computation graph.  Hooking multiple layers
    simultaneously breaks the graph at each intermediate layer, making
    backward through the whole stack impossible.

    Cost on H100 80GB:
        n_layers × n_prompts forward passes  (~35ms each for 7B)
        + n_proj backward passes per (layer, prompt)  (~15ms each)
    For 32 layers × 35 prompts × 16 projections ≈ 5 minutes.
    For 32 layers × 100 prompts × 16 projections ≈ 15 minutes.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jspace.model import HookedModel

logger = logging.getLogger(__name__)


class JacobianLens:
    """
    Computes and caches the averaged Jacobian readout matrices (J-space).

    Parameters
    ----------
    model : HookedModel
    layer_indices : list[int] | None
        Layers to compute J for.  None = all layers.
    n_proj : int
        Hutchinson projections per (prompt, position) pair.
        16 is fast & good for workspace detection; 32 matches paper quality.
    top_k_vocab : int
        Vocabulary tokens to return per readout call.
    n_positions : int
        Number of tail positions to use as source per prompt
        (paper averages over all t; 4 is a good practical compromise).
    """

    def __init__(
        self,
        model: HookedModel,
        layer_indices: Optional[List[int]] = None,
        n_proj: int = 16,
        top_k_vocab: int = 20,
        n_positions: int = 4,
    ) -> None:
        self.model = model
        self.layer_indices: List[int] = (
            layer_indices if layer_indices is not None else list(range(model.num_layers))
        )
        self.n_proj = n_proj
        self.top_k_vocab = top_k_vocab
        self.n_positions = n_positions

        # {layer_idx: (sum_J_matrix (d,d), count)}
        self._accum: Dict[int, Tuple[torch.Tensor, int]] = {}

        # {layer_idx: J_l (d_model, d_model)}
        self.avg_jacobians: Dict[int, torch.Tensor] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        prompts: List[str],
        max_length: int = 128,
        show_progress: bool = True,
    ) -> "JacobianLens":
        """
        Compute averaged J-lens matrices over a corpus of prompts.

        Parameters
        ----------
        prompts : list[str]
            Diverse corpus.  100–1000 prompts (paper uses 1000).
        max_length : int
        show_progress : bool
        """
        d_model = self.model.model.config.hidden_size
        n_layers = len(self.layer_indices)
        logger.info(
            "Fitting J-lens: %d prompts x %d layers x %d projections  "
            "(d_model=%d, n_positions=%d)",
            len(prompts), n_layers, self.n_proj, d_model, self.n_positions,
        )
        self._accum.clear()

        pbar = tqdm(
            total=len(prompts) * n_layers,
            desc="J-lens fit",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for prompt in prompts:
            enc = self.model.tokenize([prompt], max_length=max_length)
            seq_len = int(enc["attention_mask"][0].sum()) if "attention_mask" in enc else enc["input_ids"].shape[1]
            tgt_pos = seq_len - 1
            src_positions = list(range(max(0, seq_len - self.n_positions), seq_len))

            for layer_idx in self.layer_indices:
                try:
                    self._process_one_layer(enc, layer_idx, src_positions, tgt_pos)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM layer %d prompt %s (skipped)", layer_idx, prompt[:40])
                    torch.cuda.empty_cache()
                    gc.collect()
                pbar.update(1)

        pbar.close()

        for layer_idx, (sum_j, count) in self._accum.items():
            self.avg_jacobians[layer_idx] = sum_j / max(count, 1)

        logger.info("J-lens fit complete: %d layers, J shape: (%d, %d)",
                    len(self.avg_jacobians), d_model, d_model)
        return self

    def readout(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Paper readout formula: lens(h_l) = softmax(W_U norm(J_l @ h_l))

        Parameters
        ----------
        hidden : (d_model,) hidden state at layer layer_idx
        layer_idx : int
        k : int top-k tokens

        Returns
        -------
        list of (token_str, prob) sorted descending
        """
        logits = self.readout_logits(hidden, layer_idx)
        return self.model.top_tokens(logits, k=k)

    def readout_logits(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Return full logit vector via paper readout: W_U(norm(J_l @ h_l)).

        The model's unembed() applies final layernorm then lm_head,
        which implements norm(.) then W_U in sequence — matching the formula.
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens.")
        J = self.avg_jacobians[layer_idx].to(self.model.device)   # (d, d)
        h = hidden.float().to(self.model.device)                   # (d,)
        Jh = J @ h                                                 # (d,)
        # unembed applies layernorm + W_U — the paper's norm(.) + W_U
        return self.model.unembed(Jh.unsqueeze(0).unsqueeze(0)).squeeze().detach()

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "avg_jacobians": {k: v.cpu() for k, v in self.avg_jacobians.items()},
                "layer_indices": self.layer_indices,
                "model_id": self.model.model_id,
                "n_proj": self.n_proj,
                "n_positions": self.n_positions,
            },
            path,
        )
        logger.info("Saved J-lens (%d layers) to %s", len(self.avg_jacobians), path)

    @classmethod
    def load(cls, path: str | Path, model: HookedModel) -> "JacobianLens":
        data = torch.load(path, map_location="cpu", weights_only=True)
        jl = cls(
            model,
            layer_indices=data["layer_indices"],
            n_proj=data.get("n_proj", 16),
            n_positions=data.get("n_positions", 4),
        )
        jl.avg_jacobians = {k: v for k, v in data["avg_jacobians"].items()}
        logger.info("Loaded J-lens from %s (%d layers)", path, len(jl.avg_jacobians))
        return jl

    # ── internals ───────────────────────────────────────────────────────────

    def _process_one_layer(
        self,
        enc: Dict[str, torch.Tensor],
        layer_idx: int,
        src_positions: List[int],
        tgt_pos: int,
    ) -> None:
        """
        Single forward pass with only layer_idx hooked for gradient.

        The hook makes h_l a leaf tensor (requires_grad=True) while the
        forward continues downstream — h_final is thus a function of h_l
        and we can backprop through it.

        We capture h_final at the last transformer block output (before
        lm_head/norm) via a non-detaching hook on the final layer.
        """
        d_model = self.model.model.config.hidden_size
        device = self.model.device

        h_l_store: Dict[int, torch.Tensor] = {}
        h_final_store: Dict[str, torch.Tensor] = {}

        # Hook layer_idx: make output a grad leaf, re-inject into graph
        def grad_hook(module, _input, output, _idx=layer_idx):
            hs = output[0] if isinstance(output, tuple) else output
            clone = hs.clone().requires_grad_(True)
            h_l_store[_idx] = clone
            return (clone,) + output[1:] if isinstance(output, tuple) else clone

        # Hook final layer: store h_final WITHOUT detaching so grad flows through
        final_idx = self.model.num_layers - 1

        def final_hook(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            h_final_store["h"] = hs   # keep in graph — no detach

        h1 = self.model._layers[layer_idx].register_forward_hook(grad_hook)
        # only register final hook if layer_idx is not the last layer
        h2 = None
        if layer_idx < final_idx:
            h2 = self.model._layers[final_idx].register_forward_hook(final_hook)

        try:
            with torch.enable_grad():
                _ = self.model.model(**enc)
        finally:
            h1.remove()
            if h2 is not None:
                h2.remove()

        if layer_idx not in h_l_store:
            return

        # if layer_idx is the last layer, use it as its own "final"
        if layer_idx == final_idx:
            h_final_store["h"] = h_l_store[layer_idx]

        if "h" not in h_final_store:
            return

        h_final_seq = h_final_store["h"]  # (1, seq, d_model) — grad-enabled

        # Hutchinson estimator over source positions
        J_sum = torch.zeros(d_model, d_model, dtype=torch.float32, device="cpu")
        n_accumulated = 0

        for src_pos in src_positions:
            if src_pos >= h_l_store[layer_idx].shape[1]:
                continue
            h_src = h_l_store[layer_idx][0, src_pos]   # (d_model,) leaf with grad

            # Rademacher vectors: (n_proj, d_model)
            vs = (torch.randint(0, 2, (self.n_proj, d_model), device=device).float() * 2 - 1)

            for i in range(self.n_proj):
                v = vs[i]  # (d_model,)
                # scalar = h_final_{tgt_pos} . v
                # grad w.r.t. h_src = J_l^T @ v
                scalar = (h_final_seq[0, tgt_pos].float() * v).sum()
                grad = torch.autograd.grad(
                    scalar,
                    h_src,
                    retain_graph=(i < self.n_proj - 1 or src_pos != src_positions[-1]),
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is None:
                    continue
                g = grad.detach().float().cpu()   # J_l^T @ v, shape (d_model,)
                # Hutchinson: E_v[outer(v, J^T v)] = E_v[v v^T J^T]^T ... see docs
                # correct accumulation: outer(v, g) where g = J^T v  → estimates J
                J_sum += torch.outer(v.cpu(), g)
                n_accumulated += 1

        if n_accumulated == 0:
            return

        J_estimate = J_sum / n_accumulated

        if layer_idx not in self._accum:
            self._accum[layer_idx] = (J_estimate, 1)
        else:
            s, c = self._accum[layer_idx]
            self._accum[layer_idx] = (s + J_estimate, c + 1)

        # free graph memory
        del h_final_store, h_l_store
        torch.cuda.empty_cache()
        gc.collect()
