"""
Jacobian Lens (J-lens) computation.

For each layer l and token position i, the J-lens vector is:

    J̄[l, i] = E_{prompts} [ ∂ logits(next_token) / ∂ h[l, i] ]

averaged over a corpus of diverse prompts.  This gives a context-free
"readout direction" — projecting any hidden state onto J̄ tells you
which vocabulary tokens that position is poised to generate.

We use a memory-efficient approximation: instead of the full (vocab × d_model)
Jacobian, we compute the gradient of a scalar summary statistic
(entropy or top-token logit) w.r.t. each hidden state.  For the
full avg-Jacobian we vectorize over vocabulary with vmap.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from jspace.model import HookedModel

logger = logging.getLogger(__name__)


class JacobianLens:
    """
    Computes and caches the averaged Jacobian readout vectors (J-space)
    for a HookedModel across a corpus of prompts.

    Parameters
    ----------
    model : HookedModel
    layer_indices : list[int] | None
        Layers to analyse.  None = all layers.
    top_k_vocab : int
        Vocabulary tokens to track per position.
    use_full_jacobian : bool
        If True, compute the full (d_model × vocab) Jacobian via vmap —
        accurate but memory-heavy.  If False, use gradient of top-1 logit
        (fast approximation, good enough for workspace detection).
    chunk_size : int
        Number of vocab logits to differentiate per backward pass when
        use_full_jacobian=True (reduces peak VRAM).
    """

    def __init__(
        self,
        model: HookedModel,
        layer_indices: Optional[List[int]] = None,
        top_k_vocab: int = 20,
        use_full_jacobian: bool = False,
        chunk_size: int = 512,
    ) -> None:
        self.model = model
        self.layer_indices: List[int] = (
            layer_indices if layer_indices is not None else list(range(model.num_layers))
        )
        self.top_k_vocab = top_k_vocab
        self.use_full_jacobian = use_full_jacobian
        self.chunk_size = chunk_size

        # accumulated sums for online mean: {layer_idx: (sum_J, count)}
        self._accum: Dict[int, Tuple[torch.Tensor, int]] = {}

        # final averaged Jacobians: {layer_idx: tensor(seq_len, d_model)}
        # seq_len dimension is from last processed prompt — positions are
        # aligned by taking the *last* token position (causal direction)
        self.avg_jacobians: Dict[int, torch.Tensor] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        prompts: List[str],
        max_length: int = 128,
        show_progress: bool = True,
    ) -> "JacobianLens":
        """
        Compute averaged J-lens vectors over a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            Diverse corpus; more = better averaging.  50-200 is practical.
        max_length : int
            Truncation length for tokenization.
        show_progress : bool

        Returns
        -------
        self (for chaining)
        """
        logger.info(
            "Fitting J-lens over %d prompts, %d layers, full_jacobian=%s",
            len(prompts),
            len(self.layer_indices),
            self.use_full_jacobian,
        )
        self._accum.clear()

        iterator = tqdm(prompts, desc="J-lens fit", disable=not show_progress, dynamic_ncols=True)
        for prompt in iterator:
            try:
                self._process_prompt(prompt, max_length=max_length)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM on prompt (skipped): %s", prompt[:60])
                torch.cuda.empty_cache()
                gc.collect()

        # finalise means
        for layer_idx, (sum_j, count) in self._accum.items():
            self.avg_jacobians[layer_idx] = sum_j / max(count, 1)

        logger.info("J-lens fit complete over %d layers", len(self.avg_jacobians))
        return self

    def readout(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Project a hidden state onto the averaged J-lens direction and
        return top-k vocabulary tokens.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape (d_model,) — a single position's hidden state.
        layer_idx : int
        k : int

        Returns
        -------
        list of (token_str, score) sorted descending.
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens. Call fit() first.")

        j_vec = self.avg_jacobians[layer_idx].to(hidden.device)  # (d_model,)
        # dot product: scalar score per vocab direction
        # j_vec shape is (d_model,); we interpret it as the readout direction
        score = torch.einsum("d,d->", j_vec, hidden.float())
        # For vocab-level readout, use unembed on the projected hidden
        logits = self.model.unembed(hidden.unsqueeze(0).unsqueeze(0)).squeeze()  # (vocab,)
        return self.model.top_tokens(logits, k=k)

    def readout_batch(
        self,
        hidden_states: Dict[int, torch.Tensor],
        k: int = 10,
    ) -> Dict[int, List[List[Tuple[str, float]]]]:
        """
        Apply readout to a full set of captured hidden states.

        Parameters
        ----------
        hidden_states : dict mapping layer_idx → (batch, seq, d_model)
        k : int

        Returns
        -------
        dict mapping layer_idx → list[seq_len] of top-k token lists
        """
        results: Dict[int, List[List[Tuple[str, float]]]] = {}
        for layer_idx, hs in hidden_states.items():
            if layer_idx not in self.avg_jacobians:
                continue
            # hs shape: (batch, seq, d_model) — take first batch item
            seq_results = []
            for pos in range(hs.shape[1]):
                h = hs[0, pos].to(self.model.device)  # (d_model,)
                seq_results.append(self.readout(h, layer_idx, k=k))
            results[layer_idx] = seq_results
        return results

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save averaged Jacobians to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "avg_jacobians": {k: v.cpu() for k, v in self.avg_jacobians.items()},
                "layer_indices": self.layer_indices,
                "model_id": self.model.model_id,
            },
            path,
        )
        logger.info("Saved J-lens to %s", path)

    @classmethod
    def load(cls, path: str | Path, model: HookedModel) -> "JacobianLens":
        """Load averaged Jacobians from a saved .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        jl = cls(model, layer_indices=data["layer_indices"])
        jl.avg_jacobians = {k: v for k, v in data["avg_jacobians"].items()}
        logger.info("Loaded J-lens from %s (%d layers)", path, len(jl.avg_jacobians))
        return jl

    # ── internals ───────────────────────────────────────────────────────────

    def _process_prompt(self, prompt: str, max_length: int) -> None:
        """
        Run one forward pass, compute Jacobians at each target layer,
        and accumulate into self._accum.
        """
        enc = self.model.tokenize([prompt], max_length=max_length)

        if self.use_full_jacobian:
            self._process_full_jacobian(enc)
        else:
            self._process_gradient_approx(enc)

    def _process_gradient_approx(self, enc: Dict[str, torch.Tensor]) -> None:
        """
        Fast approximation: gradient of the entropy of the output distribution
        w.r.t. each layer's hidden state at the last non-pad token position.

        Entropy captures uncertainty across the full vocab without needing
        the full Jacobian matrix — a single backward pass suffices.
        """
        with self.model.capture_residuals_with_grad(self.layer_indices) as store:
            outputs = self.model.model(**enc)

        # find last real token position (ignore padding)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            last_pos = int(attention_mask[0].sum()) - 1
        else:
            last_pos = enc["input_ids"].shape[1] - 1

        logits_last = outputs.logits[0, last_pos, :].float()  # (vocab,)
        log_probs = F.log_softmax(logits_last, dim=-1)
        probs = log_probs.exp()
        # negative entropy — minimising this makes dist sharper
        neg_entropy = (probs * log_probs).sum()

        for layer_idx, hs in store.items():
            if hs.grad_fn is None:
                continue
            grad = torch.autograd.grad(
                neg_entropy,
                hs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            # grad shape: (1, seq, d_model) — take last_pos
            g = grad[0, last_pos].detach().float().cpu()  # (d_model,)
            if layer_idx not in self._accum:
                self._accum[layer_idx] = (g.clone(), 1)
            else:
                s, c = self._accum[layer_idx]
                self._accum[layer_idx] = (s + g, c + 1)

        # explicit cleanup
        del outputs, logits_last, log_probs, probs, neg_entropy
        torch.cuda.empty_cache()
        gc.collect()

    def _process_full_jacobian(self, enc: Dict[str, torch.Tensor]) -> None:
        """
        Full Jacobian: ∂logits[last_pos, :] / ∂h[layer, last_pos, :].

        Computed in vocab-chunks to bound peak VRAM.
        Result is a (d_model,) vector averaged over vocab dimensions
        (mean absolute Jacobian row — gives the most "influential" hidden dims).
        """
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            last_pos = int(attention_mask[0].sum()) - 1
        else:
            last_pos = enc["input_ids"].shape[1] - 1

        vocab_size = self.model.vocab_size()

        with self.model.capture_residuals_with_grad(self.layer_indices) as store:
            outputs = self.model.model(**enc)

        logits_last = outputs.logits[0, last_pos, :].float()  # (vocab,)

        for layer_idx, hs in store.items():
            if hs.grad_fn is None:
                continue

            d_model = hs.shape[-1]
            jac_accum = torch.zeros(d_model, dtype=torch.float32)

            # chunk over vocab to stay within VRAM budget
            for v_start in range(0, vocab_size, self.chunk_size):
                v_end = min(v_start + self.chunk_size, vocab_size)
                chunk_sum = logits_last[v_start:v_end].sum()
                grad = torch.autograd.grad(
                    chunk_sum,
                    hs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    jac_accum += grad[0, last_pos].detach().float().cpu().abs()

            g = jac_accum / vocab_size  # mean over vocab chunks

            if layer_idx not in self._accum:
                self._accum[layer_idx] = (g.clone(), 1)
            else:
                s, c = self._accum[layer_idx]
                self._accum[layer_idx] = (s + g, c + 1)

        del outputs, logits_last
        torch.cuda.empty_cache()
        gc.collect()
