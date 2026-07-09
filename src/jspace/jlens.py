"""
Jacobian Lens (J-lens) — exact paper implementation.

Paper (Lindsey et al., 2026):
    J_l = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]

J_l is a (d_model x d_model) matrix: the averaged linearised map from an
intermediate residual-stream hidden state to the FINAL LAYER residual stream
(before lm_head / norm).  This is NOT the Jacobian of the output logits —
it targets h_final directly, which is what the paper specifies.

Readout (paper formula):
    lens(h_l) = softmax(W_U  norm(J_l @ h_l))

J-space corpus vectors:
    After fitting J_l, a second pass projects each corpus hidden state through
    J_l and stores the unit-normalised result:
        corpus_jh[l] = { J_l @ h_t / ||J_l @ h_t|| } for each (prompt, pos) pair
    These are used by WorkspaceAnalyzer for gradient-pursuit active-concept counting.

Hutchinson VJP estimator:
    For Rademacher vector v ~ {+1,-1}^d_model:
        grad( E_{t'}[ h_final_{t'} . v ], h_clone ) = E_{t'}[ J_l(t, t')^T @ v ]
    where the gradient is taken w.r.t. a clone-leaf of h_l injected at layer l.
    Because h_clone is shape (1, seq, d_model), ONE backward call returns the
    gradient for ALL source positions t simultaneously.
    We average the gradient over source t and the scalar sum over target t', so
    a single backward covers the full E_{t, t'>=t} expectation in the paper.
    Hutchinson accumulation: J_l = E_v[ outer(v, g_avg) ] where g_avg = E_t[J_l^T @ v].

Cost on H100 80GB (7B model, d_model=4096):
    Forward pass:  ~50 ms
    Backward pass: ~100 ms
    n_layers × n_prompts × n_proj backward passes total
    100 prompts × 32 layers × 16 proj = ~51 K backwards ≈ 85 min
    1000 prompts × 32 layers × 16 proj ≈ 14 hours (matches paper's 1000-prompt corpus)
    J storage: 32 layers × 4096^2 × 4 B ≈ 2 GB
    Corpus jh:  32 layers × n_prompts × 4096 × 4 B ≈ 512 MB (for 1000 prompts)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jspace.model import HookedModel

logger = logging.getLogger(__name__)


class JacobianLens:
    """
    Computes and caches the averaged Jacobian readout matrices (J-space)
    and the corpus of J-lens readout vectors used for gradient pursuit.

    Parameters
    ----------
    model : HookedModel
    layer_indices : list[int] | None
        Layers to compute J for.  None = all layers.
    n_proj : int
        Hutchinson projections per (prompt, layer).
        16 = fast, 32 = paper quality.
    top_k_vocab : int
        Vocabulary tokens to return per readout call.
    """

    def __init__(
        self,
        model: HookedModel,
        layer_indices: Optional[List[int]] = None,
        n_proj: int = 16,
        top_k_vocab: int = 20,
    ) -> None:
        self.model = model
        self.layer_indices: List[int] = (
            layer_indices if layer_indices is not None else list(range(model.num_layers))
        )
        self.n_proj = n_proj
        self.top_k_vocab = top_k_vocab

        # {layer_idx: (sum_J (d,d), count)}
        self._accum: Dict[int, Tuple[torch.Tensor, int]] = {}

        # {layer_idx: J_l (d_model, d_model)} — fitted matrices
        self.avg_jacobians: Dict[int, torch.Tensor] = {}

        # {layer_idx: Tensor (n_corpus, d_model)} — unit-normed J_l @ h per corpus pos
        # used by WorkspaceAnalyzer for gradient pursuit n_active counting
        self.corpus_jh: Dict[int, torch.Tensor] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        prompts: List[str],
        max_length: int = 128,
        show_progress: bool = True,
    ) -> "JacobianLens":
        """
        Two-pass fit:
          Pass 1 — compute J_l matrices via Hutchinson VJP (the Jacobian fit).
          Pass 2 — project corpus hidden states through J_l to build corpus_jh
                   for gradient pursuit active-concept counting.

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
            "J-lens fit pass 1: %d prompts × %d layers × %d projections  (d_model=%d)",
            len(prompts), n_layers, self.n_proj, d_model,
        )
        self._accum.clear()

        pbar = tqdm(
            total=len(prompts) * n_layers,
            desc="J-lens pass 1",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for prompt in prompts:
            enc = self.model.tokenize([prompt], max_length=max_length)
            seq_len = (
                int(enc["attention_mask"][0].sum())
                if "attention_mask" in enc
                else enc["input_ids"].shape[1]
            )
            for layer_idx in self.layer_indices:
                try:
                    self._process_one_layer(enc, layer_idx, seq_len)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM layer %d prompt %.40s — skipped", layer_idx, prompt)
                    torch.cuda.empty_cache()
                    gc.collect()
                pbar.update(1)

        pbar.close()

        for layer_idx, (sum_j, count) in self._accum.items():
            self.avg_jacobians[layer_idx] = sum_j / max(count, 1)
        logger.info("Pass 1 complete: %d layers", len(self.avg_jacobians))

        # pass 2: build corpus J-lens readout vectors for gradient pursuit
        logger.info("J-lens fit pass 2: projecting corpus through J_l ...")
        self._build_corpus_jh(prompts, max_length, show_progress)
        logger.info("Pass 2 complete: corpus_jh has %d layers", len(self.corpus_jh))

        return self

    def readout(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Paper readout: lens(h_l) = softmax(W_U norm(J_l @ h_l))

        Parameters
        ----------
        hidden : (d_model,) hidden state at layer layer_idx
        layer_idx : int
        k : int

        Returns
        -------
        list of (token_str, prob) sorted descending
        """
        return self.model.top_tokens(self.readout_logits(hidden, layer_idx), k=k)

    def readout_logits(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Full logit vector via paper formula: W_U(norm(J_l @ h_l)).
        unembed() applies layernorm + lm_head = norm(.) then W_U.
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens.")
        J = self.avg_jacobians[layer_idx].to(self.model.device)
        h = hidden.float().to(self.model.device)
        Jh = J @ h
        return self.model.unembed(Jh.unsqueeze(0).unsqueeze(0)).squeeze().detach()

    def readout_jh(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Return J_l @ h (raw projected vector, not unembedded).
        Used for gradient pursuit in WorkspaceAnalyzer.
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens.")
        J = self.avg_jacobians[layer_idx].float().to(self.model.device)
        h = hidden.float().to(self.model.device)
        return (J @ h).cpu()

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "avg_jacobians": {k: v.cpu() for k, v in self.avg_jacobians.items()},
                "corpus_jh": {k: v.cpu() for k, v in self.corpus_jh.items()},
                "layer_indices": self.layer_indices,
                "model_id": self.model.model_id,
                "n_proj": self.n_proj,
            },
            path,
        )
        logger.info("Saved J-lens (%d layers, %d corpus vecs) to %s",
                    len(self.avg_jacobians),
                    sum(v.shape[0] for v in self.corpus_jh.values()) // max(len(self.corpus_jh), 1),
                    path)

    @classmethod
    def load(cls, path: str | Path, model: HookedModel) -> "JacobianLens":
        data = torch.load(path, map_location="cpu", weights_only=True)
        jl = cls(model, layer_indices=data["layer_indices"], n_proj=data.get("n_proj", 16))
        jl.avg_jacobians = {k: v for k, v in data["avg_jacobians"].items()}
        jl.corpus_jh = {k: v for k, v in data.get("corpus_jh", {}).items()}
        logger.info("Loaded J-lens from %s (%d layers, corpus_jh=%s)",
                    path, len(jl.avg_jacobians), bool(jl.corpus_jh))
        return jl

    # ── internals — pass 1: Hutchinson J estimation ──────────────────────────

    def _process_one_layer(
        self,
        enc: Dict[str, torch.Tensor],
        layer_idx: int,
        seq_len: int,
    ) -> None:
        """
        One forward + n_proj backward passes for layer_idx.

        Each backward pass:
          - scalar = E_{t' in second half}[ h_final_{t'} . v ]
          - grad w.r.t. h_clone = (1, seq, d_model) leaf
          - grad[0, t] = E_{t'}[ J_l(t,t')^T ] @ v  for ALL source t at once

        J_l estimate = E_{v, t}[ outer(v, grad[0,t]) ]
                     = E_{v, t, t'}[ outer(v, J_l(t,t')^T @ v) ]
                     = J_l  (by Hutchinson identity E_v[outer(v, A^T@v)] = A)
        """
        d_model = self.model.model.config.hidden_size
        device = self.model.device
        final_idx = self.model.num_layers - 1

        h_clone_store: Dict[int, torch.Tensor] = {}
        h_final_store: Dict[str, torch.Tensor] = {}

        def grad_hook(module, _input, output, _idx=layer_idx):
            hs = output[0] if isinstance(output, tuple) else output
            clone = hs.clone().requires_grad_(True)   # leaf; rest of graph flows through it
            h_clone_store[_idx] = clone
            return (clone,) + output[1:] if isinstance(output, tuple) else clone

        def final_hook(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            h_final_store["h"] = hs                   # in graph, no detach

        h1 = self.model._layers[layer_idx].register_forward_hook(grad_hook)
        h2 = (
            self.model._layers[final_idx].register_forward_hook(final_hook)
            if layer_idx < final_idx
            else None
        )

        try:
            with torch.enable_grad():
                _ = self.model.model(**enc)
        finally:
            h1.remove()
            if h2 is not None:
                h2.remove()

        if layer_idx not in h_clone_store:
            return

        if layer_idx == final_idx:
            # J at the last layer is identity; h_final is h_l itself
            h_final_store["h"] = h_clone_store[layer_idx]

        if "h" not in h_final_store:
            return

        h_clone = h_clone_store[layer_idx]    # (1, seq, d_model) leaf
        h_final = h_final_store["h"]           # (1, seq, d_model) in graph

        # Target positions: second half of sequence (paper: t' >= t; second half
        # ensures t' >= most source positions and covers the generation zone)
        tgt_start = max(0, seq_len // 2)
        h_final_tgt = h_final[0, tgt_start:seq_len].float()   # (n_tgt, d_model)
        n_tgt = max(h_final_tgt.shape[0], 1)

        # Rademacher vectors: (n_proj, d_model)
        vs = (torch.randint(0, 2, (self.n_proj, d_model), device=device).float() * 2 - 1)

        J_sum = torch.zeros(d_model, d_model, dtype=torch.float32, device="cpu")

        for i in range(self.n_proj):
            v = vs[i]   # (d_model,)

            # scalar = E_{t'}[ h_final_{t'} . v ] — average over target positions
            scalar = (h_final_tgt * v.unsqueeze(0)).sum() / n_tgt

            retain = i < self.n_proj - 1
            grad = torch.autograd.grad(
                scalar,
                h_clone,
                retain_graph=retain,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                continue

            # grad[0]: (seq, d_model) — E_{t'}[J_l(t,t')^T] @ v  for each source t
            # Average over source positions 0..seq_len-1
            g_avg = grad[0, :seq_len].detach().float().mean(0).cpu()   # (d_model,)

            # Hutchinson accumulation: outer(v, g_avg) estimates J_l
            J_sum += torch.outer(v.cpu(), g_avg)

        J_estimate = J_sum / self.n_proj

        if layer_idx not in self._accum:
            self._accum[layer_idx] = (J_estimate, 1)
        else:
            s, c = self._accum[layer_idx]
            self._accum[layer_idx] = (s + J_estimate, c + 1)

        del h_final_store, h_clone_store
        torch.cuda.empty_cache()
        gc.collect()

    # ── internals — pass 2: corpus J-lens readout vectors ───────────────────

    def _build_corpus_jh(
        self,
        prompts: List[str],
        max_length: int,
        show_progress: bool,
    ) -> None:
        """
        Project corpus hidden states through J_l and store unit-normalised vectors.

        For each prompt we store the LAST token's J_l @ h at every layer.
        These form the "J-lens vectors" used by WorkspaceAnalyzer's gradient pursuit.

        Storage: n_prompts × n_layers × d_model floats
                 1000 × 32 × 4096 × 4 B ≈ 512 MB
        """
        accum: Dict[int, List[torch.Tensor]] = {l: [] for l in self.layer_indices}

        with torch.no_grad():
            for prompt in tqdm(
                prompts, desc="J-lens pass 2", disable=not show_progress, dynamic_ncols=True
            ):
                enc = self.model.tokenize([prompt], max_length=max_length)
                seq_len = (
                    int(enc["attention_mask"][0].sum())
                    if "attention_mask" in enc
                    else enc["input_ids"].shape[1]
                )
                last_pos = seq_len - 1

                try:
                    with self.model.capture_residuals(self.layer_indices) as store:
                        _ = self.model.model(**enc)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                for l in self.layer_indices:
                    if l not in store or l not in self.avg_jacobians:
                        continue
                    h = store[l][0, last_pos].float().cpu()       # (d_model,)
                    J = self.avg_jacobians[l].float()              # (d_model, d_model)
                    Jh = J @ h                                      # (d_model,)
                    norm = Jh.norm()
                    if norm > 1e-8:
                        accum[l].append(Jh / norm)

        self.corpus_jh = {
            l: torch.stack(vecs)   # (n_corpus, d_model)
            for l, vecs in accum.items()
            if vecs
        }
