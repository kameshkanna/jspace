"""
Jacobian Lens (J-lens) — exact paper implementation, fused multi-layer estimator.

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

Fused Hutchinson VJP estimator (all layers per prompt):
    One forward pass injects differentiable clone-leaves at every layer l.
    For each Rademacher vector v:
        scalar = E_{t'}[ h_final_{t'} . v ]
        torch.autograd.grad(scalar, [h_clone_0, ..., h_clone_L]) fills ALL layer
        gradients in a single backward traversal of the computation graph.
    This gives grad_l[0,t] = E_{t'}[ J_l(t,t')^T ] @ v for all l and t at once.
    J_l estimate = E_v[ outer(v, mean_t(grad_l)) ]  (Hutchinson identity)

    Cost vs original (per-layer separate forwards):
        Original:  n_prompts × n_layers × n_proj backward passes
        Fused:     n_prompts × n_proj backward passes  (~n_layers× faster)
        28 layers, 1000 prompts, 32 proj:  ~4.5 hrs → ~20-25 min on H100

Cost on H100 80GB (7B model, d_model=3584):
    Forward pass:  ~50 ms
    Backward pass: ~200 ms  (all layers simultaneously)
    1000 prompts × 32 proj = 32 K backwards ≈ 20-25 min  (paper quality)
    J storage: 28 layers × 3584^2 × 4 B ≈ 1.4 GB
    Corpus jh:  28 layers × n_prompts × 3584 × 4 B ≈ 400 MB (for 1000 prompts)
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
          Pass 1 — fused Hutchinson VJP: ONE forward + n_proj backward passes
                   per prompt, computing J_l for ALL layers simultaneously.
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
            "J-lens fit pass 1 (fused): %d prompts × %d layers × %d projections  (d_model=%d)",
            len(prompts), n_layers, self.n_proj, d_model,
        )
        self._accum.clear()

        pbar = tqdm(
            total=len(prompts),
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
            try:
                self._process_all_layers_fused(enc, seq_len)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM prompt %.40s — skipped", prompt)
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

    def readout_logits_batch(
        self,
        hidden_seq: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Vectorized readout for a full sequence.

        Parameters
        ----------
        hidden_seq : (seq, d_model) hidden states at layer layer_idx
        layer_idx : int

        Returns
        -------
        (seq, vocab_size) logit tensor (detached, CPU)
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens.")
        J = self.avg_jacobians[layer_idx].float().to(self.model.device)  # (d, d)
        h = hidden_seq.float().to(self.model.device)                     # (seq, d)
        Jh = (J @ h.T).T                                                 # (seq, d)
        # unembed expects (..., d_model); pass (1, seq, d) → (1, seq, vocab)
        logits = self.model.unembed(Jh.unsqueeze(0)).squeeze(0)         # (seq, vocab)
        return logits.detach().cpu()

    def readout_jh_batch(
        self,
        hidden_seq: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Vectorized J_l @ h for a full sequence.

        Parameters
        ----------
        hidden_seq : (seq, d_model)
        layer_idx : int

        Returns
        -------
        (seq, d_model) projected vectors (CPU)
        """
        if layer_idx not in self.avg_jacobians:
            raise KeyError(f"Layer {layer_idx} not in fitted J-lens.")
        J = self.avg_jacobians[layer_idx].float()   # (d, d) on CPU
        h = hidden_seq.float().cpu()                # (seq, d)
        return (J @ h.T).T                          # (seq, d)

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

    # ── internals — pass 1: fused Hutchinson J estimation ───────────────────

    def _process_all_layers_fused(
        self,
        enc: Dict[str, torch.Tensor],
        seq_len: int,
    ) -> None:
        """
        Fused Hutchinson VJP: ONE forward + n_proj backward passes, computing
        J_l for ALL layer_indices simultaneously.

        Correctness argument
        --------------------
        Hooks capture h_l as a REFERENCE (no clone-leaf replacement).  The full
        computation graph from h_0 → h_1 → … → h_final remains intact.
        For each Rademacher v, one backward call computes:

            torch.autograd.grad(scalar, [h_0, ..., h_L])

        where scalar = E_{t'}[h_final_{t'} · v].  The gradient w.r.t. h_l is:

            d(scalar)/d(h_l) = (dh_final/dh_l)^T @ v = J_l^T @ v  (per position)

        This is the exact same VJP the original per-layer code computed, but for
        ALL layers in one pass — mathematically identical, not an approximation.

        Why clone-leaves DON'T work for the fused case
        -----------------------------------------------
        If we had replaced each layer's output with a .clone().requires_grad_(True)
        leaf, the hook at layer l+1 would break the graph path from h_final back to
        h_clone_l, giving grad(scalar, h_clone_l) = 0 for all l < final.
        The correct fix is to NOT replace outputs — just observe them.

        Speedup vs original: n_proj backward passes instead of n_layers × n_proj.
        For 28 layers / 1000 prompts / 32 proj: ~4.5 hrs → ~20-25 min on H100.
        """
        d_model = self.model.model.config.hidden_size
        device = self.model.device
        final_idx = self.model.num_layers - 1

        h_store: Dict[int, torch.Tensor] = {}   # layer_idx → activation IN graph
        hooks: List = []

        layer_idx_set = set(self.layer_indices)
        # Always capture final layer so we have the target for the VJP
        needed = layer_idx_set | {final_idx}

        def make_capture_hook(idx: int):
            def hook(module, _input, output):
                hs = output[0] if isinstance(output, tuple) else output
                h_store[idx] = hs   # store reference; do NOT replace output
            return hook

        for l in needed:
            hooks.append(self.model._layers[l].register_forward_hook(make_capture_hook(l)))

        try:
            with torch.enable_grad():
                _ = self.model.model(**enc)
        finally:
            for h in hooks:
                h.remove()

        if final_idx not in h_store:
            return

        h_final = h_store[final_idx]                          # (1, seq, d) in graph
        tgt_start = max(0, seq_len // 2)
        h_final_tgt = h_final[0, tgt_start:seq_len].float()  # (n_tgt, d)
        n_tgt = max(h_final_tgt.shape[0], 1)

        # Non-leaf tensors that are part of the same computation graph
        active_layers = [l for l in self.layer_indices if l in h_store]
        h_tensors = [h_store[l] for l in active_layers]

        vs = (torch.randint(0, 2, (self.n_proj, d_model), device=device).float() * 2 - 1)

        J_sums: Dict[int, torch.Tensor] = {
            l: torch.zeros(d_model, d_model, dtype=torch.float32)
            for l in active_layers
        }

        for i in range(self.n_proj):
            v = vs[i]
            scalar = (h_final_tgt * v.unsqueeze(0)).sum() / n_tgt

            retain = i < self.n_proj - 1
            grads = torch.autograd.grad(
                scalar,
                h_tensors,
                retain_graph=retain,
                create_graph=False,
                allow_unused=True,
            )

            v_cpu = v.cpu()
            for l, grad in zip(active_layers, grads):
                if grad is None:
                    continue
                g_avg = grad[0, :seq_len].detach().float().mean(0).cpu()   # (d,)
                J_sums[l] += torch.outer(v_cpu, g_avg)

        for l in active_layers:
            J_estimate = J_sums[l] / self.n_proj
            if l not in self._accum:
                self._accum[l] = (J_estimate, 1)
            else:
                s, c = self._accum[l]
                self._accum[l] = (s + J_estimate, c + 1)

        del h_store, J_sums, h_tensors
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
