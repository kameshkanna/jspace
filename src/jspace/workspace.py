"""
Global Workspace analysis — paper-faithful implementation.

Paper (Lindsey et al., 2026) methodology:

  Active concept detection (gradient pursuit / sparse nonneg decomposition):
    The paper finds the minimum k such that k "concept vectors" from the J-lens
    corpus can reconstruct the query hidden state with >=95% explained variance.
    n_active = k  (averaged across prompt positions).

    Concretely:
      1. corpus_jh[l] — normalised J_l @ h vectors from the fit corpus  (n_corpus, d_model)
      2. query_jh = J_l @ h_query  (d_model,)
      3. Greedy matching pursuit: at each step pick the corpus vector with highest
         dot product with the current residual, subtract its projection, until
         95% of the original norm-squared is explained.  k = number of steps.

    The kurtosis proxy from the v1 implementation is kept as a fast fallback when
    corpus_jh is absent (e.g. old .pt files).

  Workspace layer detection:
    Three phases determined by:
      1. n_active (gradient pursuit count) per layer
      2. Hunchback entropy profile — entropy rises early, dips in workspace
      3. Variance explained by J-space direction

  Readout formula (paper eq.):
    lens(h_l) = softmax(W_U · norm(J_l @ h_l))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy, kurtosis as scipy_kurtosis

from jspace.jlens import JacobianLens
from jspace.model import HookedModel

logger = logging.getLogger(__name__)

# Fraction of query norm-squared that must be explained to stop pursuit
_PURSUIT_COVERAGE = 0.95
# Maximum k before we give up (paper upper bound is ~25)
_MAX_K = 30


@dataclass
class LayerWorkspaceStats:
    """Per-layer workspace statistics."""

    layer_idx: int
    n_active: int                          # min k from gradient pursuit (or kurtosis fallback)
    mean_entropy: float                    # mean entropy of J-lens readout distribution
    mean_kurtosis: float                   # mean excess kurtosis across positions
    var_explained: float                   # fraction of hidden-state variance in J-space direction
    top_concepts: List[Tuple[str, float]]  # top concepts at last position (via J-lens readout)
    phase: str = ""                        # "Early" | "Workspace" | "Output"


@dataclass
class WorkspaceReport:
    """Full workspace analysis for a single prompt."""

    prompt: str
    tokens: List[str]
    layer_stats: List[LayerWorkspaceStats]
    workspace_start: int = -1
    workspace_end: int = -1

    @property
    def workspace_layers(self) -> List[LayerWorkspaceStats]:
        return [s for s in self.layer_stats if s.phase == "Workspace"]

    @property
    def peak_capacity(self) -> int:
        ws = self.workspace_layers
        return max(s.n_active for s in ws) if ws else 0


class WorkspaceAnalyzer:
    """
    Analyses global workspace properties of a model on a given prompt.

    Active concept detection uses gradient pursuit over the corpus J-lens
    vectors when available, falling back to excess kurtosis when corpus_jh
    is absent (old .pt files).

    Parameters
    ----------
    jlens : JacobianLens
        Fitted J-lens (call jlens.fit() first).
    kurtosis_threshold : float
        Kurtosis threshold used only when corpus_jh is absent (fallback).
    entropy_threshold : float | None
        Optional entropy upper bound for phase detection.
        None = auto-scale to 60% of max vocab entropy.
    var_threshold : float
        Min variance fraction in J-space direction for workspace label.
    top_k : int
        Top-k vocabulary tokens to report per position.
    pursuit_coverage : float
        Fraction of norm-squared that must be explained by matching pursuit.
        Default 0.95 (paper value).
    """

    def __init__(
        self,
        jlens: JacobianLens,
        kurtosis_threshold: float = 0.0,
        entropy_threshold: Optional[float] = None,
        var_threshold: float = 0.05,
        top_k: int = 10,
        pursuit_coverage: float = _PURSUIT_COVERAGE,
    ) -> None:
        self.jlens = jlens
        self.model: HookedModel = jlens.model
        self.kurtosis_threshold = kurtosis_threshold
        self.var_threshold = var_threshold
        self.top_k = top_k
        self.pursuit_coverage = pursuit_coverage

        has_corpus = bool(jlens.corpus_jh)
        logger.info(
            "WorkspaceAnalyzer: corpus_jh=%s  n_active_method=%s  pursuit_coverage=%.2f",
            has_corpus,
            "gradient_pursuit" if has_corpus else "kurtosis_fallback",
            pursuit_coverage,
        )

        vocab_size = self.model.vocab_size()
        max_entropy = math.log(vocab_size)
        self.entropy_threshold = (
            entropy_threshold if entropy_threshold is not None else max_entropy * 0.60
        )

    # ── public API ──────────────────────────────────────────────────────────

    def analyse(self, prompt: str, max_length: int = 128) -> WorkspaceReport:
        """
        Run full workspace analysis on a single prompt.

        Parameters
        ----------
        prompt : str
        max_length : int

        Returns
        -------
        WorkspaceReport
        """
        enc = self.model.tokenize([prompt], max_length=max_length)
        tokens = [
            self.model.tokenizer.decode([int(t)])
            for t in enc["input_ids"][0]
            if int(t) != self.model.tokenizer.pad_token_id
        ]

        layer_stats: List[LayerWorkspaceStats] = []

        with torch.no_grad():
            with self.model.capture_residuals(self.jlens.layer_indices) as store:
                _ = self.model.model(**enc)

        for layer_idx in self.jlens.layer_indices:
            if layer_idx not in store:
                continue
            hs = store[layer_idx]  # (1, seq, d_model)
            stats = self._compute_layer_stats(layer_idx, hs, tokens)
            layer_stats.append(stats)

        self._assign_phases(layer_stats)
        ws_start, ws_end = self._detect_workspace_window(layer_stats)

        return WorkspaceReport(
            prompt=prompt,
            tokens=tokens,
            layer_stats=layer_stats,
            workspace_start=ws_start,
            workspace_end=ws_end,
        )

    def analyse_batch(self, prompts: List[str], max_length: int = 128) -> List[WorkspaceReport]:
        return [self.analyse(p, max_length=max_length) for p in prompts]

    # ── internals ───────────────────────────────────────────────────────────

    def _compute_layer_stats(
        self,
        layer_idx: int,
        hs: torch.Tensor,
        tokens: List[str],
    ) -> LayerWorkspaceStats:
        """
        Compute workspace stats for one layer via paper-faithful J-lens readout.

        hs : (1, seq, d_model)

        n_active is computed via gradient pursuit when corpus_jh is available,
        otherwise falls back to excess kurtosis count.
        """
        seq_len = hs.shape[1]

        entropies: List[float] = []
        kurtoses: List[float] = []
        top_concepts_last: List[Tuple[str, float]] = []
        k_list: List[int] = []

        corpus_available = layer_idx in self.jlens.corpus_jh

        for pos in range(seq_len):
            h = hs[0, pos].to(self.model.device)  # (d_model,)

            # paper readout: W_U(norm(J_l @ h))
            if layer_idx in self.jlens.avg_jacobians:
                logits = self.jlens.readout_logits(h, layer_idx)  # (vocab,)
            else:
                logits = self.model.unembed(h.unsqueeze(0).unsqueeze(0)).squeeze().detach()

            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            ent = float(scipy_entropy(probs))
            kurt = float(scipy_kurtosis(probs, fisher=True, bias=False))

            entropies.append(ent)
            kurtoses.append(kurt)

            if pos == seq_len - 1:
                top_concepts_last = self.model.top_tokens(logits, k=self.top_k)

            # --- active detection ---
            if corpus_available and layer_idx in self.jlens.avg_jacobians:
                jh = self.jlens.readout_jh(h, layer_idx)  # (d_model,)
                k = self._gradient_pursuit(jh, layer_idx)
                k_list.append(k)

        if k_list:
            # average k across sequence positions — matches paper's per-layer aggregation
            n_active = int(round(float(np.mean(k_list))))
        else:
            # fallback: kurtosis count (legacy behaviour)
            n_active = int(np.sum(np.array(kurtoses) > self.kurtosis_threshold))

        mean_entropy = float(np.mean(entropies))
        mean_kurtosis = float(np.mean(kurtoses))

        # variance explained by primary J-space direction at this layer
        j_mat = self.jlens.avg_jacobians.get(layer_idx)
        var_explained = 0.0
        if j_mat is not None:
            hs_flat = hs[0].float().cpu()                      # (seq, d_model)
            Jh = (j_mat.float() @ hs_flat.T).T                # (seq, d_model)
            j_unit = F.normalize(Jh.mean(0).unsqueeze(0), dim=-1).squeeze()  # (d_model,)
            proj = hs_flat @ j_unit                            # (seq,)
            var_proj = float(proj.var())
            var_total = float(hs_flat.var())
            var_explained = var_proj / (var_total + 1e-8)

        return LayerWorkspaceStats(
            layer_idx=layer_idx,
            n_active=n_active,
            mean_entropy=mean_entropy,
            mean_kurtosis=mean_kurtosis,
            var_explained=var_explained,
            top_concepts=top_concepts_last,
        )

    def _gradient_pursuit(self, query_jh: torch.Tensor, layer_idx: int) -> int:
        """
        Greedy matching pursuit: find minimum k corpus J-lens vectors needed
        to explain >= pursuit_coverage of query_jh norm-squared.

        Paper: "sparse nonneg gradient pursuit" — we use unconstrained matching
        pursuit (signed dot product) as the standard operationalisation.

        Parameters
        ----------
        query_jh : (d_model,) J_l @ h for the query position, already normalised
        layer_idx : int

        Returns
        -------
        k : int  (0 if query is zero-norm; up to _MAX_K if coverage not reached)
        """
        corpus = self.jlens.corpus_jh[layer_idx].float()  # (n_corpus, d_model)

        q = query_jh.float()
        q_norm_sq = float(q.dot(q))
        if q_norm_sq < 1e-12:
            return 0

        residual = q.clone()
        residual_sq = q_norm_sq

        for k in range(1, _MAX_K + 1):
            # pick corpus vector with maximum |dot product| with residual
            dots = corpus @ residual               # (n_corpus,)
            best_idx = int(torch.abs(dots).argmax())
            c = corpus[best_idx]                   # (d_model,)

            # project residual onto c and subtract
            c_norm_sq = float(c.dot(c))
            if c_norm_sq < 1e-12:
                break
            proj_scalar = float(c.dot(residual)) / c_norm_sq
            residual = residual - proj_scalar * c
            residual_sq = float(residual.dot(residual))

            explained = 1.0 - residual_sq / q_norm_sq
            if explained >= self.pursuit_coverage:
                return k

        return _MAX_K

    def _assign_phases(self, stats: List[LayerWorkspaceStats]) -> None:
        """
        Label each layer as Early / Workspace / Output.

        Strategy (following paper's hunchback entropy profile):
          1. Final 10% of layers → Output (next-token collapse zone).
          2. Find the entropy valley (minimum) among non-output layers.
          3. Layers within 40% of the entropy range above the minimum AND
             whose n_active is above the median → Workspace.
          4. Everything else → Early.
        """
        if not stats:
            return

        n_layers = len(stats)
        entropies = np.array([s.mean_entropy for s in stats])
        n_actives = np.array([s.n_active for s in stats])
        output_cutoff = int(n_layers * 0.90)

        non_output_mask = np.array([s.layer_idx < output_cutoff for s in stats])
        if non_output_mask.sum() == 0:
            for s in stats:
                s.phase = "Output"
            return

        min_ent = entropies[non_output_mask].min()
        max_ent = entropies[non_output_mask].max()
        ent_range = max(max_ent - min_ent, 1e-6)
        tolerance = ent_range * 0.40

        median_active = float(np.median(n_actives[non_output_mask]))

        for i, s in enumerate(stats):
            if not non_output_mask[i]:
                s.phase = "Output"
            elif entropies[i] <= min_ent + tolerance and n_actives[i] >= median_active:
                s.phase = "Workspace"
            else:
                s.phase = "Early"

    def _detect_workspace_window(
        self, stats: List[LayerWorkspaceStats]
    ) -> Tuple[int, int]:
        ws_indices = [s.layer_idx for s in stats if s.phase == "Workspace"]
        if not ws_indices:
            return -1, -1
        return min(ws_indices), max(ws_indices)

    # ── pretty printing ──────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: WorkspaceReport, max_layers: int = 30) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f'Workspace Analysis -- "{report.prompt[:60]}"')
            table.add_column("Layer", justify="right")
            table.add_column("Phase", justify="center")
            table.add_column("n_active", justify="right")
            table.add_column("entropy", justify="right")
            table.add_column("kurtosis", justify="right")
            table.add_column("var_expl", justify="right")
            table.add_column("Top concept", justify="left")

            step = max(1, len(report.layer_stats) // max_layers)
            for s in report.layer_stats[::step]:
                phase_color = {"Early": "dim", "Workspace": "green", "Output": "yellow"}.get(
                    s.phase, "white"
                )
                top_tok = s.top_concepts[0][0] if s.top_concepts else "-"
                table.add_row(
                    str(s.layer_idx),
                    f"[{phase_color}]{s.phase}[/{phase_color}]",
                    str(s.n_active),
                    f"{s.mean_entropy:.2f}",
                    f"{s.mean_kurtosis:.1f}",
                    f"{s.var_explained:.3f}",
                    top_tok,
                )
            console.print(table)
            console.print(
                f"Workspace window: layers {report.workspace_start}-{report.workspace_end}  "
                f"| Peak capacity: {report.peak_capacity} active positions"
            )
        except ImportError:
            print(f"\nWorkspace Analysis: {report.prompt[:60]}")
            for s in report.layer_stats[::max(1, len(report.layer_stats) // max_layers)]:
                top_tok = s.top_concepts[0][0] if s.top_concepts else "-"
                print(
                    f"  L{s.layer_idx:3d} [{s.phase:9s}]  "
                    f"n_active={s.n_active:3d}  H={s.mean_entropy:.2f}  "
                    f"kurt={s.mean_kurtosis:.1f}  var={s.var_explained:.3f}  top={top_tok}"
                )
            print(
                f"\nWorkspace: layers {report.workspace_start}-{report.workspace_end}  "
                f"peak_capacity={report.peak_capacity}"
            )
