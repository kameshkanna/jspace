"""
Global Workspace analysis — paper-faithful implementation.

Paper (Lindsey et al., 2026) methodology:

  Active concept detection:
    A position is "active" if its J-lens readout is concentrated on a small
    number of vocabulary tokens.  The paper uses EXCESS KURTOSIS of the
    readout probability distribution as the primary signal — high kurtosis
    means the distribution is spiky (one concept active), low kurtosis
    means it is flat (no clear concept).

    We threshold: position is active if kurtosis(readout_probs) > kurtosis_threshold.
    Default threshold = 0  (excess kurtosis > 0 means heavier tail than Gaussian).

  Workspace layer detection:
    The paper identifies three phases by combining:
      1. n_active  — count of positions above kurtosis threshold
      2. Hunchback entropy profile — entropy rises early, dips in workspace
      3. Variance explained by J-space direction

    Workspace phase: layers where n_active is high (relative to model max) AND
    entropy is in the valley (middle of the hunchback).

  Readout formula (paper eq.):
    lens(h_l) = softmax(W_U · norm(J_l @ h_l))

  All of the above is computed using the J-lens matrices from JacobianLens,
  which are (d_model x d_model) averaged Jacobian matrices.
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


@dataclass
class LayerWorkspaceStats:
    """Per-layer workspace statistics."""

    layer_idx: int
    n_active: int                          # positions with high-kurtosis J-lens readout
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

    Uses paper-faithful readout: lens(h_l) = W_U(norm(J_l @ h_l))
    and excess kurtosis as the active-concept detector.

    Parameters
    ----------
    jlens : JacobianLens
        Fitted J-lens (call jlens.fit() first).
        Must contain (d_model x d_model) matrices — not the old scalar vectors.
    kurtosis_threshold : float
        Excess kurtosis threshold for active detection.  Default 0 (heavier-
        tailed than Gaussian).  Increase to be stricter.
    entropy_threshold : float | None
        Optional entropy upper bound (nats).  None = auto-scale to 60% of
        max vocab entropy, used for phase detection only.
    var_threshold : float
        Minimum fraction of variance in J-space direction for workspace label.
    top_k : int
        Top-k vocabulary tokens to report per position.
    """

    def __init__(
        self,
        jlens: JacobianLens,
        kurtosis_threshold: float = 0.0,
        entropy_threshold: Optional[float] = None,
        var_threshold: float = 0.05,
        top_k: int = 10,
    ) -> None:
        self.jlens = jlens
        self.model: HookedModel = jlens.model
        self.kurtosis_threshold = kurtosis_threshold
        self.var_threshold = var_threshold
        self.top_k = top_k

        vocab_size = self.model.vocab_size()
        max_entropy = math.log(vocab_size)
        self.entropy_threshold = (
            entropy_threshold if entropy_threshold is not None else max_entropy * 0.60
        )
        logger.info(
            "WorkspaceAnalyzer: kurtosis_threshold=%.2f  entropy_threshold=%.2f nats  "
            "(vocab=%d, max_H=%.2f)",
            self.kurtosis_threshold, self.entropy_threshold, vocab_size, max_entropy,
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

        Active detection: kurtosis of J-lens readout probability distribution.
        High kurtosis = concentrated on one concept = position is "active".
        """
        seq_len = hs.shape[1]

        entropies: List[float] = []
        kurtoses: List[float] = []
        top_concepts_last: List[Tuple[str, float]] = []

        for pos in range(seq_len):
            h = hs[0, pos].to(self.model.device)  # (d_model,)

            # paper readout: W_U(norm(J_l @ h))
            if layer_idx in self.jlens.avg_jacobians:
                logits = self.jlens.readout_logits(h, layer_idx)  # (vocab,)
            else:
                # fallback: direct unembed (old behaviour)
                logits = self.model.unembed(h.unsqueeze(0).unsqueeze(0)).squeeze().detach()

            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            ent = float(scipy_entropy(probs))
            # excess kurtosis (Fisher definition): 0 for Gaussian, >0 means heavy tail/spiky
            kurt = float(scipy_kurtosis(probs, fisher=True, bias=False))

            entropies.append(ent)
            kurtoses.append(kurt)

            if pos == seq_len - 1:
                top_concepts_last = self.model.top_tokens(logits, k=self.top_k)

        mean_entropy = float(np.mean(entropies))
        mean_kurtosis = float(np.mean(kurtoses))
        # active = positions whose readout is spiky (kurtosis above threshold)
        n_active = int(np.sum(np.array(kurtoses) > self.kurtosis_threshold))

        # variance explained by primary J-space direction at this layer
        j_mat = self.jlens.avg_jacobians.get(layer_idx)
        var_explained = 0.0
        if j_mat is not None:
            hs_flat = hs[0].float().cpu()                      # (seq, d_model)
            # project each hidden state through J, take first principal direction
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
