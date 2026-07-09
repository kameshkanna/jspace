"""
Global Workspace analysis.

Given a fitted JacobianLens and a prompt, this module:
  1. Runs the prompt through the model and captures all residual states.
  2. At each layer, projects every token position through J-lens to get
     vocabulary readouts (the "active concepts").
  3. Computes workspace capacity metrics:
       - n_active      : number of positions with interpretable readout (entropy < threshold)
       - var_explained : fraction of hidden-state variance in J-space
       - layer_phase   : Early / Workspace / Output classification
  4. Detects the workspace window (layers where n_active is high and stable).

Terminology from the paper:
  J-space       — the subspace spanned by J-lens vectors across positions
  workspace     — layers where J-space carries verbalizable, causal content
  capacity      — number of distinct concepts simultaneously active
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy

from jspace.jlens import JacobianLens
from jspace.model import HookedModel

logger = logging.getLogger(__name__)


@dataclass
class LayerWorkspaceStats:
    """Per-layer workspace statistics."""

    layer_idx: int
    n_active: int                          # positions with clean J-lens readout
    mean_entropy: float                    # mean token-distribution entropy across positions
    var_explained: float                   # fraction of hidden-state variance in J-space
    top_concepts: List[Tuple[str, float]]  # top active concepts at last position
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
        """Maximum n_active observed across workspace layers."""
        ws = self.workspace_layers
        return max(s.n_active for s in ws) if ws else 0


class WorkspaceAnalyzer:
    """
    Analyses the global workspace properties of a model on a given prompt.

    Parameters
    ----------
    jlens : JacobianLens
        Fitted J-lens (call jlens.fit() first).
    entropy_threshold : float
        Positions with J-space readout entropy below this are "active".
        Lower = stricter.  Default 3.0 (nats) works for 32k–150k vocab models.
    var_threshold : float
        Minimum fraction of variance in J-space for a layer to qualify
        as "Workspace" phase.
    min_active_fraction : float
        Fraction of sequence positions that must be active for a layer
        to be in the workspace zone.
    top_k : int
        Number of vocabulary tokens to report per position.
    """

    def __init__(
        self,
        jlens: JacobianLens,
        entropy_threshold: Optional[float] = None,
        var_threshold: float = 0.05,
        min_active_fraction: float = 0.3,
        top_k: int = 10,
    ) -> None:
        self.jlens = jlens
        self.model: HookedModel = jlens.model
        self.var_threshold = var_threshold
        self.min_active_fraction = min_active_fraction
        self.top_k = top_k

        # auto-scale entropy threshold to vocab size:
        # a "sharp" distribution should sit at ~25% of max entropy
        import math
        vocab_size = self.model.vocab_size()
        max_entropy = math.log(vocab_size)  # nats
        self.entropy_threshold = entropy_threshold if entropy_threshold is not None else max_entropy * 0.60
        logger.info(
            "Entropy threshold: %.2f nats  (vocab=%d, max_entropy=%.2f)",
            self.entropy_threshold, vocab_size, max_entropy,
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

    def analyse_batch(
        self,
        prompts: List[str],
        max_length: int = 128,
    ) -> List[WorkspaceReport]:
        """Run analyse() on multiple prompts."""
        return [self.analyse(p, max_length=max_length) for p in prompts]

    # ── internals ───────────────────────────────────────────────────────────

    def _compute_layer_stats(
        self,
        layer_idx: int,
        hs: torch.Tensor,
        tokens: List[str],
    ) -> LayerWorkspaceStats:
        """
        Compute workspace stats for one layer.

        hs : (1, seq, d_model)
        """
        seq_len = hs.shape[1]

        entropies: List[float] = []
        top_concepts_last: List[Tuple[str, float]] = []

        for pos in range(seq_len):
            h = hs[0, pos].to(self.model.device)  # (d_model,)
            logits = self.model.unembed(h.unsqueeze(0).unsqueeze(0)).squeeze().detach()  # (vocab,)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            ent = float(scipy_entropy(probs))
            entropies.append(ent)
            if pos == seq_len - 1:
                top_concepts_last = self.model.top_tokens(logits, k=self.top_k)

        mean_entropy = float(np.mean(entropies))
        n_active = int(np.sum(np.array(entropies) < self.entropy_threshold))

        # variance explained by J-space direction
        j_vec = self.jlens.avg_jacobians.get(layer_idx)
        var_explained = 0.0
        if j_vec is not None:
            hs_flat = hs[0].float().cpu()  # (seq, d_model)
            j_unit = F.normalize(j_vec.float().cpu(), dim=0)  # (d_model,)
            proj = hs_flat @ j_unit  # (seq,)
            var_proj = float(proj.var())
            var_total = float(hs_flat.var())
            var_explained = var_proj / (var_total + 1e-8)

        return LayerWorkspaceStats(
            layer_idx=layer_idx,
            n_active=n_active,
            mean_entropy=mean_entropy,
            var_explained=var_explained,
            top_concepts=top_concepts_last,
        )

    def _assign_phases(self, stats: List[LayerWorkspaceStats]) -> None:
        """
        Label each layer as Early / Workspace / Output using the hunchback
        entropy profile from the paper.

        Strategy:
          1. Find the entropy minimum (most interpretable zone) — that anchors
             the workspace centre.
          2. Expand outward while entropy stays within a tolerance of the min.
          3. The final 10% of layers are always Output (next-token collapse).
          4. Everything else is Early.
        """
        if not stats:
            return

        n_layers = len(stats)
        entropies = np.array([s.mean_entropy for s in stats])
        output_cutoff = int(n_layers * 0.90)

        min_ent = entropies.min()
        max_ent = entropies.max()
        ent_range = max(max_ent - min_ent, 1e-6)

        # a layer is "workspace-like" if its entropy is within 40% of the
        # full range above the minimum AND it is not in the output zone
        tolerance = ent_range * 0.40
        for i, s in enumerate(stats):
            if s.layer_idx >= output_cutoff:
                s.phase = "Output"
            elif entropies[i] <= min_ent + tolerance:
                s.phase = "Workspace"
            else:
                s.phase = "Early"

    def _detect_workspace_window(
        self, stats: List[LayerWorkspaceStats]
    ) -> Tuple[int, int]:
        """Return (start_layer, end_layer) of the workspace zone."""
        ws_indices = [s.layer_idx for s in stats if s.phase == "Workspace"]
        if not ws_indices:
            return -1, -1
        return min(ws_indices), max(ws_indices)

    # ── pretty printing ──────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: WorkspaceReport, max_layers: int = 30) -> None:
        """Print a condensed human-readable workspace report."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f'Workspace Analysis -- "{report.prompt[:60]}"')
            table.add_column("Layer", justify="right")
            table.add_column("Phase", justify="center")
            table.add_column("n_active", justify="right")
            table.add_column("entropy", justify="right")
            table.add_column("var_expl", justify="right")
            table.add_column("Top concept", justify="left")

            step = max(1, len(report.layer_stats) // max_layers)
            for s in report.layer_stats[::step]:
                phase_color = {"Early": "dim", "Workspace": "green", "Output": "yellow"}.get(
                    s.phase, "white"
                )
                top_tok = s.top_concepts[0][0] if s.top_concepts else "—"
                table.add_row(
                    str(s.layer_idx),
                    f"[{phase_color}]{s.phase}[/{phase_color}]",
                    str(s.n_active),
                    f"{s.mean_entropy:.2f}",
                    f"{s.var_explained:.3f}",
                    top_tok,
                )
            console.print(table)
            console.print(
                f"Workspace window: layers {report.workspace_start}–{report.workspace_end}  "
                f"| Peak capacity: {report.peak_capacity} active positions"
            )
        except ImportError:
            # fallback without rich
            print(f"\nWorkspace Analysis: {report.prompt[:60]}")
            for s in report.layer_stats[::max(1, len(report.layer_stats) // max_layers)]:
                top_tok = s.top_concepts[0][0] if s.top_concepts else "—"
                print(
                    f"  L{s.layer_idx:3d} [{s.phase:9s}]  "
                    f"n_active={s.n_active:3d}  H={s.mean_entropy:.2f}  "
                    f"var={s.var_explained:.3f}  top={top_tok}"
                )
            print(
                f"\nWorkspace: layers {report.workspace_start}–{report.workspace_end}  "
                f"peak_capacity={report.peak_capacity}"
            )
