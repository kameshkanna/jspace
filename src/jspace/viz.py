"""Visualization utilities for J-lens and workspace analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from jspace.workspace import WorkspaceReport


# ── palette ─────────────────────────────────────────────────────────────────

_PHASE_COLORS = {"Early": "#7f8c8d", "Workspace": "#27ae60", "Output": "#e67e22"}


# ── layer profile ────────────────────────────────────────────────────────────

def plot_layer_profile(
    report: WorkspaceReport,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Three-panel layer profile:
      (a) n_active concepts across layers
      (b) mean output-distribution entropy across layers
      (c) variance explained by J-space direction
    Workspace zone is shaded green.
    """
    stats = report.layer_stats
    layers = [s.layer_idx for s in stats]
    n_active = [s.n_active for s in stats]
    entropy = [s.mean_entropy for s in stats]
    var_exp = [s.var_explained for s in stats]
    phases = [s.phase for s in stats]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(f"Workspace Profile\n"{report.prompt[:70]}"", fontsize=11)

    # shade workspace zone across all panels
    ws_start = report.workspace_start
    ws_end = report.workspace_end

    for ax, y, ylabel, title in zip(
        axes,
        [n_active, entropy, var_exp],
        ["n active positions", "mean entropy (nats)", "variance explained"],
        ["(a) Active Concepts", "(b) Distribution Entropy", "(c) J-space Variance"],
    ):
        colors = [_PHASE_COLORS.get(p, "gray") for p in phases]
        ax.bar(layers, y, color=colors, alpha=0.8, width=0.9)
        if ws_start >= 0 and ws_end >= 0:
            ax.axvspan(ws_start - 0.5, ws_end + 0.5, color="#27ae60", alpha=0.08, zorder=0)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
        sns.despine(ax=ax)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=p) for p, c in _PHASE_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── concept heatmap ──────────────────────────────────────────────────────────

def plot_concept_heatmap(
    report: WorkspaceReport,
    layer_indices: Optional[List[int]] = None,
    top_n: int = 15,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Heatmap of top-concept scores across layers × top-N tokens.

    Only workspace-zone layers are shown by default.
    """
    ws_stats = (
        report.workspace_layers
        if layer_indices is None
        else [s for s in report.layer_stats if s.layer_idx in layer_indices]
    )
    if not ws_stats:
        ws_stats = report.layer_stats

    # collect unique top tokens
    all_tokens: Dict[str, int] = {}
    for s in ws_stats:
        for tok, _ in s.top_concepts[:top_n]:
            if tok not in all_tokens:
                all_tokens[tok] = len(all_tokens)

    tokens_list = list(all_tokens.keys())
    matrix = np.zeros((len(ws_stats), len(tokens_list)))
    for i, s in enumerate(ws_stats):
        for tok, score in s.top_concepts[:top_n]:
            j = all_tokens.get(tok)
            if j is not None:
                matrix[i, j] = score

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.heatmap(
        matrix,
        xticklabels=tokens_list,
        yticklabels=[f"L{s.layer_idx}" for s in ws_stats],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "probability", "shrink": 0.6},
        linewidths=0.3,
        linecolor="#cccccc",
    )
    ax.set_title(f"Active Concepts Across Workspace Layers\n"{report.prompt[:60]}"", fontsize=11)
    ax.set_xlabel("Token", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── capacity summary bar ─────────────────────────────────────────────────────

def plot_capacity_comparison(
    reports: List[WorkspaceReport],
    labels: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Bar chart comparing peak workspace capacity across multiple prompts.
    Useful for checking capacity vs your manifold ~512 geometric slots.
    """
    capacities = [r.peak_capacity for r in reports]
    xs = list(range(len(reports)))
    labs = labels if labels else [f"P{i}" for i in xs]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bars = ax.bar(xs, capacities, color="#27ae60", alpha=0.85, width=0.6)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Peak active concepts (n_active)", fontsize=10)
    ax.set_title("Workspace Capacity per Prompt", fontsize=11)
    ax.axhline(25, color="#e74c3c", linestyle="--", alpha=0.6, label="paper: ~25 cap")
    ax.axhline(512, color="#3498db", linestyle=":", alpha=0.5, label="manifold: ~512 geometric slots")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
