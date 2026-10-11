"""Visualization utilities for J-lens and workspace analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# suppress missing-glyph warnings for CJK/non-latin tokens
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

from matplotlib.colors import LinearSegmentedColormap

from jspace.multitoken import MultiTokenReport
from jspace.workspace import WorkspaceReport


_PHASE_COLORS = {"Early": "#7f8c8d", "Workspace": "#27ae60", "Output": "#e67e22"}

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("jspace_blue", _BLUE_RAMP)
_SERIES = ["#2a78d6", "#eb6834"]
_GRID = "#d9d9d6"


def _ordinal_blues(n: int) -> List[str]:
    """n ordered steps of the blue ramp, starting at step 250 so the lightest line stays visible."""
    ramp = _BLUE_RAMP[3:]
    idx = np.linspace(0, len(ramp) - 1, max(n, 1)).round().astype(int)
    return [ramp[i] for i in idx]


def _safe_label(tok: str) -> str:
    """Escape matplotlib special characters in token strings used as tick labels."""
    return tok.replace("\\", "\\\\").replace("$", r"\$").replace("_", r"\_").replace("^", r"\^").replace("{", r"\{").replace("}", r"\}")


def plot_layer_profile(
    report: WorkspaceReport,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Three-panel layer profile: n_active, entropy, variance explained.
    Workspace zone is shaded green.
    """
    stats = report.layer_stats
    layers = [s.layer_idx for s in stats]
    n_active = [s.n_active for s in stats]
    entropy = [s.mean_entropy for s in stats]
    var_exp = [s.var_explained for s in stats]
    phases = [s.phase for s in stats]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(f'Workspace Profile -- "{_safe_label(report.prompt[:70])}"', fontsize=11)

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

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=p) for p, c in _PHASE_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_concept_heatmap(
    report: WorkspaceReport,
    layer_indices: Optional[List[int]] = None,
    top_n: int = 15,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Heatmap of top-concept scores across layers x top-N tokens."""
    ws_stats = (
        report.workspace_layers
        if layer_indices is None
        else [s for s in report.layer_stats if s.layer_idx in layer_indices]
    )
    if not ws_stats:
        ws_stats = report.layer_stats

    all_tokens: dict[str, int] = {}
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
        xticklabels=[_safe_label(t) for t in tokens_list],
        yticklabels=[f"L{s.layer_idx}" for s in ws_stats],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "probability", "shrink": 0.6},
        linewidths=0.3,
        linecolor="#cccccc",
    )
    ax.set_title(f'Active Concepts Across Workspace Layers -- "{_safe_label(report.prompt[:60])}"', fontsize=11)
    ax.set_xlabel("Token", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_capacity_comparison(
    reports: List[WorkspaceReport],
    labels: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Bar chart comparing peak workspace capacity across multiple prompts."""
    capacities = [r.peak_capacity for r in reports]
    xs = list(range(len(reports)))
    labs = labels if labels else [f"P{i}" for i in xs]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bars = ax.bar(xs, capacities, color="#27ae60", alpha=0.85, width=0.6)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([_safe_label(l) for l in labs], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Peak active concepts (n_active)", fontsize=10)
    ax.set_title("Workspace Capacity per Prompt", fontsize=11)
    ax.axhline(25, color="#e74c3c", linestyle="--", alpha=0.6, label="paper upper bound (~25)")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_temporal_reach(
    reach: np.ndarray,
    reliability: np.ndarray,
    layer_indices: Sequence[int],
    offsets: Sequence[int],
    save_path: Optional[str | Path] = None,
    figsize: tuple = (13, 7),
) -> plt.Figure:
    """
    Two heatmaps over (source layer, offset): relative temporal reach and the
    split-half reliability of each J_l^(Δ) estimate.

    Reach cells show log10(||J_l^(Δ)||_F / ||J_l^(Δ0)||_F) against the first
    offset; cells with zero reach (e.g. the final layer, where J^(Δ>0) = 0) are
    left blank.

    Parameters
    ----------
    reach : np.ndarray
        (n_layers, K) unbiased norms from ``OffsetJacobianLens.temporal_reach()``.
    reliability : np.ndarray
        (n_layers, K) split-half cosines from ``OffsetJacobianLens.reliability()``.
    layer_indices : Sequence[int]
    offsets : Sequence[int]
        At least two offsets; the first is the reference column.
    save_path : str | Path | None
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    reach = np.asarray(reach, dtype=np.float64)
    reliability = np.asarray(reliability, dtype=np.float64)
    expected = (len(layer_indices), len(offsets))
    if reach.shape != expected or reliability.shape != expected:
        raise ValueError(f"reach {reach.shape} and reliability {reliability.shape} must be {expected}")
    if len(offsets) < 2:
        raise ValueError("need at least two offsets to plot relative reach")

    valid = (reach[:, 1:] > 0) & (reach[:, :1] > 0)
    log_rel = np.full(valid.shape, np.nan)
    log_rel[valid] = np.log10((reach[:, 1:] / np.where(reach[:, :1] > 0, reach[:, :1], 1.0))[valid])

    fig, (ax_reach, ax_rel) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    y_labels = [f"L{layer}" for layer in layer_indices]
    sns.heatmap(
        log_rel,
        cmap=_SEQUENTIAL_BLUE,
        xticklabels=[f"Δ={o}" for o in offsets[1:]],
        yticklabels=y_labels,
        cbar_kws={"label": f"log10 ||J^(Δ)|| / ||J^(Δ={offsets[0]})||", "shrink": 0.7},
        linewidths=0.4,
        linecolor="white",
        ax=ax_reach,
    )
    ax_reach.set_title("(a) Temporal reach of layer l at t on the final residual at t+Δ", fontsize=10)

    sns.heatmap(
        reliability,
        cmap=_SEQUENTIAL_BLUE,
        vmin=0.0,
        vmax=1.0,
        annot=len(layer_indices) <= 32,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        xticklabels=[f"Δ={o}" for o in offsets],
        yticklabels=y_labels,
        cbar_kws={"label": "split-half cosine", "shrink": 0.7},
        linewidths=0.4,
        linecolor="white",
        ax=ax_rel,
    )
    ax_rel.set_title("(b) Estimate reliability (1 = signal, 0 = noise)", fontsize=10)

    for ax in (ax_reach, ax_rel):
        ax.set_xlabel("Offset", fontsize=9)
        ax.set_ylabel("Source layer", fontsize=9)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        ax.tick_params(axis="x", labelsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lookahead_profile(
    report: MultiTokenReport,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (13, 4.5),
) -> plt.Figure:
    """
    Two panels: per-offset lookahead accuracy across layers, and joint vs split
    concept counts across layers.

    Parameters
    ----------
    report : MultiTokenReport
    save_path : str | Path | None
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    layers = [s.layer_idx for s in report.layer_stats]
    acc = np.stack([s.lookahead_acc.numpy() for s in report.layer_stats])

    fig, (ax_acc, ax_cnt) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(f'Multi-token J-space -- "{_safe_label(report.prompt[:70])}"', fontsize=11)

    for color, delta, column in zip(_ordinal_blues(len(report.offsets)), report.offsets, acc.T):
        ax_acc.plot(layers, column, color=color, linewidth=2, marker="o", markersize=3.5, label=f"Δ={delta}")
    if report.offsets[0] == 0 and not np.isnan(float(report.baseline_acc[0])):
        ax_acc.axhline(float(report.baseline_acc[0]), color="#52514e", linestyle="--", linewidth=1,
                       label="model output (Δ=0)")
    ax_acc.set_xlabel("Layer", fontsize=9)
    ax_acc.set_ylabel("top-1 accuracy vs actual token t+Δ+1", fontsize=9)
    ax_acc.set_title("(a) Lookahead accuracy", fontsize=10)
    ax_acc.set_ylim(bottom=0)
    ax_acc.legend(frameon=False, fontsize=8, ncol=2)

    if report.layer_stats and report.layer_stats[0].n_joint is not None:
        ax_cnt.plot(layers, [s.n_joint for s in report.layer_stats], color=_SERIES[0],
                    linewidth=2, marker="o", markersize=3.5, label="joint (multi-token atoms)")
        ax_cnt.plot(layers, [s.n_split for s in report.layer_stats], color=_SERIES[1],
                    linewidth=2, marker="o", markersize=3.5, label="split (sum of per-offset atoms)")
        ax_cnt.set_xlabel("Layer", fontsize=9)
        ax_cnt.set_ylabel("mean active concepts per position", fontsize=9)
        ax_cnt.set_title("(b) Concept count", fontsize=10)
        ax_cnt.set_ylim(bottom=0)
        ax_cnt.legend(frameon=False, fontsize=8)
    else:
        ax_cnt.axis("off")
        ax_cnt.text(0.5, 0.5, "concept counts not computed", ha="center", va="center", fontsize=9)

    for ax in (ax_acc, ax_cnt):
        ax.grid(color=_GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
        sns.despine(ax=ax)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


_METHOD_ORDER = ["J", "J+experts", "R", "J++", "ridge", "next-token", "unigram"]
_METHOD_COLORS = dict(zip(_METHOD_ORDER, [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
]))


def plot_offset_recall(
    recall: dict,
    layer_indices: Sequence[int],
    offsets: Sequence[int],
    k: int = 10,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Small multiples of held-out recall@k against layer, one panel per offset.

    Parameters
    ----------
    recall : dict[str, np.ndarray]
        method → (n_layers, K) recall@k.
    layer_indices : Sequence[int]
    offsets : Sequence[int]
    k : int
    save_path : str | Path | None

    Returns
    -------
    plt.Figure
    """
    unknown = sorted(set(recall) - set(_METHOD_COLORS))
    if unknown:
        raise ValueError(f"no colour for methods {unknown}")
    n_panels = len(offsets)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels + 1.5, 3.6),
                             sharey=True, constrained_layout=True, squeeze=False)
    y_max = max(float(np.nanmax(v)) for v in recall.values()) * 1.1 or 1.0
    for j, (ax, delta) in enumerate(zip(axes[0], offsets)):
        for method in [m for m in _METHOD_ORDER if m in recall]:
            values = np.asarray(recall[method])[:, j]
            dashed = method in ("next-token", "unigram")
            ax.plot(layer_indices, values, color=_METHOD_COLORS[method], linewidth=2,
                    linestyle="--" if dashed else "-", marker=None if dashed else "o",
                    markersize=3.5, label=method)
        ax.set_title(f"Δ={delta}  (predicts token t+{delta + 1})", fontsize=10)
        ax.set_xlabel("Source layer", fontsize=9)
        ax.set_xticks(list(layer_indices))
        ax.set_ylim(0, y_max)
        ax.grid(color=_GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    axes[0, 0].set_ylabel(f"held-out recall@{k}", fontsize=9)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Future-token readout by offset: offset J++ vs J / R lenses and baselines", fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
