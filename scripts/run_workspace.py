"""
CLI: Run workspace analysis on a prompt using a pre-fitted J-lens.

Usage:
    # First fit J-lens (once):
    python scripts/compute_jlens.py --model Qwen/Qwen2.5-3B-Instruct

    # Then analyse any prompt:
    python scripts/run_workspace.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --jlens outputs/jlens.pt \
        --prompt "A spider has 8 legs, so three spiders have" \
        --save_dir outputs/figs/

    # Batch mode (one prompt per line in a file):
    python scripts/run_workspace.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --jlens outputs/jlens.pt \
        --prompts_file configs/prompts.txt \
        --save_dir outputs/figs/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless Lambda Labs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.jlens import JacobianLens
from jspace.model import HookedModel
from jspace.workspace import WorkspaceAnalyzer
from jspace.viz import plot_capacity_comparison, plot_concept_heatmap, plot_layer_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Workspace analysis using a fitted J-lens.")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model ID")
    p.add_argument("--jlens", default="outputs/jlens.pt", help="Path to saved J-lens .pt file")
    p.add_argument("--prompt", action="append", default=None, dest="prompts_inline", help="Prompt to analyse (can be repeated)")
    p.add_argument("--prompts_file", default=None, help="File with one prompt per line")
    p.add_argument(
        "--save_dir",
        default="outputs/figs",
        help="Directory to save figures (PNG)",
    )
    p.add_argument("--max_length", type=int, default=128, help="Token truncation length")
    p.add_argument(
        "--entropy_threshold",
        type=float,
        default=None,
        help="Max entropy (nats) to count a position as active. Default: auto (60%% of max vocab entropy)",
    )
    p.add_argument(
        "--var_threshold",
        type=float,
        default=0.05,
        help="Min variance explained by J-space for workspace label",
    )
    p.add_argument("--top_k", type=int, default=10, help="Top-k vocabulary tokens to report")
    return p.parse_args()


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompts_inline:
        prompts.extend(args.prompts_inline)
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.exists():
            logger.error("Prompts file not found: %s", path)
        else:
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            prompts.extend(lines)
    if not prompts:
        prompts = [
            "A spider has 8 legs, so three spiders have",
            "The capital of France is",
            "I feel very happy today because",
        ]
        logger.info("No prompts provided — using built-in examples")
    return prompts


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    jlens_path = Path(args.jlens)
    if not jlens_path.exists():
        logger.error(
            "J-lens file not found: %s\nRun compute_jlens.py first.", jlens_path
        )
        sys.exit(1)

    model = HookedModel(model_id=args.model)
    jlens = JacobianLens.load(jlens_path, model=model)

    analyzer = WorkspaceAnalyzer(
        jlens=jlens,
        entropy_threshold=args.entropy_threshold,
        var_threshold=args.var_threshold,
        top_k=args.top_k,
    )

    prompts = collect_prompts(args)
    reports = []

    for i, prompt in enumerate(prompts):
        logger.info("[%d/%d] Analysing: %s", i + 1, len(prompts), prompt[:70])
        report = analyzer.analyse(prompt, max_length=args.max_length)
        reports.append(report)
        WorkspaceAnalyzer.print_report(report)

        # per-prompt figures
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        plot_layer_profile(report, save_path=save_dir / f"layer_profile_{slug}.png")
        if report.workspace_layers:
            plot_concept_heatmap(report, save_path=save_dir / f"concept_heatmap_{slug}.png")

        logger.info(
            "  → workspace layers %d–%d | peak capacity %d",
            report.workspace_start,
            report.workspace_end,
            report.peak_capacity,
        )

    # summary capacity comparison
    if len(reports) > 1:
        labels = [p[:30] for p in prompts]
        plot_capacity_comparison(
            reports,
            labels=labels,
            save_path=save_dir / "capacity_comparison.png",
        )
        logger.info("Saved capacity comparison to %s", save_dir / "capacity_comparison.png")

    logger.info("All figures saved to %s", save_dir)


if __name__ == "__main__":
    main()
