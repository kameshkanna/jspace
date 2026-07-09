"""
CLI: Run workspace analysis on a prompt using a pre-fitted J-lens.

Usage:
    # Step 1 — fit J-lens for your target model (run once per model):
    python scripts/compute_jlens.py --model Qwen/Qwen2.5-7B-Instruct --output outputs/jlens_qwen7b.pt
    python scripts/compute_jlens.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --output outputs/jlens_llama3_8b.pt

    # Step 2 — analyse prompts:
    python scripts/run_workspace.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --jlens outputs/jlens_qwen7b.pt \
        --prompts_file configs/lang_pivot_prompts.txt \
        --save_dir outputs/figs_qwen7b/

    # Llama 3.1 8B (language pivot comparison):
    python scripts/run_workspace.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --jlens outputs/jlens_llama3_8b.pt \
        --prompts_file configs/lang_pivot_prompts.txt \
        --save_dir outputs/figs_llama3_8b/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless Lambda Labs
import matplotlib.pyplot as plt

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
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model ID")
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
        "--kurtosis_threshold",
        type=float,
        default=0.0,
        help="Excess kurtosis threshold for active position detection (paper method). "
             "Default 0 = heavier-tailed than Gaussian counts as active.",
    )
    p.add_argument(
        "--entropy_threshold",
        type=float,
        default=None,
        help="Optional entropy upper bound (nats) — used for phase detection only. "
             "Default: auto (60%% of max vocab entropy).",
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
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
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
        kurtosis_threshold=args.kurtosis_threshold,
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
        fig = plot_layer_profile(report, save_path=save_dir / f"layer_profile_{slug}.png")
        plt.close(fig)
        if report.workspace_layers:
            fig2 = plot_concept_heatmap(report, save_path=save_dir / f"concept_heatmap_{slug}.png")
            plt.close(fig2)

        logger.info(
            "  → workspace layers %d–%d | peak capacity %d",
            report.workspace_start,
            report.workspace_end,
            report.peak_capacity,
        )

    # summary capacity comparison
    if len(reports) > 1:
        labels = [p[:30] for p in prompts]
        fig3 = plot_capacity_comparison(
            reports,
            labels=labels,
            save_path=save_dir / "capacity_comparison.png",
        )
        plt.close(fig3)
        logger.info("Saved capacity comparison to %s", save_dir / "capacity_comparison.png")

    logger.info("All figures saved to %s", save_dir)


if __name__ == "__main__":
    main()
