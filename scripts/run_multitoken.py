"""
CLI: multi-token workspace analysis with a fitted offset J-lens.

For each prompt: latent strings (one readout per offset Δ) at every layer,
lookahead accuracy, and joint vs split concept counts.  Writes a temporal-reach
heatmap, one profile figure per prompt, and summary.json.

Usage:
    python scripts/run_multitoken.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --lens outputs/qwen05b/offset_jlens.pt \
        --prompts_file configs/lang_pivot_prompts.txt \
        --save_dir outputs/qwen05b/multitoken
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.model import HookedModel
from jspace.multitoken import MultiTokenReport, MultiTokenWorkspace
from jspace.offset_lens import OffsetJacobianLens
from jspace.utils import read_prompts
from jspace.viz import plot_lookahead_profile, plot_temporal_reach

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Multi-token workspace analysis with an offset J-lens.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model ID")
    p.add_argument("--lens", required=True, help="Offset J-lens .pt from fit_offset_jlens.py")
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        dest="prompts_inline",
        help="Prompt to analyse (repeatable)",
    )
    p.add_argument("--prompts_file", default=None, help="One prompt per line")
    p.add_argument("--save_dir", default="outputs/multitoken", help="Output directory")
    p.add_argument("--top_k", type=int, default=5, help="Tokens kept per (offset, position)")
    p.add_argument("--coverage", type=float, default=0.95, help="Pursuit coverage target")
    p.add_argument("--no_concepts", action="store_true", help="Skip concept counting")
    return p.parse_args()


def collect_prompts(args: argparse.Namespace) -> List[str]:
    """Gather prompts from --prompt and --prompts_file; at least one is required."""
    prompts = list(args.prompts_inline or [])
    if args.prompts_file:
        prompts.extend(read_prompts(args.prompts_file))
    if not prompts:
        raise SystemExit("Provide --prompt and/or --prompts_file")
    return prompts


def slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug of a prompt prefix."""
    return re.sub(r"[^\w]+", "_", text[:max_len]).strip("_") or "prompt"


def main() -> None:
    """Run the analysis and write figures and a JSON summary."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prompts = collect_prompts(args)

    model = HookedModel(model_id=args.model)
    lens = OffsetJacobianLens.load(args.lens, model)
    analyser = MultiTokenWorkspace(
        lens, top_k=args.top_k, count_concepts=not args.no_concepts, coverage=args.coverage
    )

    reach = lens.temporal_reach()
    fig = plot_temporal_reach(
        reach.numpy(),
        lens.reliability().numpy(),
        lens.layer_indices,
        lens.offsets,
        save_path=save_dir / "temporal_reach.png",
    )
    plt.close(fig)

    reports: List[MultiTokenReport] = []
    for i, prompt in enumerate(tqdm(prompts, desc="multi-token analysis", dynamic_ncols=True)):
        report = analyser.analyse(prompt)
        reports.append(report)
        fig = plot_lookahead_profile(
            report, save_path=save_dir / f"profile_{i:02d}_{slugify(prompt)}.png"
        )
        plt.close(fig)

    for report in reports:
        analyser.print_report(report)

    summary = {
        "model": args.model,
        "lens": str(args.lens),
        "offsets": lens.offsets,
        "sample_counts": lens.sample_counts.tolist(),
        "temporal_reach": dict(zip(map(str, lens.layer_indices), reach.tolist())),
        "split_half_cosine": dict(zip(map(str, lens.layer_indices), lens.reliability().tolist())),
        "reports": [analyser.to_summary(r) for r in reports],
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("Wrote figures and summary.json to %s", save_dir)


if __name__ == "__main__":
    main()
