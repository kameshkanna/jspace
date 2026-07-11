"""
CLI: Fit J-lens (d_model x d_model) Jacobian matrices over a prompt corpus.

Paper formula:  J_l = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]
Approximated via Hutchinson VJP estimator with n_proj Rademacher vectors.

Usage:
    # Qwen 7B (default) — paper quality (1000 prompts, 32 proj):
    python scripts/compute_jlens.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output outputs/jlens_qwen7b.pt

    # Llama 3.1 8B:
    python scripts/compute_jlens.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output outputs/jlens_llama3_8b.pt

    # Fast smoke-test (reduced corpus):
    python scripts/compute_jlens.py --n_prompts 100 --n_proj 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# ensure src/ is on path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.jlens import JacobianLens
from jspace.model import HookedModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── default diverse prompt corpus ───────────────────────────────────────────

DEFAULT_PROMPTS: list[str] = [
    # factual / world knowledge
    "The capital of France is",
    "Water boils at 100 degrees",
    "The largest planet in the solar system is",
    "Albert Einstein was born in",
    "The speed of light is approximately",
    "DNA stands for",
    "The Great Wall of China was built to",
    "The human brain has approximately",
    "Photosynthesis converts sunlight into",
    "The deepest ocean trench is",
    # reasoning / math
    "A spider has 8 legs, so three spiders have",
    "If today is Monday then tomorrow is",
    "The square root of 144 is",
    "2 plus 2 equals",
    "There are 7 days in a week, so 3 weeks have",
    # language / syntax
    "She went to the store and bought",
    "The quick brown fox jumped over",
    "Once upon a time in a land far away",
    "To be or not to be, that is",
    "It was the best of times, it was the",
    # emotions / social
    "I feel very happy today because",
    "The news made everyone feel sad because",
    "Anger is an emotion that",
    "Love is defined as",
    "Fear causes the body to",
    # technical
    "In Python, a list is defined using",
    "Neural networks learn by",
    "The transformer architecture uses attention to",
    "Gradient descent minimizes the",
    "A function is called recursive when",
    # miscellaneous
    "The first human to walk on the moon was",
    "Climate change is caused primarily by",
    "The internet was invented by",
    "Classical music composers include",
    "The Olympics are held every",
]


def load_prompts(prompts_file: Path, n_prompts: int) -> list[str]:
    if prompts_file.exists():
        lines = [l.strip() for l in prompts_file.read_text().splitlines() if l.strip()]
        logger.info("Loaded %d prompts from %s", len(lines), prompts_file)
        return lines[:n_prompts]
    logger.info("No prompts file found — using %d built-in diverse prompts", len(DEFAULT_PROMPTS))
    return DEFAULT_PROMPTS[:n_prompts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit J-lens Jacobians and save to disk.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model ID")
    p.add_argument("--prompts_file", default="configs/prompts.txt", help="One prompt per line")
    p.add_argument("--output", default="outputs/jlens.pt", help="Output .pt file")
    p.add_argument("--n_prompts", type=int, default=1000, help="Max prompts to use (paper: 1000)")
    p.add_argument("--max_length", type=int, default=128, help="Token truncation length")
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to analyse (default: all)",
    )
    p.add_argument(
        "--n_proj",
        type=int,
        default=32,
        help="Hutchinson projections per (prompt, layer) for J matrix estimation. "
             "32=paper-quality; 16=fast.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts_file), args.n_prompts)
    logger.info("Using %d prompts", len(prompts))

    model = HookedModel(model_id=args.model)
    layer_indices = args.layers  # None = all

    jlens = JacobianLens(
        model=model,
        layer_indices=layer_indices,
        n_proj=args.n_proj,
    )

    jlens.fit(prompts, max_length=args.max_length, show_progress=True)
    jlens.save(output_path, n_prompts=len(prompts), max_length=args.max_length)

    logger.info("Done. Saved to %s", output_path)


if __name__ == "__main__":
    main()
