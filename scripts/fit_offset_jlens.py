"""
CLI: fit offset-resolved (multi-token) J-lens matrices J_l^(Δ) over a prompt corpus.

    J_l^(Δ) = E_{prompt, t} [ ∂h_final_{t+Δ} / ∂h_l_t ]       Δ ∈ --offsets

Usage:
    # Qwen 7B, workspace-adjacent layers (accumulators: n_layers × K × d² × 4 B):
    python scripts/fit_offset_jlens.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layers 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
        --output outputs/qwen7b/offset_jlens.pt

    # Small local run (fits on an 8 GB GPU):
    python scripts/fit_offset_jlens.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --n_prompts 300 --n_proj 16 \
        --output outputs/qwen05b/offset_jlens.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.model import HookedModel
from jspace.offset_lens import OffsetJacobianLens, OffsetLensConfig
from jspace.utils import read_prompts, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_SAVE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Fit offset-resolved J-lens matrices and save to disk.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model ID")
    p.add_argument("--prompts_file", default="configs/prompts.txt", help="One prompt per line")
    p.add_argument("--output", default="outputs/offset_jlens.pt", help="Output .pt file")
    p.add_argument("--n_prompts", type=int, default=1000, help="Max prompts to use")
    p.add_argument("--max_length", type=int, default=128, help="Token truncation length")
    p.add_argument(
        "--layers", nargs="+", type=int, default=None, help="Source layers (default: all)"
    )
    p.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 6, 8],
        help="Target offsets Δ (strictly increasing)",
    )
    p.add_argument("--n_proj", type=int, default=32, help="Backward passes per prompt")
    p.add_argument("--stride", type=int, default=None, help="Comb stride (default: max offset + 1)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--no_pack",
        action="store_true",
        help="Fit on each prompt separately instead of packing into max_length sequences",
    )
    p.add_argument(
        "--save_dtype",
        choices=sorted(_SAVE_DTYPES),
        default="float32",
        help="Storage dtype for Jacobians (bfloat16 halves file size)",
    )
    return p.parse_args()


def main() -> None:
    """Fit and save an offset J-lens."""
    args = parse_args()
    seed_everything(args.seed)
    prompts = read_prompts(args.prompts_file, limit=args.n_prompts)
    logger.info("Using %d prompts from %s", len(prompts), args.prompts_file)

    config = OffsetLensConfig(
        offsets=tuple(args.offsets),
        layer_indices=tuple(args.layers) if args.layers else None,
        n_proj=args.n_proj,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed,
        pack=not args.no_pack,
    )
    model = HookedModel(model_id=args.model)
    lens = OffsetJacobianLens(model, config).fit(prompts, show_progress=True)
    lens.save(args.output, jacobian_dtype=_SAVE_DTYPES[args.save_dtype])

    reach = lens.temporal_reach()
    relative = reach[:, 1:] / reach[:, :1].clamp_min(1e-12)
    for layer, rel_row, rel_cos in zip(
        lens.layer_indices, relative.tolist(), lens.reliability().tolist()
    ):
        logger.info(
            "L%-3d reach/reach[d=%d]: %s | split-half cos: %s",
            layer,
            lens.offsets[0],
            " ".join(f"{r:.3f}" for r in rel_row),
            " ".join(f"{c:.2f}" for c in rel_cos),
        )
    logger.info("Done. Saved to %s", args.output)


if __name__ == "__main__":
    main()
