"""
CLI: fit and evaluate the multi-token (offset-resolved) J++ lens against baselines.

Pipeline
--------
1. Pack the corpus into fixed-length sequences and split them into fit / dev / test.
2. Fit offset expert Jacobians twice on the fit split: plain gradients (J) and LRP (R).
3. Learn expert weights per (layer, Δ) on dev items (future-token recall@k).
4. Score on test items, recall@1 / @10 of token p+Δ+1 over the semantic vocabulary:
     J            plain gradients, pooled              (offset J-lens)
     J+experts    plain gradients, learned weights     (Jacobian Filtering only)
     R            LRP, pooled                          (offset R-lens)
     J++          LRP, learned weights                 (offset J++ lens)
     ridge        least-squares h_l,p → h_final,p+Δ    (linear Future-Lens ceiling)
     next-token   the model's own top-k at p           (repetition baseline; Δ=0 = model)
     unigram      most frequent fit-split tokens

Usage:
    python scripts/run_offset_jpp.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --layers 6 10 14 18 21 \
        --save_dir outputs/qwen05b/offset_jpp
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.expert_offset_lens import ExpertOffsetConfig, ExpertOffsetLens
from jspace.expert_weights import WeightSearchConfig, expert_logit_cache, fit_expert_weights
from jspace.lookahead import (
    HiddenBank,
    collect_hidden,
    lookahead_items,
    readout_recall,
    ridge_future_maps,
    topk_hits,
    unigram_topk,
)
from jspace.model import HookedModel
from jspace.offset_lens import pack_prompts
from jspace.readout_filter import semantic_token_mask
from jspace.utils import read_prompts, seed_everything
from jspace.viz import plot_offset_recall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Fit and evaluate the offset J++ lens.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model ID or path")
    p.add_argument("--prompts_file", default="configs/prompts.txt")
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--layers", nargs="+", type=int, required=True, help="Source layers")
    p.add_argument("--offsets", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--projection_dim", type=int, default=64)
    p.add_argument("--rows_per_pass", type=int, default=16)
    p.add_argument("--n_sign_draws", type=int, default=2)
    p.add_argument("--skip_first", type=int, default=16)
    p.add_argument("--fit_fraction", type=float, default=0.7)
    p.add_argument("--dev_fraction", type=float, default=0.15)
    p.add_argument("--dev_items", type=int, default=160, help="Items per (layer, Δ) for weights")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--weight_steps", type=int, default=200)
    p.add_argument("--no_plain", action="store_true", help="Skip the plain-gradient (J) fit")
    p.add_argument(
        "--gpu_mem_fraction",
        type=float,
        default=None,
        help="Cap this process's share of GPU memory (protects other jobs)",
    )
    p.add_argument("--save_dir", default="outputs/offset_jpp")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_sequences(
    sequences: List[torch.Tensor], fit_fraction: float, dev_fraction: float, seed: int
) -> Dict[str, List[torch.Tensor]]:
    """Deterministic shuffled fit / dev / test split; every split is non-empty."""
    n = len(sequences)
    if n < 3:
        raise ValueError(f"need at least 3 sequences, got {n}")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_fit = max(1, min(n - 2, round(fit_fraction * n)))
    n_dev = max(1, min(n - n_fit - 1, round(dev_fraction * n)))
    pick = lambda idx: [sequences[i] for i in idx]  # noqa: E731
    return {
        "fit": pick(order[:n_fit]),
        "dev": pick(order[n_fit : n_fit + n_dev]),
        "test": pick(order[n_fit + n_dev :]),
    }


def fit_weights(
    lens: ExpertOffsetLens,
    dev: HiddenBank,
    rankable: torch.Tensor,
    norm_weight: torch.Tensor,
    unembedding: torch.Tensor,
    n_items: int,
    search: WeightSearchConfig,
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Learn (K, E) expert weights per layer on dev future-token items."""
    generator = torch.Generator().manual_seed(seed)
    counts = lens.count_weights()
    weights: Dict[int, torch.Tensor] = {}
    jobs = [
        (layer, k, delta) for layer in lens.layer_indices for k, delta in enumerate(lens.offsets)
    ]
    per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in lens.layer_indices}
    for layer, k, delta in tqdm(
        jobs, desc=f"expert weights (lrp={lens.config.lrp})", dynamic_ncols=True
    ):
        items = lookahead_items(dev.ids, delta, lens.config.skip_first, rankable)
        if len(items) == 0:
            per_layer[layer].append(counts[layer][k])
            continue
        items = items.subset(torch.randperm(len(items), generator=generator)[:n_items])
        hidden = dev.layers[layer][items.seq, items.pos]
        cache = expert_logit_cache(
            lens.expert_transports(hidden, layer, k), norm_weight, unembedding
        )
        fitted = fit_expert_weights(
            cache, items.target, rankable, search, extra_starts={"counts": counts[layer][k]}
        )
        per_layer[layer].append(fitted.weights)
        del cache
    for layer, rows in per_layer.items():
        weights[layer] = torch.stack(rows)
    return weights


def evaluate(
    lenses: Dict[str, tuple[ExpertOffsetLens, str]],
    ridge: Dict[int, torch.Tensor],
    unigram: torch.Tensor,
    test: HiddenBank,
    layers: Sequence[int],
    offsets: Sequence[int],
    skip_first: int,
    rankable: torch.Tensor,
    norm_weight: torch.Tensor,
    unembedding: torch.Tensor,
    k: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Recall@1 and @k of every method on every (layer, Δ); arrays are (n_layers, K)."""
    methods = [*lenses, "ridge", "next-token", "unigram"]
    out = {
        m: {f"r@{kk}": np.full((len(layers), len(offsets)), np.nan) for kk in (1, k)}
        for m in methods
    }
    n_items = np.zeros(len(offsets), dtype=int)
    for j, delta in enumerate(offsets):
        items = lookahead_items(test.ids, delta, skip_first, rankable)
        n_items[j] = len(items)
        if len(items) == 0:
            continue
        top_next = test.next_top[items.seq, items.pos]
        top_uni = unigram.unsqueeze(0).expand(len(items), -1)
        for i, layer in enumerate(layers):
            hidden = test.layers[layer][items.seq, items.pos]
            transports = {
                name: lens.project(hidden, layer, mode)[j] for name, (lens, mode) in lenses.items()
            }
            transports["ridge"] = hidden.to(ridge[layer].device) @ ridge[layer][j].T
            for name, vectors in transports.items():
                scores = readout_recall(
                    vectors, items.target, rankable, norm_weight, unembedding, ks=(1, k)
                )
                out[name]["r@1"][i, j], out[name][f"r@{k}"][i, j] = scores[1], scores[k]
            for name, top in (("next-token", top_next), ("unigram", top_uni)):
                out[name]["r@1"][i, j] = float(topk_hits(top[:, :1], items.target).float().mean())
                out[name][f"r@{k}"][i, j] = float(
                    topk_hits(top[:, :k], items.target).float().mean()
                )
    out["_n_items"] = {"per_offset": n_items}
    return out


def main() -> None:
    """Run the full fit-and-evaluate pipeline."""
    args = parse_args()
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = HookedModel(model_id=args.model)
    if args.gpu_mem_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    prompts = read_prompts(args.prompts_file, limit=args.n_prompts)
    sequences = pack_prompts(prompts, model.tokenizer, args.max_length, min_length=args.max_length)
    splits = split_sequences(sequences, args.fit_fraction, args.dev_fraction, args.seed)
    logger.info(
        "Sequences of %d tokens — fit %d, dev %d, test %d",
        args.max_length,
        *(len(splits[s]) for s in ("fit", "dev", "test")),
    )

    vocab = int(model.model.config.vocab_size)
    rankable = semantic_token_mask(model.tokenizer, vocab)
    norm_weight = model.model.model.norm.weight.detach()
    unembedding = model.model.lm_head.weight.detach()
    banks = {s: collect_hidden(model, splits[s], args.layers, rankable, k=args.k) for s in splits}

    search = WeightSearchConfig(k=args.k, steps=args.weight_steps)
    lenses: Dict[str, tuple[ExpertOffsetLens, str]] = {}
    reliability: Dict[str, List[List[float]]] = {}
    for lrp in ([True] if args.no_plain else [False, True]):
        config = ExpertOffsetConfig(
            offsets=tuple(args.offsets),
            layer_indices=tuple(args.layers),
            num_experts=args.num_experts,
            projection_dim=args.projection_dim,
            rows_per_pass=args.rows_per_pass,
            n_sign_draws=args.n_sign_draws,
            skip_first=args.skip_first,
            lrp=lrp,
            seed=args.seed,
        )
        lens = ExpertOffsetLens(model, config).fit_routers(splits["fit"]).fit_experts(splits["fit"])
        lens.set_weights(
            fit_weights(
                lens,
                banks["dev"],
                rankable,
                norm_weight,
                unembedding,
                args.dev_items,
                search,
                args.seed,
            )
        )
        lens.save(save_dir / f"expert_offset_lens_lrp{int(lrp)}.pt")
        tag = "R" if lrp else "J"
        lenses[tag] = (lens, "pooled")
        lenses["J++" if lrp else "J+experts"] = (lens, "weighted")
        reliability[tag] = lens.reliability().tolist()

    ridge = {
        layer: m.to(unembedding.device)
        for layer, m in ridge_future_maps(
            banks["fit"], args.layers, args.offsets, args.skip_first
        ).items()
    }
    unigram = unigram_topk(banks["fit"].ids, rankable, args.k)
    results = evaluate(
        lenses,
        ridge,
        unigram,
        banks["test"],
        args.layers,
        args.offsets,
        args.skip_first,
        rankable,
        norm_weight,
        unembedding,
        args.k,
    )

    n_items = results.pop("_n_items")["per_offset"].tolist()
    summary = {
        "model": args.model,
        "layers": args.layers,
        "offsets": args.offsets,
        "k": args.k,
        "n_sequences": {s: len(v) for s, v in splits.items()},
        "test_items_per_offset": n_items,
        "split_half_cosine": reliability,
        "recall": {m: {key: v.tolist() for key, v in r.items()} for m, r in results.items()},
    }
    (save_dir / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    fig = plot_offset_recall(
        {m: r[f"r@{args.k}"] for m, r in results.items()},
        args.layers,
        args.offsets,
        k=args.k,
        save_path=save_dir / "offset_recall.png",
    )
    plt.close(fig)

    header = f"{'method':<11}" + "".join(f"  Δ={d}: r@1  r@{args.k}" for d in args.offsets)
    logger.info(
        "Best layer per offset (test items per offset: %s)\n%s\n%s",
        n_items,
        header,
        "\n".join(
            f"{m:<11}"
            + "".join(
                (
                    f"     {np.nanmax(r['r@1'][:, j]):.3f} {np.nanmax(r[f'r@{args.k}'][:, j]):.3f}"
                    if not math.isnan(np.nanmax(r["r@1"][:, j]))
                    else "       -     -"
                )
                for j in range(len(args.offsets))
            )
            for m, r in results.items()
        ),
    )
    logger.info("Saved results.json, offset_recall.png and lens files to %s", save_dir)


if __name__ == "__main__":
    main()
