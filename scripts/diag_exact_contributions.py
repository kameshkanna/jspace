"""
Diagnostic: do exact per-instance forward contributions encode future tokens?

For source position s and offset Δ, computes with batched forward-mode autodiff
(no estimator noise):

    c(s, Δ) = J(s, s+Δ) · h_l,s                  what position s sends to s+Δ
    E(s, Δ) = ||J(s, s+Δ)||_F²  ≈  mean_v ||J(s, s+Δ) v||²   (Rademacher v)

and scores W_U · norm(c) against token s+Δ+1 on all sources, on the top-energy
("broadcast origin") sources, and on the low-energy half.  If future tokens
were carried by a few broadcast positions, averaging over all sources would
hide them; this separates the two cases.

Usage:
    python scripts/diag_exact_contributions.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --layers 12 20 \
        --output outputs/qwen05b/exact_contributions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.autograd.forward_ad as fwAD
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jspace.offset_lens import pack_prompts
from jspace.utils import read_prompts, seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Exact forward-contribution lookahead diagnostic.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompts_file", default="configs/prompts.txt")
    p.add_argument("--n_sequences", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--layers", nargs="+", type=int, default=[12, 20])
    p.add_argument("--offsets", nargs="+", type=int, default=[0, 1, 2, 4])
    p.add_argument("--skip_first", type=int, default=16)
    p.add_argument("--n_energy_probes", type=int, default=4)
    p.add_argument("--chunk", type=int, default=8, help="Source positions per forward pass")
    p.add_argument("--top_fraction", type=float, default=0.1)
    p.add_argument("--gpu_mem_fraction", type=float, default=None)
    p.add_argument("--output", default="outputs/exact_contributions.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def block_output(output: object) -> torch.Tensor:
    """Hidden states from a decoder block's output (tensor or tuple)."""
    return output[0] if isinstance(output, tuple) else output


def exact_contributions(
    model: PreTrainedModel,
    ids: torch.Tensor,
    layer: int,
    offsets: Sequence[int],
    skip_first: int,
    n_probes: int,
    chunk: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact contributions and Jacobian energies for every source s ≥ skip_first.

    Returns
    -------
    contributions : torch.Tensor
        (S, K, d) J(s, s+Δ) h_s (zero where s+Δ is out of range).
    energy : torch.Tensor
        (S, K) Hutchinson estimate of ||J(s, s+Δ)||_F².
    hidden : torch.Tensor
        (S, d) the layer-l states at the sources.
    """
    layers = model.model.layers
    store: Dict[str, torch.Tensor] = {}
    grab = layers[layer].register_forward_hook(
        lambda _m, _i, o: store.__setitem__("h", block_output(o).detach())
    )
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    grab.remove()
    h = store["h"]
    seq_len, d = h.shape[1], h.shape[2]
    sources = torch.arange(skip_first, seq_len, device=ids.device)
    offsets_t = torch.tensor(list(offsets), device=ids.device)
    contributions = torch.zeros(len(sources), len(offsets), d, device=ids.device)
    energy = torch.zeros(len(sources), len(offsets), device=ids.device)
    per = 1 + n_probes

    for start in range(0, len(sources), chunk):
        src = sources[start : start + chunk]
        n_src = len(src)
        tangent = torch.zeros(n_src * per, seq_len, d, device=ids.device)
        rows = torch.arange(n_src, device=ids.device) * per
        tangent[rows, src] = h[0, src]
        probe_rows = (rows.unsqueeze(1) + torch.arange(1, per, device=ids.device)).flatten()
        probes = torch.randint(0, 2, (n_src * n_probes, d), generator=generator, device=ids.device)
        tangent[probe_rows, src.repeat_interleave(n_probes)] = probes.float() * 2 - 1

        out: Dict[str, torch.Tensor] = {}
        with fwAD.dual_level():
            dual = fwAD.make_dual(h.expand(n_src * per, -1, -1).contiguous(), tangent)
            swap = layers[layer].register_forward_hook(
                lambda _m, _i, o: (dual,) + tuple(o[1:]) if isinstance(o, tuple) else dual
            )
            read = layers[-1].register_forward_hook(
                lambda _m, _i, o: out.__setitem__("t", fwAD.unpack_dual(block_output(o)).tangent)
            )
            try:
                with torch.no_grad():
                    model.model(input_ids=ids.expand(n_src * per, -1), use_cache=False)
            finally:
                swap.remove()
                read.remove()
        tangents = out["t"].view(n_src, per, seq_len, d)
        targets = src.unsqueeze(1) + offsets_t.unsqueeze(0)  # (n_src, K)
        valid = targets < seq_len
        idx = targets.clamp(max=seq_len - 1)
        gathered = tangents.gather(2, idx.view(n_src, 1, -1, 1).expand(-1, per, -1, d))
        contributions[start : start + n_src] = gathered[:, 0] * valid.unsqueeze(-1)
        energy[start : start + n_src] = gathered[:, 1:].pow(2).sum(-1).mean(1) * valid
    return contributions, energy, h[0, sources]


def main() -> None:
    """Run the diagnostic and write a JSON summary."""
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpu_mem_fraction is not None and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .float()
        .eval()
    )
    norm, unembed = model.model.norm, model.lm_head
    sequences = pack_prompts(
        read_prompts(args.prompts_file), tokenizer, args.seq_len, min_length=args.seq_len
    )[: args.n_sequences]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    logger.info("%d sequences of %d tokens on %s", len(sequences), args.seq_len, device)

    summary: Dict[str, object] = {"model": args.model, "offsets": args.offsets, "layers": {}}
    for layer in args.layers:
        rows: Dict[int, List[torch.Tensor]] = {k: [] for k in range(len(args.offsets))}
        for ids in tqdm(sequences, desc=f"layer {layer}", dynamic_ncols=True):
            ids = ids.to(device)
            contrib, energy, _ = exact_contributions(
                model,
                ids,
                layer,
                args.offsets,
                args.skip_first,
                args.n_energy_probes,
                args.chunk,
                generator,
            )
            with torch.no_grad():
                pred = unembed(norm(contrib)).argmax(-1)  # (S, K)
            src = torch.arange(args.skip_first, ids.shape[1], device=device)
            for k, delta in enumerate(args.offsets):
                ok = src + delta + 1 < ids.shape[1]
                hit = pred[ok, k] == ids[0, src[ok] + delta + 1]
                rows[k].append(torch.stack([energy[ok, k], hit.float()], 1).cpu())
        per_offset = {}
        for k, delta in enumerate(args.offsets):
            table = torch.cat(rows[k])
            n_top = max(1, int(args.top_fraction * len(table)))
            order = table[:, 0].argsort(descending=True)
            per_offset[str(delta)] = {
                "n": len(table),
                "acc_all": float(table[:, 1].mean()),
                "acc_top_energy": float(table[order[:n_top], 1].mean()),
                "acc_low_energy_half": float(table[order[len(table) // 2 :], 1].mean()),
                "energy_top_over_median": float(
                    table[order[:n_top], 0].mean() / table[:, 0].median()
                ),
            }
            logger.info("L%d Δ=%d: %s", layer, delta, per_offset[str(delta)])
        summary["layers"][str(layer)] = per_offset

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
