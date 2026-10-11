"""
Held-out future-token evaluation for offset lenses, with baselines.

An offset-Δ readout at position p predicts token p+Δ+1.  Items are (sequence,
position, target) triples over kept positions whose target is a semantic
(rankable) token.  Besides the lenses, three baselines are scored:

  * unigram          the k most frequent rankable tokens of the fitting text
  * model next-token the model's own top-k next-token prediction at p, scored
                     against token p+Δ+1 (at Δ=0 this is the model itself)
  * ridge            a least-squares linear map h_l,p → h_final,p+Δ fitted on
                     the fitting text, read out like a lens — the linear
                     "Future Lens" ceiling for what h_l,p encodes about p+Δ+1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from jspace.model import HookedModel


@dataclass
class HiddenBank:
    """
    Residual states of a set of equal-length sequences.

    Attributes
    ----------
    ids : torch.Tensor
        (N, T) token ids.
    layers : dict[int, torch.Tensor]
        layer → (N, T, d) float32 CPU block outputs.
    final : torch.Tensor
        (N, T, d) final-block outputs.
    next_top : torch.Tensor
        (N, T, k) the model's top-k rankable next-token ids.
    """

    ids: torch.Tensor
    layers: Dict[int, torch.Tensor]
    final: torch.Tensor
    next_top: torch.Tensor


@dataclass
class Items:
    """
    Lookahead items for one offset.

    Attributes
    ----------
    seq, pos, target : torch.Tensor
        (P,) int64 sequence index, source position, and target token id.
    """

    seq: torch.Tensor
    pos: torch.Tensor
    target: torch.Tensor

    def __len__(self) -> int:
        """Number of items."""
        return int(self.seq.numel())

    def subset(self, index: torch.Tensor) -> "Items":
        """Items at ``index``."""
        return Items(self.seq[index], self.pos[index], self.target[index])


def collect_hidden(
    model: HookedModel,
    sequences: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    rankable: torch.Tensor,
    k: int = 10,
) -> HiddenBank:
    """
    Forward each (1, T) sequence once and keep layer, final, and next-token data.

    Parameters
    ----------
    model : HookedModel
    sequences : Sequence[torch.Tensor]
        Equal-length (1, T) token-id tensors.
    layer_indices : Sequence[int]
    rankable : torch.Tensor
        (V,) bool Readout Filtering mask for the next-token baseline.
    k : int
    """
    lengths = {int(s.shape[1]) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"sequences must share one length, got {sorted(lengths)}")
    final_layer = model.num_layers - 1
    wanted = sorted(set(layer_indices) | {final_layer})
    per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in wanted}
    next_top: List[torch.Tensor] = []
    blocked = (~rankable).to(model.device)
    with torch.no_grad():
        for ids in sequences:
            with model.capture_residuals(wanted) as store:
                logits = model.model(input_ids=ids.to(model.device), use_cache=False).logits[0]
            next_top.append(
                logits.float().masked_fill(blocked, float("-inf")).topk(k, -1).indices.cpu()
            )
            for layer in wanted:
                per_layer[layer].append(store[layer][0].cpu())
            del logits
    return HiddenBank(
        ids=torch.cat(list(sequences)).cpu(),
        layers={layer: torch.stack(per_layer[layer]) for layer in layer_indices},
        final=torch.stack(per_layer[final_layer]),
        next_top=torch.stack(next_top),
    )


def lookahead_items(
    ids: torch.Tensor, offset: int, skip_first: int, rankable: torch.Tensor
) -> Items:
    """
    Items for one offset: kept source positions p with a rankable token at p+Δ+1.

    Parameters
    ----------
    ids : torch.Tensor
        (N, T) token ids.
    offset : int
    skip_first : int
    rankable : torch.Tensor
        (V,) bool.
    """
    n_seq, seq_len = ids.shape
    positions = torch.arange(skip_first, seq_len - 1 - offset)
    if positions.numel() == 0:
        return Items(*(torch.empty(0, dtype=torch.long) for _ in range(3)))
    seq = torch.arange(n_seq).repeat_interleave(positions.numel())
    pos = positions.repeat(n_seq)
    target = ids[seq, pos + offset + 1]
    keep = rankable[target]
    return Items(seq[keep], pos[keep], target[keep])


def unigram_topk(ids: torch.Tensor, rankable: torch.Tensor, k: int) -> torch.Tensor:
    """(k,) the most frequent rankable token ids in ``ids``."""
    counts = torch.bincount(ids.flatten(), minlength=rankable.numel()).float()
    return counts.masked_fill(~rankable, -1.0).topk(k).indices


def ridge_future_maps(
    bank: HiddenBank,
    layer_indices: Sequence[int],
    offsets: Sequence[int],
    skip_first: int,
    ridge: float = 1e-3,
) -> Dict[int, torch.Tensor]:
    """
    Least-squares maps A_{l,Δ} with h_final,p+Δ ≈ A h_l,p, fitted on ``bank``.

    Parameters
    ----------
    bank : HiddenBank
    layer_indices : Sequence[int]
    offsets : Sequence[int]
    skip_first : int
    ridge : float
        Tikhonov strength relative to the mean eigenvalue of XᵀX.

    Returns
    -------
    dict[int, torch.Tensor]
        layer → (K, d, d) float32.
    """
    seq_len = bank.ids.shape[1]
    maps: Dict[int, torch.Tensor] = {}
    for layer in layer_indices:
        stacks = []
        for delta in offsets:
            src = torch.arange(skip_first, seq_len - 1 - delta)
            x = bank.layers[layer][:, src].reshape(-1, bank.final.shape[-1]).double()
            y = bank.final[:, src + delta].reshape(-1, bank.final.shape[-1]).double()
            gram = x.T @ x
            gram += ridge * gram.diagonal().mean() * torch.eye(gram.shape[0], dtype=gram.dtype)
            stacks.append(torch.linalg.solve(gram, x.T @ y).T.float())
        maps[layer] = torch.stack(stacks)
    return maps


def topk_hits(top_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(P, k) candidate ids and (P,) targets → (P,) bool hit indicator."""
    return (top_ids == targets.unsqueeze(1)).any(dim=1)


def readout_recall(
    transports: torch.Tensor,
    targets: torch.Tensor,
    rankable: torch.Tensor,
    norm_weight: torch.Tensor,
    unembedding: torch.Tensor,
    ks: Sequence[int] = (1, 10),
    chunk: int = 128,
) -> Dict[int, float]:
    """
    Recall@k of final-basis transports read out over the rankable vocabulary.

    Parameters
    ----------
    transports : torch.Tensor
        (P, d) vectors in the final-residual basis.
    targets : torch.Tensor
        (P,) int64 rankable target ids.
    rankable : torch.Tensor
        (V,) bool.
    norm_weight, unembedding : torch.Tensor
        Final RMSNorm weight (d,) and LM head (V, d).
    ks : Sequence[int]
    chunk : int
        Items per readout matmul.

    Returns
    -------
    dict[int, float]
    """
    if transports.shape[0] != targets.shape[0]:
        raise ValueError("transports and targets disagree on the number of items")
    if transports.shape[0] == 0:
        return {k: float("nan") for k in ks}
    device = unembedding.device
    blocked = (~rankable).to(device)
    k_max = max(ks)
    tops = []
    for start in range(0, transports.shape[0], chunk):
        logits = (transports[start : start + chunk].to(device).float() * norm_weight.float()) @ (
            unembedding.float().T
        )
        tops.append(logits.masked_fill(blocked, float("-inf")).topk(k_max, dim=1).indices.cpu())
    top = torch.cat(tops)
    return {k: float(topk_hits(top[:, :k], targets.cpu()).float().mean()) for k in ks}
