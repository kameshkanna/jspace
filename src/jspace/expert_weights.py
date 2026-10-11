"""
Learned expert weights for Jacobian Filtering (J++), per (layer, offset).

Adapted from the J++ rank objective (Ayonrinde & Lindsey, 2026; Apache-2.0, see
THIRD_PARTY_NOTICES.md) and extended to arbitrary single-token targets, so the
same search fits weights for future-token readouts at every offset Δ.

Linearised readout.  With an RMSNorm final norm of weight g,
unembed(x) = W_U (g ⊙ x) / rms(x).  Ranks ignore the positive factor 1/rms(x),
so L(x) = W_U (g ⊙ x) ranks tokens exactly as the model's readout does and is
linear in x.  For per-expert transports T_pe the combined lens therefore has
logits Σ_e w_e L(T_pe): one cached (P, E, V) tensor makes the objective linear
in the weights, with no model pass per candidate weighting.

Search.  Recall@k is piecewise constant in w, so Adam runs on the surrogate
sigmoid(margin / (temperature · spread)) — margin = target logit − k-th largest
rankable logit, spread = per-item logit std — from several starts, evaluating
the exact recall every few steps and keeping the best exact iterate.  Weights
are unit-L2 and may be negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def linear_readout_logits(
    x: torch.Tensor, norm_weight: torch.Tensor, unembedding: torch.Tensor
) -> torch.Tensor:
    """
    Rank-equivalent readout logits W_U (g ⊙ x), in float32.

    Parameters
    ----------
    x : torch.Tensor
        (..., d) final-basis vectors.
    norm_weight : torch.Tensor
        (d,) final RMSNorm weight g.
    unembedding : torch.Tensor
        (V, d) LM-head weight W_U.

    Returns
    -------
    torch.Tensor
        (..., V) float32.
    """
    return (x.float() * norm_weight.float()) @ unembedding.float().T


def expert_logit_cache(
    transports: torch.Tensor,
    norm_weight: torch.Tensor,
    unembedding: torch.Tensor,
    chunk: int = 16,
) -> torch.Tensor:
    """
    (P, E, d) per-expert transports → (P, E, V) float32 linearised logits.

    Parameters
    ----------
    transports : torch.Tensor
    norm_weight : torch.Tensor
    unembedding : torch.Tensor
    chunk : int
        Items per matmul, bounding the temporary memory.
    """
    if transports.ndim != 3:
        raise ValueError(f"transports must be (P, E, d), got {tuple(transports.shape)}")
    return torch.cat(
        [
            linear_readout_logits(transports[i : i + chunk], norm_weight, unembedding)
            for i in range(0, transports.shape[0], chunk)
        ]
    )


def target_margins(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rankable: Optional[torch.Tensor],
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per item: (target logit − k-th largest rankable logit, logit std).

    An item is a hit at k (rank = 1 + #rankable logits strictly above the
    target ≤ k) exactly when its margin is ≥ 0.

    Parameters
    ----------
    logits : torch.Tensor
        (P, V).
    targets : torch.Tensor
        (P,) int64 target ids; each must be rankable.
    rankable : torch.Tensor | None
        (V,) bool; None ranks the full vocabulary.
    k : int
    """
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} mismatch"
        )
    spread = logits.std(dim=1).detach().clamp_min(1e-6)
    target_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    if rankable is not None:
        rankable = rankable.to(logits.device)
        if not bool(rankable[targets].all()):
            raise ValueError("every target must be rankable")
        logits = logits.masked_fill(~rankable.unsqueeze(0), float("-inf"))
    kth = logits.topk(k, dim=1).values[:, -1]
    return target_logit - kth, spread


def recall_at_k(
    logits: torch.Tensor, targets: torch.Tensor, rankable: Optional[torch.Tensor], k: int
) -> float:
    """Exact fraction of items whose target ranks in the top k."""
    margin, _ = target_margins(logits, targets, rankable, k)
    return float((margin >= 0).float().mean())


@dataclass(frozen=True)
class WeightSearchConfig:
    """
    Hyper-parameters of the expert-weight search (J++ defaults).

    Attributes
    ----------
    k : int
    steps : int
        Adam steps per start.
    learning_rate : float
    temperature : float
        Surrogate temperature, in units of the per-item logit spread.
    exact_every : int
        Evaluate the exact recall every this many steps.
    top_n : int
        Experts in the ``mean_top`` start.
    """

    k: int = 10
    steps: int = 200
    learning_rate: float = 0.05
    temperature: float = 0.1
    exact_every: int = 5
    top_n: int = 3


@dataclass(frozen=True)
class FittedWeights:
    """
    Result of :func:`fit_expert_weights`.

    Attributes
    ----------
    weights : torch.Tensor
        (E,) unit-L2 float32 on CPU.
    train_recall : float
        Exact recall@k on the fitting items.
    start : str
        Name of the winning start.
    """

    weights: torch.Tensor
    train_recall: float
    start: str


def fit_expert_weights(
    cache: torch.Tensor,
    targets: torch.Tensor,
    rankable: Optional[torch.Tensor],
    config: WeightSearchConfig = WeightSearchConfig(),
    extra_starts: Optional[Dict[str, torch.Tensor]] = None,
) -> FittedWeights:
    """
    Multi-start Adam on the recall@k surrogate; returns the best exact iterate.

    Parameters
    ----------
    cache : torch.Tensor
        (P, E, V) per-expert linearised logits.
    targets : torch.Tensor
        (P,) int64.
    rankable : torch.Tensor | None
        (V,) bool Readout Filtering mask.
    config : WeightSearchConfig
    extra_starts : dict[str, torch.Tensor] | None
        Additional named (E,) starting weightings (e.g. count shares).

    Returns
    -------
    FittedWeights
    """
    n_items, n_experts, _ = cache.shape
    if targets.shape != (n_items,):
        raise ValueError(f"targets shape {tuple(targets.shape)} != ({n_items},)")
    targets = targets.to(cache.device)

    single = torch.tensor(
        [recall_at_k(cache[:, e], targets, rankable, config.k) for e in range(n_experts)]
    )
    order = sorted(range(n_experts), key=lambda e: (-float(single[e]), e))
    starts: Dict[str, torch.Tensor] = {
        "uniform": torch.ones(n_experts),
        "select_best": torch.zeros(n_experts).index_fill_(0, torch.tensor(order[:1]), 1.0),
        "mean_top": torch.zeros(n_experts).index_fill_(0, torch.tensor(order[: config.top_n]), 1.0),
        **(extra_starts or {}),
    }

    best: Optional[Tuple[float, float, FittedWeights]] = None
    for name, init in starts.items():
        weights = (init.float() / init.float().norm()).to(cache.device).clone().requires_grad_(True)
        optimiser = torch.optim.Adam([weights], lr=config.learning_rate)
        for step in range(config.steps + 1):
            unit = weights / weights.norm()
            margin, spread = target_margins(
                torch.einsum("pev,e->pv", cache, unit), targets, rankable, config.k
            )
            surrogate = torch.sigmoid(margin / (config.temperature * spread)).mean()
            if step % config.exact_every == 0 or step == config.steps:
                exact = float((margin.detach() >= 0).float().mean())
                candidate = (
                    exact,
                    float(surrogate.detach()),
                    FittedWeights(unit.detach().cpu().clone(), exact, name),
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if step == config.steps:
                break
            optimiser.zero_grad()
            (-surrogate).backward()
            optimiser.step()
    assert best is not None
    return best[2]
