"""
Batched nonnegative matching pursuit over a fixed atom dictionary.

Counts how many dictionary atoms (corpus J-lens vectors) are needed, with
nonnegative coefficients, to explain a target fraction of a query's squared
norm — the active-concept count of Lindsey et al. (2026).

The pursuit runs in coefficient space: with dots = Q Aᵀ and Gram matrix
G = A Aᵀ, selecting atom j with coefficient a updates

    dots     ← dots − a · G[j]
    ‖r‖²     ← ‖r‖² − a² ‖a_j‖²        (a = dots_j / ‖a_j‖², the least-squares step)

so each step costs O(n · m) instead of O(n · m · D), and the residual vectors
are never materialised.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PursuitResult:
    """
    Outcome of a batched pursuit.

    Attributes
    ----------
    n_atoms : torch.Tensor
        (B, n) int64 — number of distinct atoms selected per query.
    coverage : torch.Tensor
        (B, n) float — explained fraction of squared norm, 1 − ‖r‖² / ‖q‖².
    converged : torch.Tensor
        (B, n) bool — coverage reached the target.  False means the pursuit
        either stalled (no atom had a positive dot with the residual) or hit
        ``max_steps``.
    """

    n_atoms: torch.Tensor
    coverage: torch.Tensor
    converged: torch.Tensor


def nonneg_matching_pursuit(
    queries: torch.Tensor,
    atoms: torch.Tensor,
    coverage: float = 0.95,
    max_steps: int = 30,
    eps: float = 1e-12,
) -> PursuitResult:
    """
    Greedy nonnegative matching pursuit, vectorised over batches and queries.

    Each step selects, for every still-active query, the atom with the largest
    positive dot product with the current residual and subtracts its
    least-squares projection.  A query stops when coverage is reached, when no
    atom has a positive dot product (stall), or after ``max_steps`` steps.

    Parameters
    ----------
    queries : torch.Tensor
        (B, n, D) query vectors.
    atoms : torch.Tensor
        (B, m, D) dictionary; batch ``b`` of queries uses batch ``b`` of atoms.
    coverage : float
        Target explained fraction of squared norm, in (0, 1].
    max_steps : int
        Maximum greedy steps per query.
    eps : float
        Numerical floor for norms and the positivity test.

    Returns
    -------
    PursuitResult
    """
    if queries.ndim != 3 or atoms.ndim != 3:
        raise ValueError(
            f"queries and atoms must be 3-D (B, n, D) / (B, m, D); "
            f"got {tuple(queries.shape)} and {tuple(atoms.shape)}"
        )
    if queries.shape[0] != atoms.shape[0] or queries.shape[2] != atoms.shape[2]:
        raise ValueError(
            f"batch and feature dims must match; got queries {tuple(queries.shape)}, "
            f"atoms {tuple(atoms.shape)}"
        )
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    queries = queries.float()
    atoms = atoms.to(device=queries.device, dtype=torch.float32)
    n_atoms_total = atoms.shape[1]

    atom_norm_sq = atoms.pow(2).sum(-1).clamp_min(eps)  # (B, m)
    gram = torch.bmm(atoms, atoms.transpose(1, 2))  # (B, m, m)
    dots = torch.bmm(queries, atoms.transpose(1, 2))  # (B, n, m)
    query_norm_sq = queries.pow(2).sum(-1)  # (B, n)
    residual_sq = query_norm_sq.clone()
    selected = torch.zeros(dots.shape, dtype=torch.bool, device=dots.device)

    explained = 1.0 - residual_sq / query_norm_sq.clamp_min(eps)
    active = (query_norm_sq > eps) & (explained < coverage)

    for _ in range(max_steps):
        if not bool(active.any()):
            break
        best_val, best_idx = dots.max(dim=-1)  # (B, n)
        step = active & (best_val > eps)
        best_norm_sq = atom_norm_sq.gather(1, best_idx)  # (B, n)
        coef = torch.where(step, best_val / best_norm_sq, torch.zeros_like(best_val))

        gram_rows = torch.gather(
            gram, 1, best_idx.unsqueeze(-1).expand(-1, -1, n_atoms_total)
        )  # (B, n, m)
        dots = dots - coef.unsqueeze(-1) * gram_rows
        residual_sq = residual_sq - coef.pow(2) * best_norm_sq

        idx = best_idx.unsqueeze(-1)
        selected.scatter_(-1, idx, selected.gather(-1, idx) | step.unsqueeze(-1))

        explained = 1.0 - residual_sq.clamp_min(0.0) / query_norm_sq.clamp_min(eps)
        active = step & (explained < coverage)

    explained = torch.where(query_norm_sq > eps, explained, torch.ones_like(explained))
    return PursuitResult(
        n_atoms=selected.sum(-1),
        coverage=explained,
        converged=explained >= coverage,
    )
