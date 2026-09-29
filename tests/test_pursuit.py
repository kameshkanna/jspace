"""Tests for batched nonnegative matching pursuit."""

from __future__ import annotations

import pytest
import torch

from jspace.pursuit import nonneg_matching_pursuit


def _reference_pursuit(
    queries: torch.Tensor, atoms: torch.Tensor, coverage: float, max_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual-space pursuit for one dictionary; oracle for the coefficient-space version."""
    residual = queries.clone().double()
    atoms = atoms.double()
    q_sq = residual.pow(2).sum(-1)
    selected = torch.zeros(queries.shape[0], atoms.shape[0], dtype=torch.bool)
    active = torch.ones(queries.shape[0], dtype=torch.bool)
    for _ in range(max_steps):
        dots = residual @ atoms.T
        best_val, best_idx = dots.max(-1)
        step = active & (best_val > 1e-12)
        coef = torch.where(
            step, best_val / atoms[best_idx].pow(2).sum(-1), torch.zeros_like(best_val)
        )
        residual = residual - coef.unsqueeze(-1) * atoms[best_idx]
        selected[torch.arange(len(best_idx)), best_idx] |= step
        explained = 1 - residual.pow(2).sum(-1) / q_sq
        active = step & (explained < coverage)
    return selected.sum(-1), 1 - residual.pow(2).sum(-1) / q_sq


def test_recovers_sparse_positive_combination() -> None:
    atoms = torch.eye(8)[:6].unsqueeze(0)
    query = (0.6 * torch.eye(8)[0] + 0.5 * torch.eye(8)[2] + 0.4 * torch.eye(8)[4]).view(1, 1, 8)
    result = nonneg_matching_pursuit(query, atoms, coverage=0.999)
    assert result.n_atoms.item() == 3
    assert result.converged.item()
    assert result.coverage.item() == pytest.approx(1.0, abs=1e-6)


def test_stops_as_soon_as_coverage_is_reached() -> None:
    atoms = torch.eye(4).unsqueeze(0)
    query = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
    result = nonneg_matching_pursuit(query, atoms, coverage=0.95)
    assert result.n_atoms.item() == 1
    assert result.converged.item()


def test_stalls_when_no_atom_has_positive_dot() -> None:
    atoms = torch.eye(4).unsqueeze(0)
    query = torch.tensor([[[-1.0, -1.0, 0.0, 0.0]]])
    result = nonneg_matching_pursuit(query, atoms)
    assert result.n_atoms.item() == 0
    assert not result.converged.item()
    assert result.coverage.item() == pytest.approx(0.0)


def test_zero_query_needs_no_atoms() -> None:
    result = nonneg_matching_pursuit(torch.zeros(1, 1, 4), torch.eye(4).unsqueeze(0))
    assert result.n_atoms.item() == 0
    assert result.converged.item()


def test_max_steps_caps_selection() -> None:
    atoms = torch.eye(4).unsqueeze(0)
    query = torch.full((1, 1, 4), 0.5)
    result = nonneg_matching_pursuit(query, atoms, coverage=0.99, max_steps=2)
    assert result.n_atoms.item() == 2
    assert not result.converged.item()


def test_batches_use_their_own_dictionaries() -> None:
    atoms = torch.stack([torch.eye(4), -torch.eye(4)])
    query = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 1, 4)
    result = nonneg_matching_pursuit(query, atoms)
    assert result.converged.tolist() == [[True], [False]]


def test_matches_residual_space_reference_on_random_dictionary() -> None:
    gen = torch.Generator().manual_seed(0)
    atoms = torch.nn.functional.normalize(torch.randn(40, 16, generator=gen), dim=-1)
    queries = torch.randn(25, 16, generator=gen)
    result = nonneg_matching_pursuit(
        queries.unsqueeze(0), atoms.unsqueeze(0), coverage=0.9, max_steps=12
    )
    ref_n, ref_cov = _reference_pursuit(queries, atoms, coverage=0.9, max_steps=12)
    assert torch.equal(result.n_atoms[0], ref_n)
    torch.testing.assert_close(result.coverage[0].double(), ref_cov, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "queries, atoms, kwargs",
    [
        (torch.zeros(1, 4), torch.zeros(1, 2, 4), {}),
        (torch.zeros(2, 1, 4), torch.zeros(1, 2, 4), {}),
        (torch.zeros(1, 1, 4), torch.zeros(1, 2, 3), {}),
        (torch.zeros(1, 1, 4), torch.zeros(1, 2, 4), {"coverage": 0.0}),
        (torch.zeros(1, 1, 4), torch.zeros(1, 2, 4), {"max_steps": 0}),
    ],
)
def test_rejects_invalid_inputs(queries: torch.Tensor, atoms: torch.Tensor, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        nonneg_matching_pursuit(queries, atoms, **kwargs)
