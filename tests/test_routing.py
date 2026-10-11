"""Tests for activation routing and Readout Filtering."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerFast

from jspace.readout_filter import is_semantic, semantic_token_mask
from jspace.routing import ActivationRouter, fit_pca, fit_router, kmeans


def _blobs(n_per: int = 60, d: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    centres = torch.eye(d)[:3] * 10.0
    labels = torch.arange(3).repeat_interleave(n_per)
    return centres[labels] + 0.3 * torch.randn(3 * n_per, d, generator=gen), labels


def _purity(assignments: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of points whose cluster's majority label matches their own."""
    joint = torch.zeros(int(assignments.max()) + 1, int(labels.max()) + 1)
    joint.index_put_((assignments, labels), torch.ones(len(labels)), accumulate=True)
    return float(joint.max(dim=1).values.sum() / len(labels))


def test_kmeans_recovers_separated_blobs() -> None:
    points, labels = _blobs()
    centroids, assignments = kmeans(points, 3, seed=1)
    assert centroids.shape == (3, 16)
    assert _purity(assignments, labels) == 1.0


def test_fit_pca_returns_orthonormal_components() -> None:
    points, _ = _blobs()
    mean, components = fit_pca(points, 4)
    assert mean.shape == (16,) and components.shape == (16, 4)
    torch.testing.assert_close(components.T @ components, torch.eye(4), atol=1e-5, rtol=0)


def test_router_assigns_by_direction_and_round_trips() -> None:
    points, labels = _blobs()
    router = fit_router(points, num_clusters=3, projection_dim=4, seed=0)
    assignments = router.assign(points)
    assert _purity(assignments, labels) == 1.0
    torch.testing.assert_close(router.assign(points * 5.0), assignments)
    assert router.assign(points.view(3, 60, 16)).shape == (3, 60)
    restored = ActivationRouter.from_state(router.to_state(), "cpu")
    torch.testing.assert_close(restored.assign(points), assignments)


def test_semantic_mask(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    assert (
        is_semantic(" the") and is_semantic("3") and not is_semantic(" ,.") and not is_semantic("")
    )
    vocab = len(tiny_tokenizer) + 5
    mask = semantic_token_mask(tiny_tokenizer, vocab)
    assert mask.shape == (vocab,)
    assert not mask[len(tiny_tokenizer) :].any()
    assert not mask[tiny_tokenizer.pad_token_id] and not mask[tiny_tokenizer.unk_token_id]
    assert mask[tiny_tokenizer.convert_tokens_to_ids("cat")]
    forced = semantic_token_mask(tiny_tokenizer, vocab, keep_ids=[tiny_tokenizer.pad_token_id])
    assert forced[tiny_tokenizer.pad_token_id]
