"""Tests for multi-token workspace analysis."""

from __future__ import annotations

import json
import math

import pytest
import torch

from jspace.model import HookedModel
from jspace.multitoken import MultiTokenWorkspace, lookahead_accuracy
from jspace.offset_lens import OffsetJacobianLens, OffsetLensConfig
from tests.conftest import PROMPTS


def test_lookahead_accuracy_scores_only_in_sequence_targets() -> None:
    token_ids = torch.tensor([5, 6, 7, 8, 9])
    preds = torch.tensor(
        [
            [6, 7, 8, 9, 0],
            [7, 8, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    acc = lookahead_accuracy(preds, token_ids, [0, 1, 4])
    assert acc[0].item() == pytest.approx(1.0)
    assert acc[1].item() == pytest.approx(2 / 3)
    assert math.isnan(acc[2].item())


def test_lookahead_accuracy_validates_shapes() -> None:
    with pytest.raises(ValueError):
        lookahead_accuracy(
            torch.zeros(2, 5, dtype=torch.long), torch.zeros(5, dtype=torch.long), [0]
        )
    with pytest.raises(ValueError):
        lookahead_accuracy(
            torch.zeros(1, 5, dtype=torch.long), torch.zeros(4, dtype=torch.long), [0]
        )


def test_analyse_produces_consistent_report(fitted_lens: OffsetJacobianLens) -> None:
    analyser = MultiTokenWorkspace(fitted_lens, top_k=3)
    report = analyser.analyse("the cat sat on the mat")
    seq_len = len(report.token_ids)
    assert seq_len == 6 and len(report.tokens) == seq_len
    assert [s.layer_idx for s in report.layer_stats] == fitted_lens.layer_indices

    for stats in report.layer_stats:
        assert stats.top_ids.shape == (3, seq_len, 3)
        assert (stats.top_probs[..., :-1] >= stats.top_probs[..., 1:]).all()
        assert stats.lookahead_acc.shape == (3,) and stats.entropy.shape == (3,)
        assert 0 <= stats.n_joint <= analyser.max_atoms
        assert 0 <= stats.n_split <= analyser.max_atoms * 3
        assert 0.0 <= stats.joint_converged <= 1.0 and 0.0 <= stats.split_converged <= 1.0

    assert report.baseline_acc.shape == (3,)
    latent = analyser.latent_string(report, 1)
    assert [offset for offset, _, _ in latent] == [0, 1, 2]
    json.dumps(analyser.to_summary(report))
    analyser.print_report(report)


def test_joint_and_split_counts_agree_for_a_single_offset(tiny_model: HookedModel) -> None:
    config = OffsetLensConfig(offsets=(0,), n_proj=4, max_length=32)
    lens = OffsetJacobianLens(tiny_model, config).fit(PROMPTS, show_progress=False)
    report = MultiTokenWorkspace(lens, top_k=2).analyse(PROMPTS[0])
    for stats in report.layer_stats:
        assert stats.n_joint == pytest.approx(stats.n_split)


def test_concept_counting_requires_corpus_atoms(fitted_lens: OffsetJacobianLens) -> None:
    fitted_lens.corpus_atoms = {}
    with pytest.raises(ValueError):
        MultiTokenWorkspace(fitted_lens)
    report = MultiTokenWorkspace(fitted_lens, count_concepts=False).analyse(PROMPTS[1])
    assert all(s.n_joint is None for s in report.layer_stats)
