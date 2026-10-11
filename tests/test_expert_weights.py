"""Tests for the expert-weight search and the lookahead evaluation helpers."""

from __future__ import annotations

import pytest
import torch

from jspace.expert_weights import (
    WeightSearchConfig,
    expert_logit_cache,
    fit_expert_weights,
    linear_readout_logits,
    recall_at_k,
    target_margins,
)
from jspace.lookahead import (
    HiddenBank,
    collect_hidden,
    lookahead_items,
    ridge_future_maps,
    topk_hits,
    unigram_topk,
)
from jspace.model import HookedModel

_FAST = WeightSearchConfig(k=3, steps=60, exact_every=5)


def test_linear_readout_ranks_like_the_model_unembed(tiny_model: HookedModel) -> None:
    x = torch.randn(6, tiny_model.model.config.hidden_size)
    norm = tiny_model.model.model.norm
    linear = linear_readout_logits(x, norm.weight, tiny_model.model.lm_head.weight)
    stock = tiny_model.unembed(x).float()
    assert torch.equal(linear.topk(5).indices, stock.topk(5).indices)
    cache = expert_logit_cache(
        x.view(3, 2, -1), norm.weight, tiny_model.model.lm_head.weight, chunk=2
    )
    torch.testing.assert_close(cache.view(6, -1), linear)


def test_margins_match_brute_force_ranks() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(30, 20, generator=gen)
    targets = torch.randint(0, 20, (30,), generator=gen)
    rankable = torch.ones(20, dtype=torch.bool)
    rankable[torch.tensor([1, 5, 7])] = False
    rankable[targets] = True
    margin, spread = target_margins(logits, targets, rankable, k=4)
    masked = logits.masked_fill(~rankable, float("-inf"))
    rank = 1 + (masked > logits.gather(1, targets[:, None])).sum(1)
    assert torch.equal(margin >= 0, rank <= 4)
    assert (spread > 0).all()
    assert recall_at_k(logits, targets, rankable, 4) == pytest.approx(
        float((rank <= 4).float().mean())
    )
    blocked = rankable.clone()
    blocked[targets[0]] = False
    with pytest.raises(ValueError):
        target_margins(logits, targets, blocked, k=4)


def _cache_with_signal(sign: float) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    targets = torch.randint(0, 50, (40,), generator=gen)
    cache = torch.randn(40, 3, 50, generator=gen)
    cache[:, [0, 2]] *= 4.0
    cache[torch.arange(40), 1, targets] += sign * 6.0
    return cache, targets


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_weight_search_finds_the_informative_expert(sign: float) -> None:
    cache, targets = _cache_with_signal(sign)
    assert recall_at_k(cache.mean(1), targets, None, 3) < 0.9
    fitted = fit_expert_weights(cache, targets, None, _FAST)
    assert fitted.train_recall == 1.0
    assert fitted.weights.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.sign(fitted.weights[1]) == sign
    assert fitted.weights[1].abs() == fitted.weights.abs().max()


def test_lookahead_items_unigram_and_hits() -> None:
    ids = torch.tensor([[9, 9, 2, 3, 4, 5, 6, 7], [9, 9, 4, 4, 4, 1, 1, 0]])
    rankable = torch.ones(10, dtype=torch.bool)
    rankable[0] = False
    items = lookahead_items(ids, offset=1, skip_first=2, rankable=rankable)
    pairs = set(zip(items.seq.tolist(), items.pos.tolist(), items.target.tolist()))
    assert pairs == {(0, 2, 4), (0, 3, 5), (0, 4, 6), (0, 5, 7), (1, 2, 4), (1, 3, 1), (1, 4, 1)}
    assert len(items.subset(torch.tensor([0, 1]))) == 2
    assert set(unigram_topk(ids, rankable, 2).tolist()) == {9, 4}
    assert topk_hits(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 5])).tolist() == [True, False]


def test_ridge_recovers_a_planted_linear_map() -> None:
    gen = torch.Generator().manual_seed(2)
    d, n, seq_len = 6, 40, 10
    a_true = torch.randn(d, d, generator=gen)
    layer = torch.randn(n, seq_len, d, generator=gen)
    final = torch.zeros(n, seq_len, d)
    final[:, 1:] = layer[:, :-1] @ a_true.T
    bank = HiddenBank(
        torch.zeros(n, seq_len, dtype=torch.long),
        {0: layer},
        final,
        torch.zeros(n, seq_len, 1, dtype=torch.long),
    )
    maps = ridge_future_maps(bank, [0], offsets=[1], skip_first=1, ridge=1e-8)
    torch.testing.assert_close(maps[0][0], a_true, atol=1e-3, rtol=1e-3)


def test_collect_hidden_shapes(tiny_model: HookedModel) -> None:
    seqs = [torch.randint(2, 40, (1, 9)) for _ in range(3)]
    rankable = torch.ones(tiny_model.model.config.vocab_size, dtype=torch.bool)
    bank = collect_hidden(tiny_model, seqs, [0, 1], rankable, k=4)
    assert bank.ids.shape == (3, 9) and bank.layers[1].shape == (3, 9, 32)
    assert bank.final.shape == (3, 9, 32) and bank.next_top.shape == (3, 9, 4)
    with pytest.raises(ValueError):
        collect_hidden(
            tiny_model,
            [torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long)],
            [0],
            rankable,
        )
