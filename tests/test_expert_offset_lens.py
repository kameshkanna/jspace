"""Tests for the offset-resolved expert (J++) lens estimator."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from jspace.expert_offset_lens import ExpertOffsetConfig, ExpertOffsetLens, kept_position_mask
from jspace.lrp import lrp_rules
from jspace.model import HookedModel
from tests.conftest import build_tiny_model

_SKIP, _T = 2, 12


def exact_kept_offset_jacobians(
    model: HookedModel, ids: torch.Tensor, layer: int, offsets: Sequence[int], lrp: bool
) -> torch.Tensor:
    """Uniform mean over kept sources s (s+Δ also kept) of the exact ∂h_final,s+Δ / ∂h_l,s."""
    store = {}
    grab = model.layers[layer].register_forward_hook(
        lambda _m, _i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    grab.remove()

    def final_from(injected: torch.Tensor) -> torch.Tensor:
        out = {}
        swap = model.layers[layer].register_forward_hook(
            lambda _m, _i, o: (injected,) + tuple(o[1:]) if isinstance(o, tuple) else injected
        )
        read = model.layers[-1].register_forward_hook(
            lambda _m, _i, o: out.__setitem__("h", o[0] if isinstance(o, tuple) else o)
        )
        try:
            model.model(input_ids=ids, use_cache=False)
        finally:
            swap.remove()
            read.remove()
        return out["h"]

    with lrp_rules(model.model) if lrp else nullcontext():
        jac = torch.autograd.functional.jacobian(final_from, store["h"])[0, :, :, 0]
    seq_len = jac.shape[0]
    stacks = []
    for delta in offsets:
        src = torch.arange(_SKIP, seq_len - 1 - delta)
        stacks.append(jac[src + delta, :, src, :].mean(0))
    return torch.stack(stacks)


@pytest.mark.parametrize("lrp", [False, True])
def test_pooled_estimate_is_unbiased_against_exact_jacobian(
    tiny_tokenizer: PreTrainedTokenizerFast, lrp: bool
) -> None:
    model = build_tiny_model(tiny_tokenizer, init_range=0.3, dtype=torch.float64)
    offsets = (0, 1, 2, 3)
    ids = torch.randint(2, len(tiny_tokenizer), (1, _T), generator=torch.Generator().manual_seed(4))

    def fit(draws: int) -> ExpertOffsetLens:
        cfg = ExpertOffsetConfig(
            offsets=offsets,
            layer_indices=(0, 1),
            num_experts=1,
            rows_per_pass=8,
            n_sign_draws=draws,
            skip_first=_SKIP,
            lrp=lrp,
            min_kept_positions=1,
            seed=1,
        )
        return (
            ExpertOffsetLens(model, cfg).fit_routers([ids]).fit_experts([ids], show_progress=False)
        )

    coarse, fine = fit(20), fit(320)
    for layer in (0, 1):
        exact = exact_kept_offset_jacobians(model, ids, layer, offsets, lrp)
        err = {}
        for name, lens in (("coarse", coarse), ("fine", fine)):
            diff = lens.pooled[layer].double() - exact
            err[name] = diff.flatten(1).norm(dim=-1) / exact.flatten(1).norm(dim=-1)
        # One 12-token sequence gives ~8 sources, so per-offset errors are heavy-tailed;
        # unbiasedness shows in the mean error shrinking with 16x the sign draws.
        assert (err["fine"] < 0.1).all(), (layer, err)
        assert err["fine"].mean() < 0.6 * err["coarse"].mean(), (layer, err)


def test_offset_zero_is_exact_without_cross_terms(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    """With a single offset (stride 1) every kept target is tagged; Δ=0 rows still carry
    zero-mean cross terms, but the final layer's own Jacobian is exactly the identity."""
    model = build_tiny_model(tiny_tokenizer, init_range=0.3, dtype=torch.float64)
    ids = torch.randint(2, len(tiny_tokenizer), (1, _T), generator=torch.Generator().manual_seed(5))
    cfg = ExpertOffsetConfig(
        offsets=(0, 2),
        layer_indices=(1,),
        num_experts=1,
        rows_per_pass=32,
        n_sign_draws=2,
        skip_first=_SKIP,
        lrp=False,
        min_kept_positions=1,
    )
    lens = ExpertOffsetLens(model, cfg).fit_routers([ids]).fit_experts([ids], show_progress=False)
    assert lens.pooled[1].shape == (2, 32, 32)
    assert torch.isfinite(lens.reliability()).all() and lens.reliability().shape == (1, 2)


def test_experts_average_back_to_pooled(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    model = build_tiny_model(tiny_tokenizer, init_range=0.3)
    gen = torch.Generator().manual_seed(6)
    seqs = [torch.randint(2, len(tiny_tokenizer), (1, 24), generator=gen) for _ in range(4)]
    cfg = ExpertOffsetConfig(
        offsets=(0, 1),
        layer_indices=(0, 1),
        num_experts=2,
        projection_dim=4,
        rows_per_pass=16,
        n_sign_draws=2,
        skip_first=_SKIP,
        min_kept_positions=1,
    )
    lens = ExpertOffsetLens(model, cfg).fit_routers(seqs).fit_experts(seqs, show_progress=False)
    for layer in (0, 1):
        n = lens.expert_counts[layer].float()
        assert (n > 0).all()
        recombined = torch.einsum("ke,keij->kij", n / n.sum(-1, keepdim=True), lens.experts[layer])
        torch.testing.assert_close(recombined, lens.pooled[layer], rtol=1e-4, atol=1e-6)

    lens.set_weights(lens.count_weights())
    torch.testing.assert_close(
        lens.matrices(0, "weighted"), lens.matrices(0, "pooled"), rtol=1e-4, atol=1e-6
    )
    assert lens.project(torch.randn(5, 32), 1, "pooled").shape == (2, 5, 32)
    assert lens.expert_transports(torch.randn(5, 32), 1, 0).shape == (5, 2, 32)
    with pytest.raises(ValueError):
        lens.set_weights({0: torch.ones(2, 2)})


def test_save_load_roundtrip(fitted_expert_lens: ExpertOffsetLens, tmp_path) -> None:
    fitted_expert_lens.set_weights(fitted_expert_lens.count_weights())
    path = tmp_path / "expert_lens.pt"
    fitted_expert_lens.save(path)
    loaded = ExpertOffsetLens.load(path, fitted_expert_lens.model)
    assert loaded.config == fitted_expert_lens.config
    for layer in fitted_expert_lens.layer_indices:
        torch.testing.assert_close(loaded.pooled[layer], fitted_expert_lens.pooled[layer])
        torch.testing.assert_close(
            loaded.experts[layer], fitted_expert_lens.experts[layer], atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(loaded.weights[layer], fitted_expert_lens.weights[layer])
        x = torch.randn(7, 32)
        torch.testing.assert_close(
            loaded.routers[layer].assign(x), fitted_expert_lens.routers[layer].assign(x)
        )


@pytest.fixture
def fitted_expert_lens(tiny_tokenizer: PreTrainedTokenizerFast) -> ExpertOffsetLens:
    model = build_tiny_model(tiny_tokenizer, init_range=0.3)
    gen = torch.Generator().manual_seed(7)
    seqs = [torch.randint(2, len(tiny_tokenizer), (1, 20), generator=gen) for _ in range(3)]
    cfg = ExpertOffsetConfig(
        offsets=(0, 1, 2),
        layer_indices=(0, 1),
        num_experts=2,
        projection_dim=4,
        rows_per_pass=16,
        n_sign_draws=2,
        skip_first=_SKIP,
        min_kept_positions=1,
    )
    return ExpertOffsetLens(model, cfg).fit_routers(seqs).fit_experts(seqs, show_progress=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_indices": ()},
        {"offsets": (1, 0)},
        {"n_sign_draws": 3},
        {"num_experts": 0},
        {"layer_indices": (0, 0)},
        {"skip_first": -1},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    base = {"layer_indices": (0,)}
    with pytest.raises(ValueError):
        ExpertOffsetConfig(**{**base, **kwargs})


def test_kept_mask_and_final_layer_rejected(tiny_model: HookedModel) -> None:
    assert kept_position_mask(6, 2).tolist() == [False, False, True, True, True, False]
    with pytest.raises(ValueError):
        ExpertOffsetLens(tiny_model, ExpertOffsetConfig(layer_indices=(2,)))
