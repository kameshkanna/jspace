"""Tests for the offset-resolved J-lens: estimator exactness, validation, persistence."""

from __future__ import annotations

from typing import Sequence

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from jspace.model import HookedModel
from jspace.offset_lens import OffsetJacobianLens, OffsetLensConfig, pack_prompts
from tests.conftest import PROMPTS as _PROMPTS, build_tiny_model


def exact_offset_jacobians(
    model: HookedModel, input_ids: torch.Tensor, layer_idx: int, offsets: Sequence[int]
) -> torch.Tensor:
    """Exact E_t[∂h_final_{t+Δ} / ∂h_l_t] via full autograd Jacobians; (K, d, d)."""
    captured = {}
    grab = model.layers[layer_idx].register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(
            "h", (out[0] if isinstance(out, tuple) else out).detach()
        )
    )
    with torch.no_grad():
        model.model(input_ids=input_ids, use_cache=False)
    grab.remove()

    def final_from(injected: torch.Tensor) -> torch.Tensor:
        final = {}
        replace = model.layers[layer_idx].register_forward_hook(
            lambda _m, _i, out: (injected,) + tuple(out[1:]) if isinstance(out, tuple) else injected
        )
        read = model.layers[-1].register_forward_hook(
            lambda _m, _i, out: final.__setitem__("h", out[0] if isinstance(out, tuple) else out)
        )
        try:
            model.model(input_ids=input_ids, use_cache=False)
        finally:
            replace.remove()
            read.remove()
        return final["h"]

    jac = torch.autograd.functional.jacobian(final_from, captured["h"])[0, :, :, 0]
    seq_len = jac.shape[0]
    stacks = []
    for delta in offsets:
        src = torch.arange(seq_len - delta)
        stacks.append(jac[src + delta, :, src, :].mean(0))
    return torch.stack(stacks)


def _relative_errors(lens: OffsetJacobianLens, exact: torch.Tensor, layer: int) -> torch.Tensor:
    """Per-offset relative Frobenius error of a fitted stack against the exact one."""
    diff = lens.jacobians[layer].double() - exact
    return diff.flatten(1).norm(dim=-1) / exact.flatten(1).norm(dim=-1)


def test_comb_estimator_is_unbiased_and_matches_exact_jacobian(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = build_tiny_model(tiny_tokenizer, init_range=0.3, dtype=torch.float64)
    offsets = (0, 1, 2, 3)
    input_ids = torch.randint(
        2, len(tiny_tokenizer), (1, 12), generator=torch.Generator().manual_seed(1)
    )

    def fit(n_proj: int) -> OffsetJacobianLens:
        config = OffsetLensConfig(
            offsets=offsets, layer_indices=(0, 1), n_proj=n_proj, max_length=64
        )
        return OffsetJacobianLens(model, config).fit_jacobians(
            [{"input_ids": input_ids}], show_progress=False
        )

    coarse, fine = fit(500), fit(2000)
    for row, layer in enumerate((0, 1)):
        exact = exact_offset_jacobians(model, input_ids, layer, offsets)
        err_coarse = _relative_errors(coarse, exact, layer)
        err_fine = _relative_errors(fine, exact, layer)
        assert (err_fine < 0.3).all(), f"layer {layer}: relative errors {err_fine.tolist()}"
        # 4x the projections must roughly halve the error; a biased estimator would plateau.
        assert (err_fine / err_coarse < 0.65).all(), (
            f"layer {layer}: error did not shrink as 1/sqrt(n): "
            f"{err_coarse.tolist()} -> {err_fine.tolist()}"
        )

        exact_norm = exact.flatten(1).norm(dim=-1).float()
        reach = fine.temporal_reach()[row]
        naive = fine.jacobians[layer].flatten(1).norm(dim=-1).cpu()
        assert ((reach - exact_norm).abs() / exact_norm < 0.08).all(), (reach, exact_norm)
        # On noise-dominated offsets the naive norm is inflated by estimator variance.
        naive_err = (naive[1:] - exact_norm[1:]).abs().mean()
        assert (reach[1:] - exact_norm[1:]).abs().mean() < naive_err, (reach, naive, exact_norm)
        assert (fine.reliability()[row] > coarse.reliability()[row]).all()
        assert (fine.reliability()[row] > 0.8).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offsets": ()},
        {"offsets": (-1, 0)},
        {"offsets": (2, 1)},
        {"offsets": (0, 0)},
        {"n_proj": 0},
        {"n_proj": 3},
        {"offsets": (0, 4), "stride": 4},
        {"offsets": (0, 200), "max_length": 128},
        {"layer_indices": (1, 1)},
        {"seed": -1},
    ],
)
def test_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        OffsetLensConfig(**kwargs)


def test_config_default_stride_exceeds_max_offset() -> None:
    assert OffsetLensConfig(offsets=[0, 2, 5]).comb_stride == 6
    assert OffsetLensConfig(offsets=(0, 2, 5), stride=9).comb_stride == 9


def test_rejects_out_of_range_layers(tiny_model: HookedModel) -> None:
    with pytest.raises(ValueError):
        OffsetJacobianLens(tiny_model, OffsetLensConfig(offsets=(0, 1), layer_indices=(0, 7)))


def test_rejects_padded_or_batched_sequences(tiny_model: HookedModel) -> None:
    lens = OffsetJacobianLens(tiny_model, OffsetLensConfig(offsets=(0, 1), n_proj=2, max_length=16))
    padded = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    with pytest.raises(ValueError):
        lens.fit_jacobians([padded], show_progress=False)
    with pytest.raises(ValueError):
        lens.fit_jacobians([{"input_ids": torch.ones(2, 5, dtype=torch.long)}], show_progress=False)


def test_fails_when_sequences_are_shorter_than_offsets(tiny_model: HookedModel) -> None:
    lens = OffsetJacobianLens(tiny_model, OffsetLensConfig(offsets=(0, 6), n_proj=2, max_length=16))
    with pytest.raises(RuntimeError):
        lens.fit_jacobians([{"input_ids": torch.ones(1, 4, dtype=torch.long)}], show_progress=False)


def test_fit_populates_jacobians_and_unit_norm_atoms(fitted_lens: OffsetJacobianLens) -> None:
    d = fitted_lens.d_model
    assert sorted(fitted_lens.jacobians) == [0, 1, 2]
    for layer in fitted_lens.layer_indices:
        assert fitted_lens.jacobians[layer].shape == (3, d, d)
        atoms = fitted_lens.corpus_atoms[layer]
        assert atoms.shape == (len(_PROMPTS), 3, d) and atoms.dtype == torch.float16
        assert fitted_lens.corpus_mean[layer].shape == (3, d)
    for layer in fitted_lens.layer_indices[:-1]:
        norms = fitted_lens.corpus_atoms[layer].float().norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(len(_PROMPTS), 3), atol=2e-3, rtol=0)
    assert (fitted_lens.sample_counts > 0).all()
    assert fitted_lens.n_prompts == len(_PROMPTS) and fitted_lens.n_sequences >= 1


def test_final_layer_is_exact_under_comb_tagging(fitted_lens: OffsetJacobianLens) -> None:
    """At the final layer J(t, t') = I·[t == t']: every sample has a unit diagonal, and the
    comb never tags a source and its own position for Δ > 0, so those stacks are exactly zero."""
    final = fitted_lens.jacobians[fitted_lens.final_layer]
    torch.testing.assert_close(final[0].diagonal(), torch.ones(fitted_lens.d_model))
    assert torch.count_nonzero(final[1:]) == 0


def test_readout_and_projection_shapes(fitted_lens: OffsetJacobianLens) -> None:
    d, vocab = fitted_lens.d_model, fitted_lens.model.model.config.vocab_size
    hidden = torch.randn(7, d)
    assert fitted_lens.project(hidden, 1).shape == (3, 7, d)
    assert fitted_lens.project(hidden[0], 1).shape == (3, d)
    assert fitted_lens.readout_logits(hidden, 1).shape == (3, 7, vocab)
    reach = fitted_lens.temporal_reach()
    assert reach.shape == (3, 3) and (reach >= 0).all()
    with pytest.raises(KeyError):
        fitted_lens.project(hidden, 9)
    with pytest.raises(ValueError):
        fitted_lens.project(torch.randn(7, d + 1), 1)


def test_save_load_roundtrip(fitted_lens: OffsetJacobianLens, tmp_path) -> None:
    path = tmp_path / "offset_lens.pt"
    fitted_lens.save(path)
    loaded = OffsetJacobianLens.load(path, fitted_lens.model)
    assert loaded.config == fitted_lens.config
    assert (loaded.n_prompts, loaded.n_sequences) == (
        fitted_lens.n_prompts,
        fitted_lens.n_sequences,
    )
    torch.testing.assert_close(loaded.reach_sq, fitted_lens.reach_sq)
    torch.testing.assert_close(loaded.split_half_cosine, fitted_lens.split_half_cosine)
    torch.testing.assert_close(loaded.sample_counts, fitted_lens.sample_counts)
    for layer in fitted_lens.layer_indices:
        torch.testing.assert_close(loaded.jacobians[layer], fitted_lens.jacobians[layer])
        torch.testing.assert_close(loaded.corpus_atoms[layer], fitted_lens.corpus_atoms[layer])
        torch.testing.assert_close(loaded.corpus_mean[layer], fitted_lens.corpus_mean[layer])


def test_load_rejects_hidden_size_mismatch(
    fitted_lens: OffsetJacobianLens, tiny_tokenizer: PreTrainedTokenizerFast, tmp_path
) -> None:
    path = tmp_path / "offset_lens.pt"
    fitted_lens.save(path)
    with pytest.raises(ValueError):
        OffsetJacobianLens.load(path, build_tiny_model(tiny_tokenizer, hidden_size=16))


class _CharTokenizer:
    """Minimal tokenizer stand-in: one id per character, id 0 as the special-token prefix."""

    def __call__(self, text: str | list[str], add_special_tokens: bool = True) -> dict:
        def encode(s: str) -> list[int]:
            return ([0] if add_special_tokens else []) + [ord(c) for c in s]

        return {"input_ids": [encode(t) for t in text] if isinstance(text, list) else encode(text)}


def test_pack_prompts_chunks_stream_with_prefix() -> None:
    chunks = pack_prompts(
        ["ab", "cde"], _CharTokenizer(), max_length=4, min_length=2, separator="|"
    )
    assert [c.tolist() for c in chunks] == [[[0, 97, 98, 124]], [[0, 99, 100, 101]], [[0, 124]]]
    trimmed = pack_prompts(
        ["ab", "cde"], _CharTokenizer(), max_length=4, min_length=3, separator="|"
    )
    assert len(trimmed) == 2


def test_pack_prompts_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        pack_prompts([], _CharTokenizer(), max_length=4, min_length=2)
    with pytest.raises(ValueError):
        pack_prompts(["ab"], _CharTokenizer(), max_length=1, min_length=1)
