"""Tests for the LRP backward-pass rules."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from jspace.lrp import LrpRules, lrp_rules
from tests.conftest import build_tiny_model


@pytest.fixture
def model64(tiny_tokenizer: PreTrainedTokenizerFast):
    """float64 tiny Llama, for exact gradient comparisons."""
    return build_tiny_model(tiny_tokenizer, init_range=0.2, dtype=torch.float64).model


def _input_grad(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    (out,) = torch.autograd.grad(module(x).sum(), x)
    return out


def test_forward_values_unchanged(model64: nn.Module) -> None:
    ids = torch.randint(2, 40, (1, 9), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        stock = model64(input_ids=ids).logits
        with lrp_rules(model64) as patched:
            lrp = model64(input_ids=ids).logits
    assert patched
    torch.testing.assert_close(lrp, stock, rtol=1e-12, atol=1e-12)


def test_ln_rule_detaches_the_rms_scale(model64: nn.Module) -> None:
    norm = model64.model.layers[0].input_layernorm
    x = torch.randn(3, model64.config.hidden_size, dtype=torch.float64)
    with lrp_rules(model64, LrpRules(ln_rule=True, identity_rule=False, half_rule=False)):
        grad = _input_grad(norm, x)
    expected = norm.weight * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    torch.testing.assert_close(grad, expected.expand_as(x))


def test_half_rule_halves_the_mlp_gradient(model64: nn.Module) -> None:
    mlp = model64.model.layers[1].mlp
    x = torch.randn(4, model64.config.hidden_size, dtype=torch.float64)
    with lrp_rules(model64, LrpRules(ln_rule=False, identity_rule=True, half_rule=False)):
        identity_only = _input_grad(mlp, x)
    with lrp_rules(model64, LrpRules(ln_rule=False, identity_rule=True, half_rule=True)):
        with_half = _input_grad(mlp, x)
    torch.testing.assert_close(with_half, 0.5 * identity_only)


def test_identity_rule_changes_gradient_and_context_restores_stock(model64: nn.Module) -> None:
    mlp = model64.model.layers[0].mlp
    x = torch.randn(4, model64.config.hidden_size, dtype=torch.float64)
    stock = _input_grad(mlp, x)
    with lrp_rules(model64, LrpRules(ln_rule=False, identity_rule=True, half_rule=False)):
        identity = _input_grad(mlp, x)
    restored = _input_grad(mlp, x)
    assert not torch.allclose(identity, stock)
    torch.testing.assert_close(restored, stock)
    assert "forward" not in mlp.__dict__


def test_unknown_norm_class_raises() -> None:
    class FooRMSNorm(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(RuntimeError):
        with lrp_rules(nn.Sequential(FooRMSNorm())):
            pass


def test_requested_rules_must_patch_something() -> None:
    with pytest.raises(RuntimeError):
        with lrp_rules(nn.Sequential(nn.Linear(2, 2))):
            pass
    with lrp_rules(nn.Sequential(nn.Linear(2, 2)), LrpRules(False, False, False)) as patched:
        assert patched == []
