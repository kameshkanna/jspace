# The LRP rules in this module are adapted from safety-research/jpp_lens (Ayonrinde & Lindsey,
# 2026; Apache-2.0), whose rules derive from the R-Lens implementation of Camila Blank and Agam
# Bhatia.  See THIRD_PARTY_NOTICES.md.
"""
LRP backward-pass rules (RelP) for dense Qwen2 / Qwen3 / Llama / Mistral decoders.

The J++ Lens computes its Jacobians through a Layer-wise Relevance Propagation
backward graph instead of the raw gradient.  As in J++, this is not a custom
backward: specific sub-expressions are detached in the *forward* graph, so
ordinary autograd yields the propagation coefficients while forward values stay
unchanged.

  * LN rule        residual-stream RMSNorm with its rsqrt(mean(x²) + eps) factor
                   detached (a fixed per-position scale in the backward)
  * identity rule  SiLU(z) = z · sigmoid(z) with the sigmoid detached
  * half rule      gated-MLP product p → ½p + ½p.detach(), splitting relevance
                   evenly between the gate and up paths

Attention, and RMSNorms inside attention (Qwen3's q_norm / k_norm), keep
standard gradients.  Patching is per module *instance* and fully reversed on
exit; the detached ops persist in any graph recorded inside the context, so
backward passes run later still see the LRP coefficients.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Iterator, List, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

_RMSNORM_CLASSES = ("Qwen2RMSNorm", "Qwen3RMSNorm", "LlamaRMSNorm", "MistralRMSNorm")
_GATED_MLP_CLASSES = ("Qwen2MLP", "Qwen3MLP", "LlamaMLP", "MistralMLP")
_SILU_CLASSES = ("SiLU", "SiLUActivation")
_MISSING = object()


@dataclass(frozen=True)
class LrpRules:
    """
    Which LRP rules to apply.  The defaults are the J++ / R-Lens ``rlens`` preset.

    Attributes
    ----------
    ln_rule : bool
        Detach the RMSNorm scale on residual-stream norms.
    identity_rule : bool
        Detach the sigmoid inside SiLU.
    half_rule : bool
        Half-detach the gated-MLP product.
    """

    ln_rule: bool = True
    identity_rule: bool = True
    half_rule: bool = True

    def any(self) -> bool:
        """True when at least one rule is enabled."""
        return self.ln_rule or self.identity_rule or self.half_rule


def _rmsnorm_ln_rule_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Stock Qwen2/Llama RMSNorm forward with the rsqrt factor detached."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    scale = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    hidden_states = hidden_states * scale.detach()
    return self.weight * hidden_states.to(input_dtype)


def _make_gated_mlp_forward(
    identity_rule: bool, half_rule: bool
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """Stock ``down(silu(gate(x)) * up(x))`` with the identity and/or half rule."""

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        activated = gate * torch.sigmoid(gate).detach() if identity_rule else self.act_fn(gate)
        product = activated * self.up_proj(x)
        if half_rule:
            product = 0.5 * product + 0.5 * product.detach()
        return self.down_proj(product)

    return forward


def _is_attention_internal(name: str) -> bool:
    """Modules under a ``self_attn`` block keep standard gradients."""
    return "self_attn" in name.split(".")


@contextmanager
def lrp_rules(model: nn.Module, rules: LrpRules = LrpRules()) -> Iterator[List[str]]:
    """
    Reversibly patch a dense decoder so that autograd computes LRP coefficients.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace causal LM (or its base model).
    rules : LrpRules

    Yields
    ------
    list[str]
        Qualified names of every patched module.

    Raises
    ------
    RuntimeError
        On an RMSNorm or MLP class without a verified patched forward, or when
        the requested rules patched nothing.
    ValueError
        When the identity rule meets a non-SiLU gated MLP.
    """
    saved: List[Tuple[nn.Module, object]] = []
    patched: List[str] = []

    def patch(module: nn.Module, forward: Callable[..., torch.Tensor]) -> None:
        saved.append((module, module.__dict__.get("forward", _MISSING)))
        object.__setattr__(module, "forward", MethodType(forward, module))

    try:
        if rules.any():
            mlp_forward = _make_gated_mlp_forward(rules.identity_rule, rules.half_rule)
            for name, module in model.named_modules():
                cls = type(module).__name__
                if cls.endswith("RMSNorm"):
                    if not rules.ln_rule or _is_attention_internal(name):
                        continue
                    if cls not in _RMSNORM_CLASSES:
                        raise RuntimeError(f"No LN-rule forward for RMSNorm class {cls} at {name}")
                    patch(module, _rmsnorm_ln_rule_forward)
                    patched.append(name)
                elif cls.endswith("MLP") and (rules.identity_rule or rules.half_rule):
                    if cls not in _GATED_MLP_CLASSES:
                        raise RuntimeError(f"No LRP forward for MLP class {cls} at {name}")
                    act_name = type(module.act_fn).__name__
                    if rules.identity_rule and act_name not in _SILU_CLASSES:
                        raise ValueError(f"identity rule needs SiLU, got {act_name} at {name}")
                    patch(module, mlp_forward)
                    patched.append(name)
            if not patched:
                raise RuntimeError("LRP rules requested but no module was patched")
            logger.debug("LRP rules %s patched %d modules", rules, len(patched))
        yield patched
    finally:
        for module, original in reversed(saved):
            if original is _MISSING:
                module.__dict__.pop("forward", None)
            else:
                object.__setattr__(module, "forward", original)
