"""Model loading and residual-stream hook infrastructure."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HookedModel:
    """
    Wraps a HuggingFace causal-LM and exposes per-layer residual-stream
    activations via forward hooks, without modifying model weights.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. "Qwen/Qwen2.5-3B-Instruct").
    device : str | None
        Target device.  Defaults to "cuda" if available, else "cpu".
    dtype : torch.dtype
        Weight dtype.  Defaults to bfloat16 on CUDA, float32 on CPU.
    max_new_tokens : int
        Used only for generation helpers.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if "cuda" in self.device else torch.float32)
        self.max_new_tokens = max_new_tokens

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model: %s  device=%s  dtype=%s", model_id, self.device, self.dtype)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

        self._layers: List[nn.Module] = self._discover_layers()
        self.num_layers: int = len(self._layers)
        logger.info("Discovered %d transformer layers", self.num_layers)

    # ── layer discovery ─────────────────────────────────────────────────────

    def _discover_layers(self) -> List[nn.Module]:
        """
        Return the list of transformer block modules in order.
        Supports Qwen2, LLaMA, Mistral, Gemma, GPT-NeoX naming conventions.
        """
        candidates = [
            ("model.layers",),          # Qwen2, LLaMA, Mistral, Gemma
            ("transformer.h",),          # GPT-2, GPT-J, Falcon
            ("gpt_neox.layers",),        # GPT-NeoX
            ("model.decoder.layers",),   # OPT
        ]
        for path_tuple in candidates:
            obj = self.model
            for attr in path_tuple[0].split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return list(obj)
        raise RuntimeError(
            f"Cannot discover transformer layers for {self.model_id}. "
            "Add the layer path to HookedModel._discover_layers()."
        )

    # ── tokenization helpers ────────────────────────────────────────────────

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of strings and move tensors to model device."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ── hook context managers ───────────────────────────────────────────────

    @contextmanager
    def capture_residuals(
        self,
        layer_indices: Optional[List[int]] = None,
    ) -> Iterator[Dict[int, torch.Tensor]]:
        """
        Context manager that captures post-block residual-stream tensors.

        Yields a dict mapping layer_index → tensor of shape (batch, seq, d_model).
        Tensors are detached from the compute graph (for inspection only).

        Parameters
        ----------
        layer_indices : list[int] | None
            Layers to capture.  None means all layers.
        """
        target_indices = set(layer_indices) if layer_indices is not None else set(range(self.num_layers))
        store: Dict[int, torch.Tensor] = {}
        handles = []

        for idx, layer in enumerate(self._layers):
            if idx not in target_indices:
                continue

            def _hook(module: nn.Module, _input: tuple, output: tuple, _idx: int = idx) -> None:
                # transformer blocks return (hidden_state, ...) or just hidden_state
                hs = output[0] if isinstance(output, tuple) else output
                store[_idx] = hs.detach().float()

            handles.append(layer.register_forward_hook(_hook))

        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def capture_residuals_with_grad(
        self,
        layer_indices: Optional[List[int]] = None,
    ) -> Iterator[Dict[int, torch.Tensor]]:
        """
        Like capture_residuals but retains gradients for Jacobian computation.
        Tensors in the yielded dict have requires_grad=True.
        """
        target_indices = set(layer_indices) if layer_indices is not None else set(range(self.num_layers))
        store: Dict[int, torch.Tensor] = {}
        handles = []

        for idx, layer in enumerate(self._layers):
            if idx not in target_indices:
                continue

            def _hook(module: nn.Module, _input: tuple, output: tuple, _idx: int = idx) -> None:
                hs = output[0] if isinstance(output, tuple) else output
                # clone so we can attach grad without corrupting the forward pass
                hs_clone = hs.clone().requires_grad_(True)
                store[_idx] = hs_clone
                # returning hs_clone re-wires the graph through this leaf
                if isinstance(output, tuple):
                    return (hs_clone,) + output[1:]
                return hs_clone

            handles.append(layer.register_forward_hook(_hook))

        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    # ── unembedding projection ───────────────────────────────────────────────

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits via the LM head.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape (..., d_model).

        Returns
        -------
        torch.Tensor
            Shape (..., vocab_size).
        """
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Cannot find lm_head on model.")

        # apply final layer norm if the model has one separate from lm_head
        norm = (
            getattr(self.model.model, "norm", None)
            or getattr(self.model.model, "final_layernorm", None)
            or getattr(self.model.transformer, "ln_f", None)
        )
        if norm is not None:
            hidden = norm(hidden.to(self.dtype))

        return lm_head(hidden.to(self.dtype))

    # ── vocab helpers ────────────────────────────────────────────────────────

    def top_tokens(
        self,
        logits: torch.Tensor,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return top-k (token_str, prob) pairs from a logit vector."""
        probs = torch.softmax(logits.float(), dim=-1)
        top_vals, top_ids = probs.topk(k)
        return [
            (self.tokenizer.decode([int(tid)]).strip(), float(val))
            for tid, val in zip(top_ids, top_vals)
        ]

    def vocab_size(self) -> int:
        """Return vocabulary size of the model."""
        return self.model.config.vocab_size
