"""Shared fixtures: an offline word-level tokenizer and tiny random Llama models."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from jspace.model import HookedModel
from jspace.offset_lens import OffsetJacobianLens, OffsetLensConfig

PROMPTS = [
    "the cat sat on the mat and the dog ran fast",
    "one two three four five and the sun and the moon",
    "a big red car and a small blue house by the sea",
]

_WORDS = (
    "the a cat dog sat on mat ran fast slow big small red blue green one two three four five "
    "and or but if then because so very house tree car bird fish sky sea sun moon star day "
    "night good bad happy sad"
).split()


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer built in memory (no Hub access)."""
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(_WORDS)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")


def build_tiny_model(
    tokenizer: PreTrainedTokenizerFast,
    hidden_size: int = 32,
    n_layers: int = 3,
    init_range: float = 0.02,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> HookedModel:
    """Randomly initialised Llama wrapped as a HookedModel on CPU."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        initializer_range=init_range,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config).to(dtype)
    return HookedModel.from_components(model, tokenizer, model_id="tiny-llama")


@pytest.fixture
def tiny_model(tiny_tokenizer: PreTrainedTokenizerFast) -> HookedModel:
    """Three-layer, d=32 random Llama in float32."""
    return build_tiny_model(tiny_tokenizer)


@pytest.fixture
def fitted_lens(tiny_model: HookedModel) -> OffsetJacobianLens:
    """Tiny offset lens (Δ ∈ {0, 1, 2}) fitted end-to-end on string prompts."""
    config = OffsetLensConfig(offsets=(0, 1, 2), n_proj=6, max_length=32, seed=3)
    return OffsetJacobianLens(tiny_model, config).fit(PROMPTS, show_progress=False)
