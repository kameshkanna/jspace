"""
Readout Filtering (J++): rank only tokens whose decoded string has a letter or digit.

Whitespace, punctuation, symbol tokens, and LM-head padding rows beyond the
tokenizer's vocabulary are excluded from readout rankings.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from transformers import PreTrainedTokenizerBase


def is_semantic(decoded: str) -> bool:
    """True when the decoded token contains at least one letter or digit."""
    return any(char.isalnum() for char in decoded)


def semantic_token_mask(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    keep_ids: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """
    Boolean mask over the LM-head vocabulary: True for rankable (semantic) ids.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    vocab_size : int
        LM-head width; ids at or beyond ``len(tokenizer)`` are padding rows.
    keep_ids : Iterable[int] | None
        Ids always kept rankable (e.g. evaluation targets).

    Returns
    -------
    torch.Tensor
        (vocab_size,) bool on CPU.
    """
    n_real = min(len(tokenizer), vocab_size)
    decoded = tokenizer.batch_decode([[i] for i in range(n_real)])
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:n_real] = torch.tensor([is_semantic(s) for s in decoded], dtype=torch.bool)
    special = [i for i in (tokenizer.all_special_ids or []) if i < vocab_size]
    mask[special] = False
    if keep_ids is not None:
        keep = torch.tensor(sorted(set(keep_ids)), dtype=torch.long)
        if keep.numel():
            mask[keep] = True
    return mask
