"""Shared helpers: device memory release, seeding, and prompt-file loading."""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def free_device_memory(device: torch.device | str) -> None:
    """
    Run the Python garbage collector and release cached allocator blocks.

    Parameters
    ----------
    device : torch.device | str
        Device whose caching allocator should be emptied.  CPU is a no-op
        beyond garbage collection.
    """
    gc.collect()
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (all devices) RNGs.

    Parameters
    ----------
    seed : int
        Non-negative seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_prompts(path: str | Path, limit: Optional[int] = None) -> List[str]:
    """
    Read one prompt per line, skipping blank lines and ``#`` comments.

    Parameters
    ----------
    path : str | Path
        UTF-8 text file.
    limit : int | None
        Maximum number of prompts to return.  None returns all.

    Returns
    -------
    list[str]

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file yields no prompts or ``limit`` is not positive.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts[:limit] if limit is not None else prompts
