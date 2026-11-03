from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import torch


T = TypeVar("T")


def group_contexts(contexts: Sequence[T], window_batch_size: int) -> list[list[T]]:
    """Split ordered temporal contexts into inference batches."""

    if isinstance(window_batch_size, bool) or not isinstance(window_batch_size, int):
        raise ValueError("window_batch_size must be an integer")
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be at least 1")
    return [
        list(contexts[start : start + window_batch_size])
        for start in range(0, len(contexts), window_batch_size)
    ]


def gather_context_batch(
    values: torch.Tensor, contexts: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Gather ordered equal-length windows into flattened ``batch * time``."""

    if not contexts:
        raise ValueError("contexts must not be empty")
    lengths = {len(context) for context in contexts}
    if len(lengths) != 1:
        raise ValueError("all contexts in a batch must have equal length")
    return torch.cat([values[list(context)] for context in contexts], dim=0)


def split_context_batch(
    values: torch.Tensor, *, group_size: int, num_frames: int
) -> list[torch.Tensor]:
    """Undo :func:`gather_context_batch` while preserving context order."""

    if group_size < 1 or num_frames < 1:
        raise ValueError("group_size and num_frames must be at least 1")
    expected = group_size * num_frames
    if values.shape[0] != expected:
        raise ValueError(
            f"context batch has {values.shape[0]} frames, expected {expected}"
        )
    return list(values.split(num_frames, dim=0))
