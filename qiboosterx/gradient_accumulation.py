"""
Gradient Accumulation helper for low-memory devices.

Lets you simulate a large effective batch size by accumulating gradients over
several forward-backward passes before calling optimizer.step(). On a 4GB
machine you can simulate batch_size=64 by accumulating 8 batches of 8.

Usage:
    accum = GradientAccumulator(target_batch_size=64, micro_batch_size=8)
    for i, (x, y) in enumerate(loader):
        loss = criterion(model(x), y) / accum.scale_factor
        loss.backward()
        if accum.step():  # returns True when enough micro-batches have accumulated
            optimizer.step()
            optimizer.zero_grad()

Works with or without DistributedDataParallel; just make sure you call
accum.step() exactly once per micro-batch and act on its return value.
"""

from __future__ import annotations

from typing import Optional

import torch


class GradientAccumulator:
    """Track gradient accumulation progress.

    Parameters
    ----------
    target_batch_size : int
        The effective batch size you want to simulate.
    micro_batch_size : int
        The actual batch size that fits in memory.
    """

    def __init__(self, target_batch_size: int, micro_batch_size: int) -> None:
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
        if target_batch_size < micro_batch_size:
            raise ValueError(
                f"target_batch_size ({target_batch_size}) must be >= "
                f"micro_batch_size ({micro_batch_size})"
            )
        if target_batch_size % micro_batch_size != 0:
            # Round up
            new_target = ((target_batch_size + micro_batch_size - 1)
                          // micro_batch_size) * micro_batch_size
            print(
                f"[GradientAccumulator] Warning: target_batch_size {target_batch_size} is "
                f"not a multiple of micro_batch_size {micro_batch_size}; "
                f"rounded up to {new_target}."
            )
            target_batch_size = new_target

        self.target_batch_size = target_batch_size
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps: int = target_batch_size // micro_batch_size
        self.scale_factor: float = float(self.accumulation_steps)
        self._micro_count: int = 0
        self._total_seen: int = 0

    def step(self) -> bool:
        """Advance the counter. Returns True when it's time to call optimizer.step()."""
        self._micro_count += 1
        self._total_seen += self.micro_batch_size
        if self._micro_count >= self.accumulation_steps:
            self._micro_count = 0
            return True
        return False

    def reset(self) -> None:
        self._micro_count = 0
        self._total_seen = 0

    @property
    def progress(self) -> float:
        """Fractional progress through the current accumulation window [0, 1]."""
        return self._micro_count / self.accumulation_steps

    def __repr__(self) -> str:
        return (
            f"GradientAccumulator(effective_batch={self.target_batch_size}, "
            f"micro_batch={self.micro_batch_size}, "
            f"accumulation_steps={self.accumulation_steps}, "
            f"current={self._micro_count})"
        )
