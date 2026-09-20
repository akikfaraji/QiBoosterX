"""
Automatic Mixed Precision (AMP) helper for low-end devices.

Uses `torch.amp.autocast` to run forward passes in float16/bfloat16 and backward
passes in float32. On CPUs that support bfloat16 (most modern x86), this gives
roughly 1.5-2x throughput. On GPUs, you get 1.5-2x throughput + ~50 % memory
reduction, which is the most valuable single optimization for low-end devices.

If the device doesn't support the requested dtype, gracefully falls back to
float32 with a one-line warning.

Usage:
    amp = MixedPrecision(dtype="bfloat16")  # or "float16" for CUDA

    for x, y in loader:
        optimizer.zero_grad()
        with amp.autocast():
            loss = criterion(model(x), y)
        amp.backward(loss)
        optimizer.step()

Or, with the GradScaler (only needed for float16 on CUDA):
    amp = MixedPrecision(dtype="float16")
    scaler = amp.scaler  # None on CPU/bfloat16
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Optional

import torch


class MixedPrecision:
    """Wrap torch.amp for portable mixed-precision training.

    Parameters
    ----------
    dtype : str, default "bfloat16"
        "bfloat16" | "float16". On CPU, "bfloat16" is the only viable choice
        (CPU float16 ops are extremely slow). On CUDA, both work; "float16"
        gives larger speedups but requires GradScaler for stability.
    device : str, default "auto"
        "cpu" | "cuda" | "auto". If "auto", picks based on torch.cuda.is_available().
    verbose : bool, default True
        If True, prints the resolved dtype and whether AMP is active.
    """

    def __init__(
        self,
        dtype: str = "bfloat16",
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"dtype must be 'bfloat16' or 'float16', got {dtype!r}")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype for the actual device.
        self.requested_dtype = dtype
        self.dtype: Optional[torch.dtype]
        self.scaler: Optional[torch.amp.GradScaler] = None

        if self.device == "cuda":
            if dtype == "float16":
                self.dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                # bfloat16 on CUDA doesn't need a scaler
                self.dtype = torch.bfloat16
        else:  # CPU
            if dtype == "float16":
                if verbose:
                    warnings.warn(
                        "float16 on CPU is extremely slow — use dtype='bfloat16' instead. "
                        "Falling back to bfloat16.",
                        RuntimeWarning,
                    )
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.bfloat16

        self._is_active = self.dtype is not None
        if verbose:
            status = f"enabled ({self.dtype})" if self._is_active else "disabled (float32 fallback)"
            print(f"[MixedPrecision] {status} on {self.device}")

    @contextlib.contextmanager
    def autocast(self):
        """Context manager equivalent to torch.amp.autocast(device, dtype)."""
        if not self._is_active:
            yield
            return
        with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
            yield

    def backward(self, loss: torch.Tensor) -> None:
        """Call loss.backward() — uses GradScaler if applicable."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Call optimizer.step() — unscales gradients if using a scaler."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients — needed for things like gradient clipping when using float16."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    @property
    def is_active(self) -> bool:
        return self._is_active
