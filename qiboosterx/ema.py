"""
Exponential Moving Average (EMA) of model parameters.

Maintains an EMA copy of the model's parameters. After training, swapping
the model's weights with the EMA weights typically improves generalization
by 0.3-2% on most tasks. Especially useful for:
  - Image generation (Diffusion models, GANs — practically universal there)
  - Self-supervised learning (BYOL, MoCo)
  - Any task with noisy gradients

This is conceptually similar to Lookahead (a momentum-style averaging) but:
  - Lookahead averages the "fast" weights every k steps, EMA averages every step
  - EMA keeps a separate copy of the weights, Lookahead modifies in place
  - EMA is the gold standard for image generation; Lookahead is more general

References:
    Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging"
    (SIAM 1992). https://epubs.siam.org/doi/10.1137/0330046

Usage:
    ema = EMA(model, decay=0.999)

    for x, y in dataloader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        ema.update()        # call AFTER each optimizer step

    # Use EMA weights for evaluation / inference
    with ema.swap():        # context manager: swap in EMA weights
        val_loss, val_acc = evaluate(model, val_loader)
    # context exits -> back to training weights
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """Maintain an exponential moving average of model parameters.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be averaged.
    decay : float, default 0.999
        EMA decay factor in [0, 1). Larger = slower (more averaging).
        Typical: 0.99 (small datasets), 0.999 (large datasets),
        0.9999 (image generation models).
    warmup_steps : int, default 0
        During the first `warmup_steps` updates, use a smaller effective
        decay so the EMA catches up quickly to the (rapidly-changing) early
        weights. After warmup, use the configured `decay`.
    device : str, optional
        Where to store the EMA copy. Defaults to the model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.device = device or next(model.parameters()).device
        self._step = 0
        self._original_weights: dict = {}

        # Initialize EMA copy with current weights
        self._ema_weights: dict[str, torch.Tensor] = {
            name: p.data.clone().detach().to(self.device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self) -> None:
        """Call after each optimizer.step(). Updates the EMA copy."""
        self._step += 1
        if self.warmup_steps > 0 and self._step <= self.warmup_steps:
            # During warmup, use a smaller decay so the EMA can catch up
            decay = min(self.decay, (1.0 + self._step) / (10.0 + self._step))
        else:
            decay = self.decay

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self._ema_weights:
                    self._ema_weights[name] = p.data.clone().detach().to(self.device)
                    continue
                ema_p = self._ema_weights[name].to(p.device)
                ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)
                # Move back to storage device if needed
                if self.device != p.device:
                    self._ema_weights[name] = ema_p.to(self.device)

    @contextlib.contextmanager
    def swap(self):
        """Temporarily replace the model's weights with the EMA weights.

        On context exit, restores the original (training) weights.

            with ema.swap():
                evaluate(model)  # uses EMA weights
            # back to training weights
        """
        # Save originals
        self._original_weights = {
            name: p.data.clone() for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        # Swap in EMA
        try:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name in self._ema_weights:
                        p.data.copy_(self._ema_weights[name].to(p.device))
            yield
        finally:
            # Restore originals
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in self._original_weights:
                        p.data.copy_(self._original_weights[name])
            self._original_weights = {}

    def copy_to_model(self) -> None:
        """Permanently replace the model's weights with the EMA weights."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._ema_weights:
                    p.data.copy_(self._ema_weights[name].to(p.device))

    def state_dict(self) -> dict:
        return {
            "ema_weights": self._ema_weights,
            "decay": self.decay,
            "step": self._step,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._ema_weights = sd["ema_weights"]
        self.decay = sd["decay"]
        self._step = sd["step"]
        self.warmup_steps = sd.get("warmup_steps", 0)

    @property
    def step_count(self) -> int:
        return self._step
