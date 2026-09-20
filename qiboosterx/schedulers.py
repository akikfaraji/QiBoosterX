"""
Learning-rate scheduler helpers.

Wraps the most useful PyTorch schedulers into a single builder so you can
switch between them with a single string change.

Schedulers included:
    - "onecycle" : OneCycleLR (Smith 2017) — often gives +1-2% accuracy
    - "cosine"   : CosineAnnealingLR — classical baseline
    - "cosine_wr": CosineAnnealingWarmRestarts (SGDR-style restarts)
    - "steplr"   : StepLR — old-school but occasionally useful
    - "swa"      : SWALR (constant low LR for SWA phase)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    _LRScheduler,
)

try:
    from torch.optim.swa_utils import SWALR
except ImportError:
    SWALR = None  # type: ignore


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    steps_per_epoch: Optional[int] = None,
    num_epochs: int = 1,
    max_lr: Optional[float] = None,
    **kwargs,
) -> _LRScheduler:
    """Build a learning-rate scheduler by name.

    Parameters
    ----------
    name : str
        One of: "onecycle", "cosine", "cosine_wr", "steplr", "swa".
    optimizer : Optimizer
        The optimizer to schedule.
    steps_per_epoch : int, optional
        Number of optimizer steps per epoch. Required for "onecycle".
    num_epochs : int
        Total number of epochs to schedule over.
    max_lr : float, optional
        Maximum LR for OneCycleLR. Defaults to optimizer's current lr * 10.
    """
    name = name.lower()
    if name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for the 'onecycle' scheduler")
        max_lr = max_lr or optimizer.param_groups[0]["lr"]
        total_steps = steps_per_epoch * num_epochs
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
        )
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", num_epochs),
            eta_min=kwargs.get("eta_min", 0),
        )
    if name == "cosine_wr":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", num_epochs),
            T_mult=kwargs.get("T_mult", 1),
            eta_min=kwargs.get("eta_min", 0),
        )
    if name == "steplr":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", num_epochs // 4),
            gamma=kwargs.get("gamma", 0.1),
        )
    if name == "swa":
        if SWALR is None:
            raise ImportError("SWALR is not available in your PyTorch version.")
        return SWALR(
            optimizer,
            swa_lr=kwargs.get("swa_lr", optimizer.param_groups[0]["lr"] * 0.1),
            anneal_epochs=kwargs.get("anneal_epochs", 10),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
        )
    raise ValueError(f"Unknown scheduler name: {name!r}")
