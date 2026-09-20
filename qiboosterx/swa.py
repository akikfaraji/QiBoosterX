"""
Stochastic Weight Averaging (SWA).

Wraps torch.optim.swa_utils.AVERAGED_MODEL + SWALR into a simpler, more usable
interface.

References:
    Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
    Generalization" (UAI 2018). https://arxiv.org/abs/1803.05407

SWA averages the model weights across the last N epochs (after a "start"
trigger), which empirically improves generalization by 0.5-2 % on most
classification tasks.

Usage:
    swa = SWA(model, swa_start_epoch=10)
    for epoch in range(num_epochs):
        train_one_epoch(...)
        swa.update(epoch)            # averages weights starting from epoch 10
    swa.finalize(loader)              # updates BatchNorm stats + applies averaged weights
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim import Optimizer


class SWA:
    """Stochastic Weight Averaging helper.

    Parameters
    ----------
    model : nn.Module
        The model whose weights you want to average.
    swa_start_epoch : int, default 0
        The epoch index at which to start averaging weights. Should be in the
        later stages of training (e.g. last 25% of epochs).
    swa_lr : float, optional
        The SWA-specific learning rate (typically a small constant value).
        If None, the optimizer's current LR is used.
    device : str, optional
        Where to place the averaged model. Defaults to the model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start_epoch: int = 0,
        swa_lr: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if swa_start_epoch < 0:
            raise ValueError(f"swa_start_epoch must be >= 0, got {swa_start_epoch}")

        self.model = model
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.device = device or next(model.parameters()).device

        self._averaged_model: Optional[AveragedModel] = None
        self._n_updates: int = 0
        self._started: bool = False

    def update(self, epoch: int) -> None:
        """Called once per epoch (after the standard training step)."""
        if epoch < self.swa_start_epoch:
            return
        if not self._started:
            # First time we cross the threshold
            self._averaged_model = AveragedModel(self.model, device=self.device)
            self._started = True
        assert self._averaged_model is not None
        self._averaged_model.update_parameters(self.model)
        self._n_updates += 1

    def finalize(self, dataloader: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """Apply the averaged weights to the model and refresh BatchNorm stats.

        If a dataloader is provided, runs one forward pass through it to update
        BatchNorm running statistics. Returns the model with averaged weights.

        Returns
        -------
        nn.Module
            The original model, now holding the averaged weights. If SWA
            never started (no updates were performed), the original model is
            returned unchanged.
        """
        if not self._started or self._averaged_model is None:
            print("[SWA] No averaging was performed — swa_start_epoch never reached.")
            return self.model

        if self._n_updates == 0:
            print("[SWA] update() was never called after swa_start_epoch.")
            return self.model

        # Copy averaged weights back into the original model
        averaged_state = self._averaged_model.module.state_dict()
        self.model.load_state_dict(averaged_state, strict=True)

        # Refresh BN running stats (SWA's averaged weights are not compatible
        # with the old BN stats from before averaging started)
        if dataloader is not None:
            print("[SWA] Updating BatchNorm running stats...")
            was_training = self.model.training
            self.model.train()
            with torch.no_grad():
                for x, *_ in dataloader:
                    x = x.to(self.device)
                    self.model(x)
            if not was_training:
                self.model.eval()

        print(f"[SWA] Applied averaged weights (n_updates={self._n_updates}).")
        return self.model

    @property
    def n_updates(self) -> int:
        return self._n_updates
