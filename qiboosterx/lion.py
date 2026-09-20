"""
Lion (EvoLved Sign Momentum) optimizer.

From:
    Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023).
    https://arxiv.org/abs/2302.06675

Discovered via program search by Google Research. Lion uses the sign of the
exponential moving average of gradients (not the gradient magnitude itself)
as the update direction, with a momentum-like update.

Lion typically matches or beats AdamW on vision + language tasks while:
  - Using ~50% less memory (only keeps momentum, NOT variance)
  - Requiring a smaller learning rate than AdamW (typically lr_Lion = lr_AdamW / 10)
  - Requiring a smaller weight decay than AdamW (typically wd_Lion = wd_AdamW / 10)

Drop-in:
    # AdamW version
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Lion equivalent (note the 10x smaller lr AND wd)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.001)
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Implements the Lion optimizer.

    Parameters
    ----------
    params : iterable
    lr : float, default 1e-4
        IMPORTANT: Lion typically needs 10x smaller LR than AdamW.
        Use 1e-4 for most tasks (vs 1e-3 for AdamW).
    betas : (float, float), default (0.9, 0.99)
        (momentum, intermomentum). Default (0.9, 0.99) from the paper.
    weight_decay : float, default 0.0
        Decoupled weight decay. IMPORTANT: Lion typically needs 10x smaller
        weight decay than AdamW. If you used wd=0.01 with AdamW, use wd=0.001
        with Lion.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay (decoupled, applied BEFORE the update direction)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # The Lion update:
                # 1. update = sign(beta1 * exp_avg + (1-beta1) * grad)
                # 2. exp_avg = beta2 * exp_avg + (1-beta2) * grad
                # 3. p = p - lr * update
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
                p.add_(update.sign(), alpha=-lr)

                # Update momentum AFTER computing the update direction
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
