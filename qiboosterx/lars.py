"""
LARS (Layer-wise Adaptive Rate Scaling).

From:
    You et al., "Large Batch Training of Convolutional Networks" (2017).
    https://arxiv.org/abs/1708.03888

LARS scales the per-layer learning rate by `||w|| / (||grad|| + ||w|| * weight_decay + eps)`.
This lets you use MUCH larger batches than vanilla SGD (e.g. batch 8192 on
ImageNet with ResNet50) without losing accuracy.

Use case: distributed training with many GPUs, large batches.
Not typically useful for small-batch CPU training.

Usage:
    base = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    lars = LARS(base, trust_coefficient=0.001, weight_decay=1e-4)

    for x, y in loader:
        lars.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        lars.step()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling wrapper.

    Parameters
    ----------
    base_optimizer : Optimizer
        Typically SGD with momentum.
    trust_coefficient : float, default 0.001
        The "trust" parameter — smaller = more conservative. Default 0.001
        for SGD; try 0.002 for AdamW.
    weight_decay : float, default 0
        Per-layer weight decay. Note this is LARS's own WD, NOT the base
        optimizer's. Set the base optimizer's wd to 0 if using LARS's WD.
    eps : float, default 1e-8
        For numerical stability.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        trust_coefficient: float = 0.001,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if trust_coefficient <= 0:
            raise ValueError(f"trust_coefficient must be > 0, got {trust_coefficient}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")

        self.base_optimizer = base_optimizer
        self.trust_coefficient = trust_coefficient
        self.weight_decay = weight_decay
        self.eps = eps

        # Add LARS-specific defaults to each param group
        for group in self.base_optimizer.param_groups:
            group["trust_coefficient"] = trust_coefficient
            group["weight_decay"] = weight_decay
            group["eps"] = eps

        self.defaults = self.base_optimizer.defaults
        self.defaults.update({
            "trust_coefficient": trust_coefficient,
            "weight_decay": weight_decay,
            "eps": eps,
        })
        self.state: Dict[Any, Any] = self.base_optimizer.state
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure: Optional = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            trust = group["trust_coefficient"]
            wd = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients.")

                # Per-layer local LR: ||w|| / (||grad|| + wd * ||w|| + eps)
                weight_norm = p.norm(p=2)
                grad_norm = grad.norm(p=2)
                if weight_norm > 0 and grad_norm > 0:
                    local_lr = (
                        trust
                        * weight_norm
                        / (grad_norm + wd * weight_norm + eps)
                    )
                else:
                    local_lr = 1.0  # no scaling if either is zero

                # Compute the update direction (with weight decay applied)
                if wd > 0:
                    update = grad + wd * p
                else:
                    update = grad

                # Apply scaled update (scale by lr * local_lr)
                p.add_(update, alpha=-lr * local_lr)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_optimizer_state": self.base_optimizer.state_dict(),
            "trust_coefficient": self.trust_coefficient,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])
        self.trust_coefficient = state_dict.get("trust_coefficient", self.trust_coefficient)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)
        self.eps = state_dict.get("eps", self.eps)
