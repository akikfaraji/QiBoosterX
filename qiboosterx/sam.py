"""
Sharpness-Aware Minimization (SAM) optimizer.

Implements the algorithm from:
    Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
    Generalization" (ICLR 2021).
    https://arxiv.org/abs/2010.01412

SAM finds parameters in neighborhoods with uniformly low loss (flatter minima)
rather than parameters that just have low loss at a single point. Flatter minima
generalize better.

Usage:
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sam = SAM(model.parameters(), base_optimizer, rho=0.05)

    def forward_loss():
        return loss_fn(model(inputs), targets)

    # First forward-backward: find the perturbation direction
    loss = forward_loss()
    loss.backward()
    sam.first_step(zero_grad=True)

    # Second forward-backward: actually step using the perturbed gradients
    loss2 = forward_loss()
    loss2.backward()
    sam.second_step(zero_grad=True)

Typical improvement: +0.5 to +1.5 % accuracy on image classification, especially
on small datasets. Roughly doubles training cost per step (two forward-backward
passes per step).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper.

    Parameters
    ----------
    params : iterable
        Model parameters (will be passed to base_optimizer).
    base_optimizer : Optimizer
        Instantiated base optimizer (Adam, SGD, AdamW, ...).
    rho : float, default 0.05
        Neighborhood radius. Larger rho = stronger regularization but less stable.
        Typical range: 0.01 to 0.5. For SGD-based training use 0.05; for
        adaptive optimizers (Adam) use 0.1-0.5.
    adaptive : bool, default False
        If True, use per-parameter rho scaling (ASAM variant from Liu et al. 2021).
        Often works better with adaptive optimizers like Adam.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive

        # Make sure base_optimizer is bound to the same params
        params_list = list(params)
        for group in self.base_optimizer.param_groups:
            group["rho"] = rho

        self.defaults = dict(self.base_optimizer.defaults)
        self.defaults.update({"rho": rho, "adaptive": adaptive})
        self.state: Dict[Any, Any] = self.base_optimizer.state
        self.param_groups = self.base_optimizer.param_groups

    # ------------------------------------------------------------------ #
    # SAM two-step API.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Compute the perturbation direction and apply it to the parameters.

        Saves the original weights so they can be restored in second_step().
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_old = p.data.clone()
                # Save the original position so we can restore it
                self.state[p]["old_param"] = p_old
                # Perturbation direction = gradient scaled to have norm rho
                e_w = p.grad * scale.to(p.device)
                if self.adaptive:
                    # ASAM (Liu et al. 2021): also scale by parameter norm
                    p_norm = self._param_norm(p).to(p.device)
                    e_w = e_w * (p_norm + 1e-12)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights and step using the (now recomputed) gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_param" not in self.state[p]:
                    raise RuntimeError(
                        "second_step() called before first_step() — "
                        "you must call first_step() first, then recompute gradients."
                    )
                p.data.copy_(self.state[p]["old_param"])
                del self.state[p]["old_param"]
        # Take the actual optimizer step using gradients computed at the perturbed point
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    # Convenience for code that wants a one-call API.
    # ------------------------------------------------------------------ #
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """Two-forward-pass step. closure() must return the loss (no backward).

        The closure is called twice. Each call should perform a fresh forward
        pass and return the loss; SAM handles the backward() calls internally.
        """
        # Pass 1: compute loss & gradient at current (unperturbed) parameters
        loss = closure()
        loss.backward()
        self.first_step(zero_grad=True)
        # Pass 2: compute loss & gradient at perturbed parameters
        loss2 = closure()
        loss2.backward()
        self.second_step(zero_grad=True)
        return loss2

    def first_forward(self, closure: Callable[[], float]) -> float:
        """Run closure() — does NOT call backward. Use this in a two-step manual loop:

            loss = sam.first_forward(closure)  # forward only
            loss.backward()                    # backward pass 1
            sam.first_step(zero_grad=True)
            loss2 = closure()                  # forward pass 2
            loss2.backward()                   # backward pass 2
            sam.second_step(zero_grad=True)
        """
        return closure()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.grad) if self.adaptive else 1.0) * p.grad)
                .norm(p=2)
                .to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def _param_norm(self, p: torch.Tensor) -> torch.Tensor:
        return p.norm(p=2)

    # ------------------------------------------------------------------ #
    # Pass-through methods.
    # ------------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_optimizer_state": self.base_optimizer.state_dict(),
            "rho": self.rho,
            "adaptive": self.adaptive,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])
        self.rho = state_dict.get("rho", self.rho)
        self.adaptive = state_dict.get("adaptive", self.adaptive)
