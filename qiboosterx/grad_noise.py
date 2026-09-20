"""
Gradient Noise Adder.

From:
    Neelakantan et al., "Adding Gradient Noise Improves Learning for Very
    Deep Networks" (ICLR 2015 Workshop).
    https://arxiv.org/abs/1511.06807

Adds Gaussian noise to the GRADIENT (not the parameters, like SGLD) before the
optimizer step. The noise scale decays as training progresses, following a
1/t schedule.

Difference from SGLD:
  - SGLD: adds noise to PARAMETERS after the step. Mathematically grounded
    in Langevin dynamics. Used for Bayesian sampling.
  - GradientNoiseAdder: adds noise to the GRADIENT before the step. Empirical
    observation that helps very deep networks. The paper reports improvements
    on tasks with very deep networks (10+ layers).
  - The two are NOT interchangeable; pick based on your use case.

Usage:
    base = torch.optim.Adam(model.parameters(), lr=1e-3)
    gna = GradientNoiseAdder(base, eta=0.01, gamma=0.55)

    for x, y in loader:
        base.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        gna.step(step_number=global_step)  # noise decays with step number
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer


class GradientNoiseAdder(Optimizer):
    """Wrap a base optimizer; add decaying noise to gradients before each step.

    Parameters
    ----------
    base_optimizer : Optimizer
    eta : float, default 0.01
        Initial noise variance. Larger = more noise early in training.
    gamma : float, default 0.55
        Decay exponent. The noise variance at step t is `eta / (1 + t)^gamma`.
        gamma=0.55 is the value recommended in the paper.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        eta: float = 0.01,
        gamma: float = 0.55,
    ) -> None:
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        self.base_optimizer = base_optimizer
        self.eta = eta
        self.gamma = gamma
        self._step: int = 0

        self.defaults = self.base_optimizer.defaults
        self.defaults.update({"eta": eta, "gamma": gamma})
        self.state: Dict[Any, Any] = self.base_optimizer.state
        self.param_groups = self.base_optimizer.param_groups

    def step(self, closure=None, step_number: Optional[int] = None) -> Optional[float]:
        """Step the optimizer with noise added to gradients.

        Parameters
        ----------
        closure : callable, optional
        step_number : int, optional
            If None, increments an internal counter. Pass in your global step
            count for reproducibility across resume-from-checkpoint runs.
        """
        if step_number is not None:
            self._step = int(step_number)
        else:
            self._step += 1

        # Compute the noise variance: eta / (1 + t)^gamma
        variance = self.eta / ((1.0 + self._step) ** self.gamma)
        std = variance ** 0.5

        # Add Gaussian noise to each parameter's gradient
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    noise = torch.randn_like(p.grad) * std
                    p.grad.add_(noise)

        # Step the base optimizer using the noised gradient
        return self.base_optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_optimizer_state": self.base_optimizer.state_dict(),
            "eta": self.eta,
            "gamma": self.gamma,
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])
        self.eta = state_dict.get("eta", self.eta)
        self.gamma = state_dict.get("gamma", self.gamma)
        self._step = state_dict.get("step", 0)
