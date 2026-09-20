"""
Lookahead optimizer wrapper.

Implements the algorithm from:
    Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019).
    https://arxiv.org/abs/1907.08610

Lookahead maintains two sets of weights: "fast" (updated by the inner optimizer
such as Adam/SGD) and "slow" (an exponential moving average of the fast weights
every k steps). This consistently improves generalization and stability across
tasks, with no extra hyperparameter tuning beyond choosing k and alpha.

Typical improvement: +0.5 to +2.0 % accuracy on image classification, with
smoother loss curves. Original paper reports improvements on CIFAR-10/ImageNet
across ResNet, WideResNet, etc.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    """Wrap any base optimizer with the Lookahead algorithm.

    Parameters
    ----------
    base_optimizer : Optimizer
        Any torch.optim.Optimizer (Adam, SGD, AdamW, ...).
    k : int, default 5
        Number of inner (fast) steps before each outer (slow) sync.
    alpha : float, default 0.5
        Slow-weights interpolation factor in [0, 1]. Larger means the slow
        weights track the fast weights more closely; smaller means more smoothing.
    pullback_momentum : str, default "none"
        "none" | "reset" | "pullback". How to handle inner optimizer momentum
        after the slow-weight sync.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = "none",
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if pullback_momentum not in {"none", "reset", "pullback"}:
            raise ValueError(
                "pullback_momentum must be 'none', 'reset', or 'pullback'; "
                f"got {pullback_momentum!r}"
            )

        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum
        self._step_counter: int = 0
        self._slow_weights: Dict[int, torch.Tensor] = {}

        # Register the slow weights as buffers on the param tensors' first
        # device so state_dict round-trips without surprises.
        for group in self.base_optimizer.param_groups:
            group["step_counter"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    self._slow_weights[id(p)] = p.data.clone().detach()

        # Lookahead-specific defaults (used by state_dict round-trip)
        self.defaults = self.base_optimizer.defaults
        self.defaults.update({"lookahead_k": k, "lookahead_alpha": alpha})
        self.state: Dict[Any, Any] = self.base_optimizer.state
        self.param_groups = self.base_optimizer.param_groups

    # ------------------------------------------------------------------ #
    # Core Lookahead step.
    # ------------------------------------------------------------------ #
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = self.base_optimizer.step(closure=closure)
        self._step_counter += 1
        if self._step_counter % self.k == 0:
            self._sync_slow_weights()
        return loss

    def _sync_slow_weights(self) -> None:
        """Pull fast weights toward slow weights (the 'look ahead' step)."""
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                slow = self._slow_weights[id(p)]
                # slow <- slow + alpha * (fast - slow)
                slow.add_(p.data - slow, alpha=self.alpha)
                # fast <- slow  (the network now uses the slow weights)
                p.data.copy_(slow)

                # Optional momentum handling
                if self.pullback_momentum == "pullback":
                    state = self.base_optimizer.state.get(p, {})
                    for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                        if key in state and state[key].shape == p.shape:
                            state[key].mul_(self.alpha).add_(slow.data - p.data, alpha=1.0 - self.alpha)
                elif self.pullback_momentum == "reset":
                    state = self.base_optimizer.state.get(p, {})
                    for key in ("momentum_buffer",):
                        if key in state:
                            state[key].zero_()

    # ------------------------------------------------------------------ #
    # Pass-through methods so it behaves like a normal Optimizer.
    # ------------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        base_state = self.base_optimizer.state_dict()
        return {
            "base_optimizer_state": base_state,
            "lookahead_k": self.k,
            "lookahead_alpha": self.alpha,
            "lookahead_step": self._step_counter,
            "slow_weights": {pid: t for pid, t in self._slow_weights.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])
        self.k = state_dict.get("lookahead_k", self.k)
        self.alpha = state_dict.get("lookahead_alpha", self.alpha)
        self._step_counter = state_dict.get("lookahead_step", 0)
        self._slow_weights = {int(pid): t for pid, t in state_dict.get("slow_weights", {}).items()}

    @property
    def param_groups(self) -> list:  # type: ignore[override]
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list) -> None:  # noqa: D401
        # No-op; we always delegate to the base optimizer.
        pass
