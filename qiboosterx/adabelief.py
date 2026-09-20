"""
AdaBelief optimizer.

From:
    Zhuang et al., "AdaBelief Optimizer: Adapting Stepsizes by the Belief in
    Observed Gradients" (NeurIPS 2020).
    https://arxiv.org/abs/2010.07468

AdaBelief is a drop-in replacement for Adam that adapts the step size based on
the "belief" in the current gradient direction (variance of past gradients vs
the current gradient). It typically gives:

  - Faster convergence than Adam (like RMSProp)
  - Better generalization than Adam (like SGD)
  - More stable training than Adam on noisy gradients

Drop-in: replace `torch.optim.Adam(params, lr=1e-3)` with
`AdaBelief(params, lr=1e-3)` and you're done.

Reference paper reports improvements of +0.5 to +2.0 % on CIFAR-10/ImageNet
across ResNet, WideResNet, and BERT compared to Adam.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """Implements the AdaBelief optimization algorithm.

    Parameters
    ----------
    params : iterable
    lr : float, default 1e-3
    betas : (float, float), default (0.9, 0.999)
    eps : float, default 1e-8
        Epsilon for numerical stability (added outside sqrt). The AdaBelief
        paper recommends eps=1e-16 for full precision, but this is unstable
        in mixed precision and on CPU. Use 1e-8 (default) for stability;
        advanced users can try 1e-12 with float32 + AMP.
    weight_decay : float, default 0
        L2 weight decay (decoupled from the adaptive step, similar to AdamW).
    amsgrad : bool, default False
        If True, uses the maximum of past squared gradients (like AMSGrad).
    rectify : bool, default False
        If True, uses the rectified variant (RAdam-style) for more stable
        early training when not enough gradient history is available.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        rectify: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            rectify=rectify,
        )
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
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            rectify = group["rectify"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay (like AdamW, not original Adam)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # IMPORTANT: compute the residual BEFORE updating exp_avg.
                # exp_avg gets the current grad blended in; if we compute
                # residual AFTER the update, it becomes (grad - new_exp_avg)
                # which is biased toward zero on the first step.
                grad_residual = grad - exp_avg

                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update biased second raw moment estimate
                # KEY DIFFERENCE FROM ADAM: AdaBelief uses (grad - exp_avg)^2
                # instead of grad^2. This is the "belief" — variance of the
                # gradient relative to its moving average.
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual,
                                                 value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Compute bias-corrected first and second moments
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                if rectify:
                    # RAdam-style rectification
                    rho_inf = 2.0 / (1.0 - beta2) - 1.0
                    rho = rho_inf - 2.0 * t * (beta2 ** t) / (1.0 - beta2 ** t)
                    if rho >= 5:
                        rect = (
                            (rho - 4.0) * (rho - 2.0) * rho_inf
                            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho)
                        )
                        step_size = lr * rect / bias_c1
                        p.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        # Use plain SGD with momentum when not enough history
                        step_size = lr / bias_c1
                        p.add_(exp_avg, alpha=-step_size)
                else:
                    # Standard AdaBelief step
                    step_size = lr / bias_c1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
