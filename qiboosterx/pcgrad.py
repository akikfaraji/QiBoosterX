"""
PCGrad (Projecting Conflicting Gradients).

From:
    Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
    https://arxiv.org/abs/2001.06782

For multi-task learning where you compute multiple losses and sum them,
gradients from different tasks may CONFLICT (point in opposite directions).
Naive summing cancels them out. PCGrad projects each task's gradient onto the
normal plane of any conflicting gradient, removing the conflict.

Shown to improve multi-task learning by 5-15% on MultiMNIST, Cityscapes, etc.

Usage:
    pcgrad = PCGrad(base_optimizer, num_tasks=2)

    for x, y1, y2 in loader:
        base_optimizer.zero_grad()
        loss1, loss2 = model(x), targets=(y1, y2)
        # IMPORTANT: compute both losses' grads into the same .grad buffer
        # (use retain_graph=True so the first backward doesn't free the graph)
        loss1.backward(retain_graph=True)
        loss2.backward()
        # PCGrad projects conflicting gradients in place
        pcgrad.step(closure=None)  # steps the base optimizer
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer


class PCGrad:
    """Projecting Conflicting Gradients wrapper.

    Parameters
    ----------
    base_optimizer : Optimizer
    num_tasks : int
        Number of tasks (losses) you're summing.
    """

    def __init__(self, base_optimizer: Optimizer, num_tasks: int) -> None:
        if num_tasks < 2:
            raise ValueError(f"num_tasks must be >= 2 (PCGrad is for multi-task), got {num_tasks}")

        self.base_optimizer = base_optimizer
        self.num_tasks = num_tasks
        self._grad_buffer: list[list[torch.Tensor]] = []  # [task_idx][param_idx] -> grad

    def collect_task_grads(self, task_idx: int) -> None:
        """Snapshot the current .grad buffer for task `task_idx`.

        Call this AFTER each task's loss.backward() but BEFORE the next task's
        backward (which will overwrite .grad).

            loss1.backward(retain_graph=True)
            pcgrad.collect_task_grads(0)
            loss2.backward(retain_graph=True)
            pcgrad.collect_task_grads(1)
            pcgrad.step()
        """
        if task_idx < 0 or task_idx >= self.num_tasks:
            raise IndexError(f"task_idx {task_idx} out of range [0, {self.num_tasks})")

        # Grow the buffer to fit this task
        while len(self._grad_buffer) <= task_idx:
            self._grad_buffer.append([])

        # Snapshot current grads
        grads = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                else:
                    grads.append(None)
        self._grad_buffer[task_idx] = grads

    @torch.no_grad()
    def step(self, closure: Optional = None) -> Optional[float]:
        """Project conflicting gradients, sum, and step the base optimizer.

        For each task's gradient g_i, for every other task's gradient g_j:
          - If g_i . g_j < 0 (conflicting), project g_i onto the normal plane of g_j:
            g_i <- g_i - (g_i . g_j / |g_j|^2) * g_j

        After all projections, sum the (now conflict-free) gradients and apply.
        """
        if len(self._grad_buffer) != self.num_tasks:
            raise RuntimeError(
                f"Expected grads for {self.num_tasks} tasks, "
                f"got {len(self._grad_buffer)}. Did you forget to call "
                "collect_task_grads() after each loss.backward()?"
            )

        # Stack grads into a per-task list indexed by param position
        # grads_per_task[i] = list of grad tensors for task i (one per param)
        grads_per_task = self._grad_buffer  # alias for clarity

        # Get the number of params (assume all tasks have same param count)
        n_params = len(grads_per_task[0])

        # Project each task's gradient against every other task's gradient
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i == j:
                    continue
                # Compute g_i . g_j summed across all params
                dot_ij = 0.0
                norm_sq_j = 0.0
                for k in range(n_params):
                    g_i = grads_per_task[i][k]
                    g_j = grads_per_task[j][k]
                    if g_i is None or g_j is None:
                        continue
                    dot_ij += torch.dot(g_i.flatten(), g_j.flatten()).item()
                    norm_sq_j += torch.dot(g_j.flatten(), g_j.flatten()).item()

                if dot_ij < 0:
                    # Conflict! Project g_i onto the normal plane of g_j
                    scale = dot_ij / (norm_sq_j + 1e-12)
                    for k in range(n_params):
                        g_i = grads_per_task[i][k]
                        g_j = grads_per_task[j][k]
                        if g_i is None or g_j is None:
                            continue
                        grads_per_task[i][k] = g_i - scale * g_j

        # Sum the projected gradients and write them to .grad
        for group in self.base_optimizer.param_groups:
            for k, p in enumerate(group["params"]):
                summed = None
                for i in range(self.num_tasks):
                    g = grads_per_task[i][k]
                    if g is None:
                        continue
                    if summed is None:
                        summed = g.clone()
                    else:
                        summed = summed + g
                if summed is not None:
                    if p.grad is None:
                        p.grad = summed.to(p.device)
                    else:
                        p.grad.copy_(summed.to(p.device))

        # Clear the buffer for the next iteration
        self._grad_buffer = []

        return self.base_optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
