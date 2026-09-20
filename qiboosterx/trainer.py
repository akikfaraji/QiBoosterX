"""
Booster-aware Trainer — the new QiBoosterX entry point.

Wraps a standard PyTorch training loop with optional QiBoosterX components
(Lookahead, SAM, SGLD, gradient accumulation, mixed precision, SWA) and
makes them all configurable through a single config object.

Usage:
    from qiboosterx import BoosterConfig, BoosterTrainer

    cfg = BoosterConfig(
        use_lookahead=True,        # wrap optimizer with Lookahead
        use_sam=True,              # use SAM (slower but better generalization)
        use_mixed_precision=True,  # use bfloat16 on CPU / float16 on CUDA
        grad_accum_target=64,      # simulate batch=64 by accumulating
        grad_accum_micro=8,        # actual batch size = 8
        scheduler="onecycle",
        num_epochs=5,
    )

    trainer = BoosterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        base_optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=cfg,
    )
    trainer.fit()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .gradient_accumulation import GradientAccumulator
from .lookahead import Lookahead
from .mixed_precision import MixedPrecision
from .sam import SAM
from .sgld import SGLD
from .swa import SWA


# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #


@dataclass
class BoosterConfig:
    """Configuration for BoosterTrainer. All fields are optional."""

    num_epochs: int = 5

    # Lookahead
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    # SAM
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = False

    # SGLD (decaying noise)
    use_sgld: bool = False
    sgld_noise_decay: float = 0.55
    sgld_noise_floor: float = 1e-5
    sgld_initial_noise_scale: float = 0.1

    # Gradient accumulation
    grad_accum_target: int = 0  # 0 = no accumulation
    grad_accum_micro: int = 1

    # Mixed precision
    use_mixed_precision: bool = False
    amp_dtype: str = "bfloat16"

    # SWA
    use_swa: bool = False
    swa_start_epoch: int = 0
    swa_lr: Optional[float] = None

    # Scheduler
    scheduler: str = "none"  # "none" | "onecycle" | "cosine" | "cosine_wr" | "steplr" | "swa"

    # Gradient clipping
    grad_clip_norm: Optional[float] = None

    # Logging
    log_every_n_steps: int = 50
    verbose: bool = True

    device: str = "auto"

    def __post_init__(self) -> None:
        if self.use_sam and self.use_sgld:
            # Both modify the gradient direction; combining them is non-trivial
            # and not standard practice. Allow it but warn.
            print("[BoosterConfig] Warning: combining SAM + SGLD is non-standard.")
        if self.use_swa and self.scheduler != "swa" and self.num_epochs - self.swa_start_epoch > 0:
            # Using SWA without SWALR scheduler is fine; SWALR is optional.
            pass


# ---------------------------------------------------------------------------- #
# Trainer
# ---------------------------------------------------------------------------- #


class BoosterTrainer:
    """Booster-aware PyTorch training loop.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader, optional
    criterion : callable
    base_optimizer : Optimizer
        The base optimizer (Adam, SGD, AdamW, ...). QiBoosterX boosters wrap this.
    config : BoosterConfig
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None,
        base_optimizer: Optional[Optimizer] = None,
        config: Optional[BoosterConfig] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.config = config or BoosterConfig()

        # Resolve device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Build the optimizer stack:  base  ->  SGLD  ->  Lookahead  ->  SAM
        # (SAM is the outermost because it needs to do two forward passes per step)
        if base_optimizer is None:
            base_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.base_optimizer = base_optimizer

        opt: Optimizer = base_optimizer

        if self.config.use_sgld:
            opt = SGLD(
                opt,
                noise_decay=self.config.sgld_noise_decay,
                noise_floor=self.config.sgld_noise_floor,
                initial_noise_scale=self.config.sgld_initial_noise_scale,
            )

        if self.config.use_lookahead:
            opt = Lookahead(
                opt,
                k=self.config.lookahead_k,
                alpha=self.config.lookahead_alpha,
            )

        # SAM must wrap the outermost optimizer (we use base_optimizer's params)
        if self.config.use_sam:
            opt = SAM(
                self.model.parameters(),
                base_optimizer=opt,  # wrap whatever stack we built so far
                rho=self.config.sam_rho,
                adaptive=self.config.sam_adaptive,
            )

        self.optimizer = opt

        # Gradient accumulator
        self.accum: Optional[GradientAccumulator] = None
        if self.config.grad_accum_target > 0:
            self.accum = GradientAccumulator(
                target_batch_size=self.config.grad_accum_target,
                micro_batch_size=self.config.grad_accum_micro,
            )

        # Mixed precision
        self.amp: Optional[MixedPrecision] = None
        if self.config.use_mixed_precision:
            self.amp = MixedPrecision(
                dtype=self.config.amp_dtype,
                device=str(self.device) if self.device.type == "cuda" else "cpu",
                verbose=self.config.verbose,
            )

        # SWA
        self.swa: Optional[SWA] = None
        if self.config.use_swa:
            self.swa = SWA(
                model=self.model,
                swa_start_epoch=self.config.swa_start_epoch,
                swa_lr=self.config.swa_lr,
                device=str(self.device),
            )

        # Scheduler
        self.scheduler = None
        if self.config.scheduler != "none":
            from .schedulers import build_scheduler
            steps_per_epoch = len(train_loader) if self.accum is None else (
                len(train_loader) // self.accum.accumulation_steps
            )
            self.scheduler = build_scheduler(
                name=self.config.scheduler,
                optimizer=self.optimizer if not self.config.use_sam else self.base_optimizer,
                steps_per_epoch=steps_per_epoch,
                num_epochs=self.config.num_epochs,
            )

        self.history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def fit(self) -> dict:
        """Run training. Returns history dict."""
        cfg = self.config
        steps_per_epoch = len(self.train_loader)
        global_step = 0

        for epoch in range(cfg.num_epochs):
            epoch_loss = 0.0
            t0 = time.time()
            self.model.train()

            for i, batch in enumerate(self.train_loader):
                x, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)

                step_fraction = epoch + i / max(steps_per_epoch, 1)

                if cfg.use_sam:
                    # SAM requires two forward-backward passes. The closure
                    # performs only the forward pass; SAM calls backward().
                    def closure():
                        if self.amp is not None:
                            with self.amp.autocast():
                                out = self.model(x)
                                loss = self.criterion(out, y)
                        else:
                            out = self.model(x)
                            loss = self.criterion(out, y)
                        return loss

                    loss = self.optimizer.first_forward(closure)
                    if self.amp is not None and self.amp.scaler is not None:
                        self.amp.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    loss2 = closure()
                    if self.amp is not None and self.amp.scaler is not None:
                        self.amp.scaler.scale(loss2).backward()
                    else:
                        loss2.backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    # Standard step
                    self.optimizer.zero_grad()
                    if self.amp is not None:
                        with self.amp.autocast():
                            out = self.model(x)
                            loss = self.criterion(out, y)
                        loss_scaled = loss / (self.accum.scale_factor if self.accum else 1.0)
                        self.amp.backward(loss_scaled)
                    else:
                        out = self.model(x)
                        loss = self.criterion(out, y)
                        loss_scaled = loss / (self.accum.scale_factor if self.accum else 1.0)
                        loss_scaled.backward()

                    # Wait to step if accumulating
                    if self.accum is not None:
                        if self.accum.step():
                            self._maybe_clip_grads()
                            if isinstance(self.optimizer, SGLD):
                                self.optimizer.step(step_fraction=step_fraction)
                            else:
                                self.optimizer.step()
                    else:
                        self._maybe_clip_grads()
                        if isinstance(self.optimizer, SGLD):
                            self.optimizer.step(step_fraction=step_fraction)
                        else:
                            self.optimizer.step()

                epoch_loss += float(loss.item())
                global_step += 1

                if cfg.verbose and (global_step % cfg.log_every_n_steps == 0):
                    print(
                        f"  step {global_step:>6}  loss={float(loss.item()):.4f}  "
                        f"lr={self.base_optimizer.param_groups[0]['lr']:.2e}"
                    )

            avg_train_loss = epoch_loss / max(steps_per_epoch, 1)
            self.history["train_loss"].append(avg_train_loss)
            elapsed = time.time() - t0

            # Validation
            val_loss, val_acc = None, None
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            # Scheduler step (most schedulers step per epoch)
            if self.scheduler is not None and not isinstance(self.scheduler, type(None)):
                # OneCycleLR steps per batch, others per epoch
                from torch.optim.lr_scheduler import OneCycleLR
                if not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            # SWA update
            if self.swa is not None:
                self.swa.update(epoch)

            if cfg.verbose:
                msg = f"Epoch {epoch+1:>3}/{cfg.num_epochs}  loss={avg_train_loss:.4f}  ({elapsed:.1f}s)"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
                print(msg)

        # Finalize SWA
        if self.swa is not None:
            self.swa.finalize(self.val_loader)
            # Re-evaluate with SWA weights
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate()
                if cfg.verbose:
                    print(f"[SWA] Final val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        return self.history

    @torch.no_grad()
    def evaluate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch in self.val_loader:
            x, y = self._unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)
            if self.amp is not None:
                with self.amp.autocast():
                    out = self.model(x)
                    loss = self.criterion(out, y)
            else:
                out = self.model(x)
                loss = self.criterion(out, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = out.argmax(dim=1)
            total_correct += int((preds == y).sum().item())
            total_seen += int(y.size(0))
        return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _unpack_batch(self, batch):
        """Handle (x, y) tuples, (x, y, ...) longer tuples, and dict batches."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _maybe_clip_grads(self) -> None:
        if self.config.grad_clip_norm is None:
            return
        if self.amp is not None and self.amp.scaler is not None:
            self.amp.unscale_(self.base_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
