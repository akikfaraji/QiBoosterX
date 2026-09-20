"""
QiBoosterX — Genuine training boosters for low-end devices.

Components
----------
- Lookahead: wraps any optimizer, +0.5-2% accuracy, smoother loss
- SAM (Sharpness-Aware Minimization): finds flatter minima, +0.5-1.5% accuracy
- SGLD: decaying noise injection (the *correct* "quantum-inspired" booster)
- GradientAccumulator: simulate large batches on low-memory devices
- MixedPrecision: torch.amp wrapper for 1.5-2x speedup
- SWA: Stochastic Weight Averaging for +0.5-2% accuracy in late training
- BoosterTrainer: end-to-end training loop with any combination of the above

Quick start
-----------
    from qiboosterx import BoosterConfig, BoosterTrainer, Lookahead
    import torch, torch.nn as nn

    model = MyModel()
    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    booster_opt = Lookahead(base_opt, k=5, alpha=0.5)

    # ... or use the all-in-one trainer:
    cfg = BoosterConfig(num_epochs=5, use_lookahead=True, use_mixed_precision=True)
    trainer = BoosterTrainer(model, train_loader, val_loader, config=cfg)
    trainer.fit()

References
----------
- Lookahead: Zhang et al., NeurIPS 2019. https://arxiv.org/abs/1907.08610
- SAM: Foret et al., ICLR 2021. https://arxiv.org/abs/2010.01412
- SGLD: Welling & Teh, ICML 2011. https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
- SWA: Izmailov et al., UAI 2018. https://arxiv.org/abs/1803.05407
"""

from .lookahead import Lookahead
from .sam import SAM
from .sgld import SGLD
from .gradient_accumulation import GradientAccumulator
from .mixed_precision import MixedPrecision
from .swa import SWA
from .schedulers import build_scheduler
from .trainer import BoosterConfig, BoosterTrainer

__version__ = "0.2.0"
__author__ = "Akik Forazi"

__all__ = [
    # Core boosters
    "Lookahead",
    "SAM",
    "SGLD",
    "GradientAccumulator",
    "MixedPrecision",
    "SWA",
    "build_scheduler",
    # End-to-end trainer
    "BoosterConfig",
    "BoosterTrainer",
]
