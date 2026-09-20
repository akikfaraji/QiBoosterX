"""
QiBoosterX — Genuine training boosters for low-end devices.

Components (v0.3.0)
-------------------
Core optimizers:
- Lookahead: wraps any optimizer, +0.5-2% accuracy, smoother loss
- SAM (Sharpness-Aware Minimization): finds flatter minima, +0.5-1.5% accuracy
- SGLD: decaying noise injection (the *correct* "quantum-inspired" booster)
- AdaBelief: drop-in Adam replacement with better generalization
- Lion: Google's sign-momentum optimizer (memory-efficient, 10x smaller lr)
- LARS: layer-wise adaptive rate scaling for large-batch training

Memory / speed helpers:
- GradientAccumulator: simulate large batches on low-memory devices
- MixedPrecision: torch.amp wrapper for 1.5-2x speedup
- EMA: exponential moving average of parameters (gold standard for image gen)

Multi-task / advanced:
- PCGrad: projecting conflicting gradients for multi-task learning
- GradientNoiseAdder: decaying gradient noise (Neelakantan et al. 2015)
- SWA: Stochastic Weight Averaging for +0.5-2% accuracy in late training

End-to-end:
- BoosterConfig: dataclass configuring all boosters
- BoosterTrainer: training loop that wires them together

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
- AdaBelief: Zhuang et al., NeurIPS 2020. https://arxiv.org/abs/2010.07468
- Lion: Chen et al., 2023. https://arxiv.org/abs/2302.06675
- LARS: You et al., 2017. https://arxiv.org/abs/1708.03888
- PCGrad: Yu et al., NeurIPS 2020. https://arxiv.org/abs/2001.06782
- Gradient noise: Neelakantan et al., 2015. https://arxiv.org/abs/1511.06807
- SWA: Izmailov et al., UAI 2018. https://arxiv.org/abs/1803.05407
- EMA: Polyak & Juditsky, SIAM 1992. https://epubs.siam.org/doi/10.1137/0330046
"""

# Core boosters (v0.2.0)
from .lookahead import Lookahead
from .sam import SAM
from .sgld import SGLD
from .gradient_accumulation import GradientAccumulator
from .mixed_precision import MixedPrecision
from .swa import SWA
from .schedulers import build_scheduler

# New in v0.3.0
from .adabelief import AdaBelief
from .lion import Lion
from .lars import LARS
from .ema import EMA
from .pcgrad import PCGrad
from .grad_noise import GradientNoiseAdder

# End-to-end trainer
from .trainer import BoosterConfig, BoosterTrainer

__version__ = "0.3.0"
__author__ = "Akik Forazi"

__all__ = [
    # v0.2.0
    "Lookahead",
    "SAM",
    "SGLD",
    "GradientAccumulator",
    "MixedPrecision",
    "SWA",
    "build_scheduler",
    # v0.3.0
    "AdaBelief",
    "Lion",
    "LARS",
    "EMA",
    "PCGrad",
    "GradientNoiseAdder",
    # End-to-end
    "BoosterConfig",
    "BoosterTrainer",
]
