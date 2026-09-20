"""
QiBoosterX command-line interface.

Trains a small MLP on MNIST using a configurable combination of boosters.
Useful for: (1) verifying the library installs and works, (2) running
quick benchmarks to compare booster combinations.

Examples
--------
    # Vanilla baseline (no boosters)
    qiboostx --epochs 2 --no-boost

    # Lookahead + mixed precision
    qiboostx --epochs 2 --lookahead --amp

    # Full stack: Lookahead + SAM + mixed precision + gradient accumulation
    qiboostx --epochs 2 --lookahead --sam --amp --grad-accum-target 64 --grad-accum-micro 8

    # Print all options
    qiboostx --help
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .trainer import BoosterConfig, BoosterTrainer


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qiboostx",
        description="QiBoosterX CLI — train a small MLP on MNIST with configurable boosters.",
    )
    p.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--data-dir", type=str, default="/tmp/qiboostx_data",
                   help="Where to cache MNIST (default: /tmp/qiboostx_data)")
    p.add_argument("--subset-size", type=int, default=0,
                   help="Use only N training samples (0 = full MNIST). "
                        "Useful for quick tests on slow machines.")

    # Boosters (all default OFF — the "no-boost" run is the baseline)
    p.add_argument("--no-boost", action="store_true",
                   help="Disable all boosters (vanilla Adam baseline).")
    p.add_argument("--lookahead", action="store_true", help="Enable Lookahead (k=5, alpha=0.5).")
    p.add_argument("--sam", action="store_true", help="Enable SAM (rho=0.05).")
    p.add_argument("--sgld", action="store_true", help="Enable SGLD noise injection.")
    p.add_argument("--swa", action="store_true", help="Enable SWA (starts at 75%% of epochs).")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision (bfloat16 on CPU, float16 on CUDA).")
    p.add_argument("--grad-accum-target", type=int, default=0,
                   help="Target effective batch size for gradient accumulation (0 = off).")
    p.add_argument("--grad-accum-micro", type=int, default=8,
                   help="Micro batch size for gradient accumulation.")
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "onecycle", "cosine", "cosine_wr", "steplr", "swa"],
                   help="LR scheduler (default: none).")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Gradient clip norm (0 = off).")

    p.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available.")
    return p


def _build_mlp() -> nn.Module:
    """A small MLP that runs in seconds even on a 2-core CPU."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )


def _load_mnist(data_dir: str, batch_size: int, subset_size: int = 0):
    """Load MNIST. Falls back to a synthetic dataset if torchvision is unavailable."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(
            "ERROR: torchvision is required for the MNIST CLI demo.\n"
            "Install it with: pip install torchvision\n"
            "Or use the library API directly with your own data."
        ) from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    except Exception as exc:
        print(f"[qiboostx] Could not download MNIST ({exc}); using synthetic dataset.")
        return _synthetic_loaders(batch_size)

    if subset_size > 0:
        train_ds = Subset(train_ds, list(range(min(subset_size, len(train_ds)))))
        test_ds = Subset(test_ds, list(range(min(subset_size // 5, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _synthetic_loaders(batch_size: int):
    """Tiny synthetic dataset for when MNIST download fails (offline environments)."""
    print("[qiboostx] Using synthetic dataset (random tensors + labels).")
    torch.manual_seed(0)
    X = torch.randn(2000, 1, 28, 28)
    Y = torch.randint(0, 10, (2000,))
    X_test = torch.randn(400, 1, 28, 28)
    Y_test = torch.randint(0, 10, (400,))
    from torch.utils.data import TensorDataset
    return (
        DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False),
    )


def main(argv: Optional[list] = None) -> int:
    args = _build_argparser().parse_args(argv)

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[qiboostx] device = {device}")

    train_loader, test_loader = _load_mnist(args.data_dir, args.batch_size, args.subset_size)
    print(f"[qiboostx] train batches = {len(train_loader)}, test batches = {len(test_loader)}")

    model = _build_mlp()
    base_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Build config from CLI flags
    cfg = BoosterConfig(
        num_epochs=args.epochs,
        use_lookahead=args.lookahead and not args.no_boost,
        use_sam=args.sam and not args.no_boost,
        use_sgld=args.sgld and not args.no_boost,
        use_swa=args.swa and not args.no_boost,
        use_mixed_precision=args.amp and not args.no_boost,
        grad_accum_target=args.grad_accum_target,
        grad_accum_micro=args.grad_accum_micro,
        scheduler=args.scheduler if args.scheduler != "none" else "none",
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
        swa_start_epoch=max(0, args.epochs * 3 // 4),
        device=device,
    )

    # Print effective config
    print(f"[qiboostx] config: "
          f"lookahead={cfg.use_lookahead}  sam={cfg.use_sam}  sgld={cfg.use_sgld}  "
          f"swa={cfg.use_swa}  amp={cfg.use_mixed_precision}  "
          f"grad_accum={cfg.grad_accum_target or 'off'}  scheduler={cfg.scheduler}")

    trainer = BoosterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        base_optimizer=base_opt,
        config=cfg,
    )

    history = trainer.fit()
    print("\n[qiboostx] training complete.")
    if history["val_acc"]:
        print(f"[qiboostx] final val_acc = {history['val_acc'][-1] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
