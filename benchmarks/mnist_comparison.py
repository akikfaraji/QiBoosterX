"""
MNIST benchmark: vanilla Adam vs each booster combination.

Runs on CPU by default. With --subset-size you can run it in 1-2 minutes;
without, the full MNIST takes ~5 minutes per epoch on a 2-core machine.

Output: a table to stdout + a JSON file at benchmarks/results.json.

Examples
--------
    # Fastest: 1000 samples, 1 epoch, all boosters
    python3 benchmarks/mnist_comparison.py --subset-size 1000 --epochs 1

    # Full MNIST, 3 epochs, all boosters (~15-30 min on CPU)
    python3 benchmarks/mnist_comparison.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# Allow running from the repo root without installing
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiboosterx import BoosterConfig, BoosterTrainer


def build_mlp() -> nn.Module:
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


def load_mnist(data_dir: str, batch_size: int, subset_size: int = 0):
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("torchvision not available — using synthetic dataset")
        return _synthetic_loaders(batch_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    except Exception as exc:
        print(f"Could not download MNIST ({exc}); using synthetic dataset.")
        return _synthetic_loaders(batch_size)

    if subset_size > 0:
        train_ds = Subset(train_ds, list(range(min(subset_size, len(train_ds)))))
        test_ds = Subset(test_ds, list(range(min(subset_size // 5, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _synthetic_loaders(batch_size: int):
    torch.manual_seed(0)
    X = torch.randn(2000, 1, 28, 28)
    Y = torch.randint(0, 10, (2000,))
    X_test = torch.randn(400, 1, 28, 28)
    Y_test = torch.randint(0, 10, (400,))
    return (
        DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False),
    )


def run_one(name: str, config: BoosterConfig, train_loader, test_loader, epochs: int, lr: float) -> dict:
    torch.manual_seed(42)  # same init for every run
    model = build_mlp()
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    config.num_epochs = epochs
    trainer = BoosterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        base_optimizer=base_opt,
        config=config,
    )
    t0 = time.time()
    history = trainer.fit()
    elapsed = time.time() - t0
    return {
        "name": name,
        "config": {k: v for k, v in vars(config).items()},
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", type=str, default="/tmp/qiboostx_data")
    p.add_argument("--subset-size", type=int, default=0)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "results.json"))
    p.add_argument("--amp", action="store_true", help="Use mixed precision everywhere it can be used.")
    args = p.parse_args()

    print(f"[benchmark] Loading MNIST (subset={args.subset_size or 'full'})...")
    train_loader, test_loader = load_mnist(args.data_dir, args.batch_size, args.subset_size)
    print(f"[benchmark] train batches={len(train_loader)}, test batches={len(test_loader)}")

    amp = args.amp
    configs = [
        ("baseline_adam", BoosterConfig()),
        ("lookahead", BoosterConfig(use_lookahead=True, use_mixed_precision=amp)),
        ("sam", BoosterConfig(use_sam=True, use_mixed_precision=amp)),
        ("sgld", BoosterConfig(use_sgld=True, use_mixed_precision=amp)),
        ("lookahead+sam", BoosterConfig(use_lookahead=True, use_sam=True, use_mixed_precision=amp)),
        ("lookahead+sgld", BoosterConfig(use_lookahead=True, use_sgld=True, use_mixed_precision=amp)),
        ("lookahead+sam+swa", BoosterConfig(use_lookahead=True, use_sam=True, use_swa=True,
                                            swa_start_epoch=max(0, args.epochs * 3 // 4),
                                            use_mixed_precision=amp)),
    ]

    results = []
    for name, cfg in configs:
        print(f"\n[benchmark] ===== {name} =====")
        try:
            res = run_one(name, cfg, train_loader, test_loader, args.epochs, args.lr)
        except Exception as exc:
            print(f"[benchmark] FAILED: {exc}")
            res = {"name": name, "error": str(exc)}
        results.append(res)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'name':<25}  {'val_acc':>8}  {'val_loss':>10}  {'time':>6}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25}  {'ERROR':>8}  {'—':>10}  {'—':>6}  ({r['error'][:40]})")
        else:
            acc = f"{r['final_val_acc']*100:.2f}%" if r['final_val_acc'] is not None else "—"
            loss = f"{r['final_val_loss']:.4f}" if r['final_val_loss'] is not None else "—"
            t = f"{r['elapsed_seconds']:.1f}s"
            print(f"{r['name']:<25}  {acc:>8}  {loss:>10}  {t:>6}")
    print("=" * 80)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] Results written to {args.out}")


if __name__ == "__main__":
    main()
