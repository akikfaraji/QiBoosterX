"""
Minimal end-to-end example: train a small MLP on a synthetic classification
problem using every QiBoosterX booster in combination.

Runs in ~10 seconds on CPU. No MNIST download required.

    python3 examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qiboosterx import BoosterConfig, BoosterTrainer


def make_synthetic_data(n_train=800, n_val=200, n_classes=4, dim=32):
    """Synthetic multi-class classification problem."""
    torch.manual_seed(0)
    centers = torch.randn(n_classes, dim) * 3.0
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i in range(n_train):
        c = torch.randint(0, n_classes, (1,)).item()
        X_train.append(centers[c] + torch.randn(dim) * 0.7)
        Y_train.append(c)
    for i in range(n_val):
        c = torch.randint(0, n_classes, (1,)).item()
        X_val.append(centers[c] + torch.randn(dim) * 0.7)
        Y_val.append(c)
    return (
        DataLoader(TensorDataset(torch.stack(X_train), torch.tensor(Y_train)),
                  batch_size=32, shuffle=True),
        DataLoader(TensorDataset(torch.stack(X_val), torch.tensor(Y_val)),
                   batch_size=32, shuffle=False),
    )


def build_model():
    return nn.Sequential(
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 4),
    )


def main():
    train_loader, val_loader = make_synthetic_data()
    print(f"[example] train batches={len(train_loader)}, val batches={len(val_loader)}")

    # Full booster stack
    config = BoosterConfig(
        num_epochs=15,
        use_lookahead=True,
        use_sam=True,
        sam_rho=0.1,
        use_sgld=False,
        use_swa=True,
        swa_start_epoch=11,
        use_mixed_precision=True,
        grad_clip_norm=1.0,
        scheduler="cosine",
        log_every_n_steps=20,
    )

    model = build_model()
    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = BoosterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        base_optimizer=base_opt,
        config=config,
    )

    history = trainer.fit()
    print(f"\n[example] final val_acc = {history['val_acc'][-1] * 100:.2f}%")
    print(f"[example] final val_loss = {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
