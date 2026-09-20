"""Smoke test: import everything and run a single forward-backward pass."""

import torch
import torch.nn as nn

from qiboosterx import (
    BoosterConfig,
    BoosterTrainer,
    GradientAccumulator,
    Lookahead,
    MixedPrecision,
    SAM,
    SGLD,
    SWA,
    build_scheduler,
)


def test_imports() -> None:
    assert Lookahead is not None
    assert SAM is not None
    assert SGLD is not None
    assert SWA is not None
    assert GradientAccumulator is not None
    assert MixedPrecision is not None
    assert BoosterConfig is not None
    assert BoosterTrainer is not None
    assert build_scheduler is not None


def test_lookahead_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    base = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = Lookahead(base, k=2, alpha=0.5)
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    for _ in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 1.5, f"Lookahead didn't converge, loss={loss.item():.3f}"


def test_sam_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    base = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    sam = SAM(model.parameters(), base_optimizer=base, rho=0.05)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    last_loss = None
    for _ in range(50):
        # First pass
        loss1 = nn.functional.cross_entropy(model(x), y)
        loss1.backward()
        sam.first_step(zero_grad=True)
        # Second pass
        loss2 = nn.functional.cross_entropy(model(x), y)
        loss2.backward()
        sam.second_step(zero_grad=True)
        last_loss = loss2
    assert last_loss is not None and last_loss.item() < 1.0, f"SAM didn't converge, loss={last_loss.item():.3f}"


def test_sgld_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    base = torch.optim.SGD(model.parameters(), lr=1e-2)
    sgld = SGLD(base, noise_decay=0.55)
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    for i in range(5):
        sgld.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        sgld.step(step_fraction=i / 5.0)
    assert loss.item() < 1.5, f"SGLD didn't converge, loss={loss.item():.3f}"


def test_gradient_accumulator() -> None:
    accum = GradientAccumulator(target_batch_size=64, micro_batch_size=8)
    assert accum.accumulation_steps == 8
    assert accum.scale_factor == 8.0
    fires = []
    for i in range(16):
        fires.append(accum.step())
    assert fires.count(True) == 2, f"Expected 2 fires, got {fires.count(True)}"


def test_mixed_precision_cpu_bfloat16() -> None:
    amp = MixedPrecision(dtype="bfloat16", device="cpu", verbose=False)
    assert amp.is_active
    assert amp.dtype == torch.bfloat16
    # Run a real autocast forward
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    with amp.autocast():
        out = model(x)
    assert out.dtype == torch.bfloat16


def test_swa_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    swa = SWA(model, swa_start_epoch=2)
    for epoch in range(5):
        # Fake update — pretend we trained
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight) * 0.01)
        swa.update(epoch)
    swa.finalize(dataloader=None)
    assert swa.n_updates == 3


def test_build_scheduler() -> None:
    model = nn.Linear(10, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = build_scheduler("cosine", opt, num_epochs=5)
    assert sched is not None
    for _ in range(5):
        sched.step()


if __name__ == "__main__":
    test_imports()
    test_lookahead_runs()
    test_sam_runs()
    test_sgld_runs()
    test_gradient_accumulator()
    test_mixed_precision_cpu_bfloat16()
    test_swa_runs()
    test_build_scheduler()
    print("All smoke tests passed.")
