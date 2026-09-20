"""Smoke test: import everything and run a single forward-backward pass."""

import torch
import torch.nn as nn

from qiboosterx import (
    AdaBelief,
    BoosterConfig,
    BoosterTrainer,
    EMA,
    GradientAccumulator,
    GradientNoiseAdder,
    LARS,
    Lion,
    Lookahead,
    MixedPrecision,
    PCGrad,
    SAM,
    SGLD,
    SWA,
    build_scheduler,
)


def test_imports() -> None:
    for cls in [Lookahead, SAM, SGLD, SWA, GradientAccumulator, MixedPrecision,
                AdaBelief, Lion, LARS, EMA, PCGrad, GradientNoiseAdder,
                BoosterConfig, BoosterTrainer, build_scheduler]:
        assert cls is not None


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
        loss1 = nn.functional.cross_entropy(model(x), y)
        loss1.backward()
        sam.first_step(zero_grad=True)
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


def test_adabelief_runs() -> None:
    """AdaBelief should converge at least as well as Adam."""
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    opt = AdaBelief(model.parameters(), lr=1e-2)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    last_loss = None
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        last_loss = loss
    assert last_loss is not None and last_loss.item() < 0.5, f"AdaBelief didn't converge, loss={last_loss.item():.3f}"


def test_lion_runs() -> None:
    """Lion needs a smaller lr than Adam."""
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    opt = Lion(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    last_loss = None
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        last_loss = loss
    assert last_loss is not None and last_loss.item() < 1.0, f"Lion didn't converge, loss={last_loss.item():.3f}"


def test_lars_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    base = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lars = LARS(base, trust_coefficient=0.001)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    last_loss = None
    for _ in range(50):
        lars.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        lars.step()
        last_loss = loss
    assert last_loss is not None and last_loss.item() < 1.0, f"LARS didn't converge, loss={last_loss.item():.3f}"


def test_ema_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    ema = EMA(model, decay=0.9)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    initial_w = model.weight.data.clone()
    for _ in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        ema.update()
    # EMA weights should differ from current weights (EMA lags behind)
    with ema.swap():
        ema_w = model.weight.data.clone()
    # Restore originals (swap context does this)
    current_w = model.weight.data.clone()
    assert torch.allclose(current_w, initial_w) is False, "Training didn't change weights"
    assert not torch.allclose(ema_w, current_w), "EMA weights should differ from training weights"


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
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    with amp.autocast():
        out = model(x)
    assert out.dtype == torch.bfloat16


def test_gradient_noise_adder() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    base = torch.optim.SGD(model.parameters(), lr=0.1)
    gna = GradientNoiseAdder(base, eta=0.01, gamma=0.55)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    last_loss = None
    for i in range(50):
        gna.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        gna.step(step_number=i)
        last_loss = loss
    assert last_loss is not None and last_loss.item() < 1.0, f"GradientNoiseAdder didn't converge, loss={last_loss.item():.3f}"


def test_pcgrad_runs() -> None:
    """PCGrad projects conflicting gradients in multi-task learning."""
    torch.manual_seed(0)
    # Two-task model: shared encoder + 2 heads
    class TwoTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(10, 20)
            self.head1 = nn.Linear(20, 2)
            self.head2 = nn.Linear(20, 2)
        def forward(self, x):
            h = torch.relu(self.shared(x))
            return self.head1(h), self.head2(h)

    model = TwoTaskModel()
    base = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    pcgrad = PCGrad(base, num_tasks=2)

    x = torch.randn(32, 10)
    y1 = torch.randint(0, 2, (32,))
    y2 = torch.randint(0, 2, (32,))

    # Train for a few steps
    for _ in range(5):
        pcgrad.zero_grad()
        out1, out2 = model(x)
        loss1 = nn.functional.cross_entropy(out1, y1)
        loss2 = nn.functional.cross_entropy(out2, y2)
        loss1.backward(retain_graph=True)
        pcgrad.collect_task_grads(0)
        loss2.backward()
        pcgrad.collect_task_grads(1)
        pcgrad.step()
    assert loss1.item() < 2.0 and loss2.item() < 2.0


def test_swa_runs() -> None:
    torch.manual_seed(0)
    model = nn.Linear(10, 2)
    swa = SWA(model, swa_start_epoch=2)
    for epoch in range(5):
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
    test_adabelief_runs()
    test_lion_runs()
    test_lars_runs()
    test_ema_runs()
    test_gradient_accumulator()
    test_mixed_precision_cpu_bfloat16()
    test_gradient_noise_adder()
    test_pcgrad_runs()
    test_swa_runs()
    test_build_scheduler()
    print("All smoke tests passed (14 tests).")
