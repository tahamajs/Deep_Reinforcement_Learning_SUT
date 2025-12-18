import torch
from paperAssignments.Assignments1_50.CA2.src.optim.sophia import Sophia  # type: ignore
from paperAssignments.Assignments1_50.CA2.src.agent import build_optimizer  # type: ignore


def test_sophia_reduces_quadratic_loss():
    # simple quadratic f(x) = 0.5 * a * x^2  -> grad = a * x
    a = 2.0
    x = torch.tensor([5.0], requires_grad=True)

    opt = Sophia([x], lr=1e-2, beta1=0.9, beta2=0.99, gamma=0.1, eps=1e-12, clip=1.0)

    def closure():
        opt.zero_grad()
        loss = 0.5 * a * x.pow(2).sum()
        loss.backward()
        return loss

    initial_loss = float(0.5 * a * x.pow(2).sum())

    # perform a few steps and ensure loss decreases
    for _ in range(10):
        loss = closure()
        opt.step()

    final_loss = float(0.5 * a * x.pow(2).sum().detach())

    assert final_loss < initial_loss, "Sophia did not decrease the quadratic loss"


def test_sophia_state_contains_m_and_h():
    p = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    opt = Sophia([p], lr=1e-3)
    p.grad = torch.tensor([0.5])
    opt.step()
    st = opt.state[p]
    assert "m" in st and "h" in st
    assert st["m"].shape == p.data.shape
    assert st["h"].shape == p.data.shape


def test_hutchinson_step_runs():
    # small linear model
    model = torch.nn.Linear(3, 1)
    opt = build_optimizer(model, optim_name="sophia", lr=1e-2, hessian_estimator="hutchinson", hutchinson_samples=1)

    # random batch
    obs = torch.randn(4, 3)
    target = torch.randn(4, 1)

    def closure():
        opt.zero_grad()
        out = model(obs)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward(create_graph=True)
        return loss

    # Should run without errors and update state
    loss_before = float(torch.nn.functional.mse_loss(model(obs), target).item())
    opt.step(closure)
    loss_after = float(torch.nn.functional.mse_loss(model(obs), target).item())
    assert "m" in opt.state[next(iter(model.parameters()))]

