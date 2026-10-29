"""Training diagnostics.

Loss alone hides a bad step size. The gradient norm and the parameter norm
say whether updates are exploding, vanishing, or still moving.
"""

import math

import torch
from torch import nn


def gradient_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def parameter_norm(module):
    total = 0.0
    for param in module.parameters():
        total += float(param.detach().float().pow(2).sum())
    return math.sqrt(total)


def learning_rate(optimizer):
    rates = [group["lr"] for group in optimizer.param_groups]
    return rates[0] if rates else 0.0


def flag_issues(grad_norm, loss_history, grad_limit=100.0, vanish_limit=1e-8):
    """Return plain-language warnings for this step."""
    issues = []
    if not math.isfinite(grad_norm) or grad_norm > grad_limit:
        issues.append("exploding gradients")
    elif grad_norm < vanish_limit:
        issues.append("vanishing gradients")
    if len(loss_history) >= 4 and all(
        later > earlier for earlier, later in zip(loss_history[-4:], loss_history[-3:])
    ):
        issues.append("loss rising")
    if loss_history and not math.isfinite(loss_history[-1]):
        issues.append("non-finite loss")
    return issues


def main():
    holder = nn.Linear(4, 1, bias=False)
    holder.weight.grad = torch.full_like(holder.weight, 80.0)
    assert "exploding gradients" in flag_issues(gradient_norm(holder), [])

    holder.weight.grad = torch.full_like(holder.weight, 1e-12)
    assert "vanishing gradients" in flag_issues(gradient_norm(holder), [])
    assert "loss rising" in flag_issues(1.0, [1.0, 1.2, 1.4, 1.6])
    assert flag_issues(1.0, [5.0, 4.0, 3.0, 2.0]) == []

    torch.manual_seed(0)
    features = torch.randn(32, 4)
    targets = torch.randn(32, 1)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(features) - targets).pow(2).mean()
        loss.backward()
        issues = flag_issues(gradient_norm(model), losses)
        assert issues == []
        assert parameter_norm(model) > 0
        assert learning_rate(optimizer) == 1e-2
        optimizer.step()
        losses.append(float(loss.detach()))
    print("stable run issues: none")
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
