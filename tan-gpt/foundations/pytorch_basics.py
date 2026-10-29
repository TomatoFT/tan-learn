"""PyTorch basics: tensors, autograd, and one training step.

A tensor carries numbers and, when asked, the operations that produced it.
loss.backward() fills parameter.grad using those recorded operations.
The optimizer then applies the same update as gradient descent.
"""

import torch
from torch import nn


def autograd_square(x_value):
    x = torch.tensor(x_value, requires_grad=True)
    y = x ** 2
    y.backward()
    return y.item(), x.grad.item()


def linear_matches_manual():
    layer = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    manual = x @ layer.weight.T + layer.bias
    return torch.allclose(layer(x), manual)


def one_step_decreases_loss():
    torch.manual_seed(0)
    x = torch.randn(64, 3)
    true_w = torch.tensor([1.5, -2.0, 0.5])
    y = x @ true_w + 0.25
    layer = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    before = loss_fn(layer(x), y.unsqueeze(1)).item()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(layer(x), y.unsqueeze(1))
    loss.backward()
    optimizer.step()
    after = loss_fn(layer(x), y.unsqueeze(1)).item()
    return before, after


def main():
    value, grad = autograd_square(4.0)
    assert value == 16.0
    assert grad == 8.0
    assert linear_matches_manual()
    before, after = one_step_decreases_loss()
    assert after < before
    print(f"d/dx x^2 at 4 is {grad}")
    print(f"one SGD step  {before:.4f} -> {after:.4f}")


if __name__ == "__main__":
    main()
