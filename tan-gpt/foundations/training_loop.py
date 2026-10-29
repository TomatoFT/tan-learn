"""A reusable training loop.

Every supervised model in this project, from a line fit to GPT, repeats
the same four lines:

    prediction = model(batch)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

train_batches must be re-iterable, because each epoch walks it again.
"""

import torch
from torch import nn


def training_loop(model, loss_fn, optimizer, train_batches, epochs=1):
    history = []
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for features, targets in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        history.append(total / max(count, 1))
    return history


@torch.no_grad()
def evaluate(model, loss_fn, batches):
    model.eval()
    total = 0.0
    count = 0
    for features, targets in batches:
        total += float(loss_fn(model(features), targets))
        count += 1
    return total / max(count, 1)


def main():
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 40).unsqueeze(1)
    y = 2 * x + 1
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = training_loop(model, nn.MSELoss(), optimizer, [(x, y)], epochs=40)
    assert history[-1] < history[0] * 0.01
    weight = model.weight.detach().item()
    bias = model.bias.detach().item()
    assert abs(weight - 2.0) < 0.05
    assert abs(bias - 1.0) < 0.05
    held_out = evaluate(model, nn.MSELoss(), [(x, y)])
    assert held_out < 1e-3
    print(f"loss {history[0]:.4f} -> {history[-1]:.6f}")
    print(f"learned y = {weight:.3f} x + {bias:.3f}")


if __name__ == "__main__":
    main()
