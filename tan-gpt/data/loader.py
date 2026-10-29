"""Batched training data.

Each example is a context window x and the same window shifted by one
character, y. The model reads x and is scored on predicting y.
"""

import torch


def get_batch(data, block_size, batch_size, device="cpu"):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if data.numel() <= block_size:
        raise ValueError("data must be longer than block_size")
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    contexts = torch.stack([data[start:start + block_size] for start in starts])
    targets = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return contexts.to(device), targets.to(device)


def main():
    torch.manual_seed(0)
    data = torch.arange(100)
    contexts, targets = get_batch(data, block_size=8, batch_size=4)
    assert contexts.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert torch.equal(contexts[:, 1:], targets[:, :-1])
    # The last target of each row is the token just after the context.
    for row in range(4):
        start = int(contexts[row, 0])
        assert int(targets[row, -1]) == start + 8
    try:
        get_batch(torch.arange(4), block_size=4, batch_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("a short corpus should fail")
    print("batch x", contexts[0].tolist())
    print("batch y", targets[0].tolist())


if __name__ == "__main__":
    main()
