"""Digit classifier.

Each digit is an 8 by 8 glyph. Training adds a little noise, then a small
MLP maps the 64 pixels to 10 class logits. The training loop is the same
one used to fit a line; only the model and the data changed.
"""

import numpy as np
import torch
from torch import nn

from foundations.training_loop import evaluate, training_loop
from foundations.weight_init import init_linear_

_GLYPHS = {
    0: ["00111000", "01000100", "01000100", "01000100", "01000100", "01000100", "01000100", "00111000"],
    1: ["00010000", "00110000", "00010000", "00010000", "00010000", "00010000", "00010000", "00111000"],
    2: ["00111000", "01000100", "00000100", "00001000", "00010000", "00100000", "01000000", "01111100"],
    3: ["00111000", "01000100", "00000100", "00011000", "00000100", "00000100", "01000100", "00111000"],
    4: ["00001000", "00011000", "00101000", "01001000", "01111100", "00001000", "00001000", "00001000"],
    5: ["01111100", "01000000", "01000000", "01111000", "00000100", "00000100", "01000100", "00111000"],
    6: ["00111000", "01000000", "01000000", "01111000", "01000100", "01000100", "01000100", "00111000"],
    7: ["01111100", "00000100", "00001000", "00010000", "00100000", "00100000", "00100000", "00100000"],
    8: ["00111000", "01000100", "01000100", "00111000", "01000100", "01000100", "01000100", "00111000"],
    9: ["00111000", "01000100", "01000100", "00111100", "00000100", "00000100", "01000100", "00111000"],
}


def _prototype(digit):
    rows = _GLYPHS[digit]
    pixels = [1.0 if char == "1" else 0.0 for row in rows for char in row]
    return np.array(pixels, dtype=np.float32)


def make_digits(n_per_class=80, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for digit in range(10):
        base = _prototype(digit)
        batch = base + rng.normal(scale=noise, size=(n_per_class, base.size)).astype(np.float32)
        images.append(batch)
        labels.append(np.full(n_per_class, digit, dtype=np.int64))
    return np.vstack(images), np.concatenate(labels)


class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(64, 32)
        self.out = nn.Linear(32, 10)
        init_linear_(self.hidden, "kaiming")
        init_linear_(self.out, "xavier")

    def forward(self, pixels):
        return self.out(torch.relu(self.hidden(pixels)))


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        preds = model(images).argmax(dim=-1)
    return float((preds == labels).float().mean())


def main():
    torch.manual_seed(0)
    train_x, train_y = make_digits(n_per_class=80, noise=0.2, seed=0)
    test_x, test_y = make_digits(n_per_class=30, noise=0.2, seed=1)
    train_images = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y)
    test_images = torch.from_numpy(test_x)
    test_labels = torch.from_numpy(test_y)

    model = DigitMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    history = training_loop(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        [(train_images, train_labels)],
        epochs=25,
    )
    score = accuracy(model, test_images, test_labels)
    held_out_loss = evaluate(model, nn.CrossEntropyLoss(), [(test_images, test_labels)])
    assert history[-1] < history[0]
    assert score > 0.9
    print(f"loss {history[0]:.3f} -> {history[-1]:.3f}")
    print(f"test accuracy {score:.3f}  test loss {held_out_loss:.3f}")


if __name__ == "__main__":
    main()
