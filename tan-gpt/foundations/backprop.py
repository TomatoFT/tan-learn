"""Backpropagation through one neuron.

The forward pass is

    z = w · x + b
    a = sigmoid(z)
    L = -(y log a + (1 - y) log(1 - a))

The chain rule collapses to a simple residual. With a = sigmoid(z):

    dL/dz = a - y
    dL/dw = (a - y) x
    dL/db = a - y

That residual is what deeper networks pass from layer to layer.
"""

import numpy as np

from foundations.activations import sigmoid


def forward(x, w, b):
    z = x @ w + b
    a = sigmoid(z)
    return a, z


def binary_cross_entropy(a, y, eps=1e-12):
    a = np.clip(a, eps, 1.0 - eps)
    return -float(np.mean(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)))


def backward(x, y, w, b):
    a, _z = forward(x, w, b)
    residual = (a - y) / len(y)
    dw = x.T @ residual
    db = float(np.sum(residual))
    loss = binary_cross_entropy(a, y)
    return loss, dw, db


def _numerical_grads(x, y, w, b, eps=1e-6):
    base_w = w.copy()
    dw = np.empty_like(w)
    for i in range(w.size):
        w[i] = base_w[i] + eps
        loss_plus = binary_cross_entropy(forward(x, w, b)[0], y)
        w[i] = base_w[i] - eps
        loss_minus = binary_cross_entropy(forward(x, w, b)[0], y)
        w[i] = base_w[i]
        dw[i] = (loss_plus - loss_minus) / (2 * eps)
    loss_plus = binary_cross_entropy(forward(x, w, b + eps)[0], y)
    loss_minus = binary_cross_entropy(forward(x, w, b - eps)[0], y)
    db = (loss_plus - loss_minus) / (2 * eps)
    return dw, db


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4))
    true_w = np.array([0.4, -0.7, 0.2, 0.1])
    true_b = -0.2
    probs = sigmoid(x @ true_w + true_b)
    y = (probs > 0.5).astype(float)

    w = rng.normal(scale=0.1, size=4)
    b = 0.0
    _loss, dw, db = backward(x, y, w, b)
    dw_num, db_num = _numerical_grads(x, y, w, b)
    np.testing.assert_allclose(dw, dw_num, atol=1e-6)
    np.testing.assert_allclose(db, db_num, atol=1e-6)
    grad_error = float(np.max(np.abs(dw - dw_num)))

    losses = []
    for _ in range(200):
        loss, dw, db = backward(x, y, w, b)
        w -= 0.5 * dw
        b -= 0.5 * db
        losses.append(loss)
    assert losses[-1] < losses[0]
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"grad error {grad_error:.2e}")


if __name__ == "__main__":
    main()
