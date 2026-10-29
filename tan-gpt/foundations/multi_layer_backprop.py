"""Backpropagation through two layers.

Forward pass, with a ReLU hidden layer and a linear output:

    z1 = x W1 + b1
    a1 = relu(z1)
    z2 = a1 W2 + b2

Loss, half the mean squared error over every element of the prediction.
M is the number of elements in z2, not just the batch size:

    L = 1/(2M) * sum (z2 - y)^2

Backward pass, applying the chain rule one layer at a time:

    dL/dz2 = (z2 - y) / M
    dL/dW2 = a1^T dL/dz2
    dL/da1 = dL/dz2 W2^T
    dL/dz1 = dL/da1 * (z1 > 0)
    dL/dW1 = x^T dL/dz1
"""

import numpy as np

from foundations.activations import relu


def forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    return z2, z1, a1


def loss(z2, y):
    return 0.5 * float(np.mean(np.square(z2 - y)))


def backward(x, y, w1, b1, w2, b2):
    z2, z1, a1 = forward(x, w1, b1, w2, b2)
    # np.mean divides by every element, and the 1/2 in the loss cancels the square's 2.
    dz2 = (z2 - y) / z2.size
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ w2.T) * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss(z2, y), (dw1, db1, dw2, db2)


def _flat_grads(x, y, params, eps=1e-6):
    grads = []
    for index, param in enumerate(params):
        grad = np.empty_like(param)
        for coordinate in np.ndindex(param.shape):
            original = param[coordinate]
            param[coordinate] = original + eps
            loss_plus = loss(forward(x, *params)[0], y)
            param[coordinate] = original - eps
            loss_minus = loss(forward(x, *params)[0], y)
            param[coordinate] = original
            grad[coordinate] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(grad)
    return grads


def main():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3))
    w1 = rng.normal(scale=0.4, size=(3, 5))
    b1 = rng.normal(scale=0.1, size=5)
    w2 = rng.normal(scale=0.4, size=(5, 2))
    b2 = rng.normal(scale=0.1, size=2)
    y = rng.normal(size=(20, 2))

    value, grads = backward(x, y, w1, b1, w2, b2)
    numerical = _flat_grads(x, y, [w1, b1, w2, b2])
    for analytic, approx in zip(grads, numerical):
        np.testing.assert_allclose(analytic, approx, atol=1e-5, rtol=1e-4)
    assert value > 0
    print(f"two-layer loss {value:.4f}")
    print(f"max grad error {max(np.max(np.abs(a - n)) for a, n in zip(grads, numerical)):.2e}")


if __name__ == "__main__":
    main()
