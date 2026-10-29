"""Train linear regression with gradient descent.

The loss is half the mean squared error,

    L = 1/(2N) * sum_i (y_hat_i - y_i)^2

so the gradients are

    dL/dw = X^T (y_hat - y) / N
    dL/db = mean(y_hat - y)

Gradient descent on this surface recovers the least-squares line.
"""

import numpy as np

from foundations.gradient_descent import gradient_descent
from foundations.linear_regression import forward


def pack(w, b):
    return np.concatenate([np.asarray(w, dtype=float), np.atleast_1d(float(b))])


def unpack(theta, n_features):
    return theta[:n_features], float(theta[n_features])


def mean_squared_error(X, y, w, b):
    error = forward(X, w, b) - y
    return 0.5 * float(np.mean(np.square(error)))


def mse_gradient(X, y, theta):
    n_features = X.shape[1]
    w, b = unpack(theta, n_features)
    error = forward(X, w, b) - y
    dw = (X.T @ error) / len(y)
    db = float(np.mean(error))
    return pack(dw, db)


def train(X, y, lr=0.1, steps=400):
    theta0 = np.zeros(X.shape[1] + 1)
    theta, history = gradient_descent(
        lambda current: mse_gradient(X, y, current),
        theta0,
        lr=lr,
        steps=steps,
    )
    w, b = unpack(theta, X.shape[1])
    return w, b, history


def least_squares(X, y):
    design = np.column_stack([X, np.ones(len(X))])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return theta[:-1], float(theta[-1])


def main():
    rng = np.random.default_rng(0)
    n = 256
    X = rng.normal(size=(n, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    true_b = -0.3
    y = forward(X, true_w, true_b) + rng.normal(scale=0.05, size=n)

    w, b, history = train(X, y, lr=0.2, steps=500)
    w_ls, b_ls = least_squares(X, y)
    np.testing.assert_allclose(w, w_ls, atol=1e-4)
    np.testing.assert_allclose(b, b_ls, atol=1e-4)
    np.testing.assert_allclose(w, true_w, atol=0.05)
    assert abs(b - true_b) < 0.05

    start = unpack(history[0], X.shape[1])
    end_loss = mean_squared_error(X, y, w, b)
    start_loss = mean_squared_error(X, y, *start)
    assert end_loss < start_loss
    print(f"learned w={w}, b={b:.3f}")
    print(f"loss {start_loss:.4f} -> {end_loss:.4f}")


if __name__ == "__main__":
    main()
