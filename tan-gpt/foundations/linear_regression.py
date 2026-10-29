"""Linear regression forward pass.

A linear model predicts a scalar from features with one weight per feature
and a bias:

    y_hat = X w + b

X has shape (examples, features). w has shape (features,). b is a scalar.
"""

import numpy as np


def forward(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (examples, features)")
    if w.shape != (X.shape[1],):
        raise ValueError("w must have shape (features,)")
    return X @ w + float(b)


def main():
    X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]])
    w = np.array([1.5, -2.0])
    b = 0.25
    y_hat = forward(X, w, b)
    expected = np.array([1.5 - 4.0 + 0.25, 2.0 + 0.25, 4.5 - 1.0 + 0.25])
    np.testing.assert_allclose(y_hat, expected)
    assert y_hat.shape == (3,)
    print("predictions", y_hat)


if __name__ == "__main__":
    main()
