"""A single neuron.

The neuron is an affine function followed by an activation:

    z = w · x + b
    a = activation(z)

A layer is many of these neurons side by side, which is the matrix product
in the multilayer perceptron.
"""

import numpy as np

from foundations.activations import relu, sigmoid


def neuron_forward(x, w, b, activation=relu):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.ndim == 1:
        if w.shape != x.shape:
            raise ValueError("w and x must have the same shape")
        z = float(np.dot(x, w) + b)
        return activation(np.array(z)).item(), z
    if w.shape != (x.shape[1],):
        raise ValueError("w must have shape (features,)")
    z = x @ w + float(b)
    return activation(z), z


def main():
    x = np.array([1.0, 2.0, -1.0])
    w = np.array([0.5, -1.0, 0.25])
    b = 0.25
    # z = 0.5 - 2.0 - 0.25 + 0.25 = -1.5
    activation, z = neuron_forward(x, w, b, activation=relu)
    assert abs(z - (-1.5)) < 1e-8
    assert activation == 0.0

    batch = np.array([[1.0, 2.0, -1.0], [0.0, 0.0, 4.0]])
    activations, scores = neuron_forward(batch, w, b, activation=sigmoid)
    assert activations.shape == (2,)
    assert np.all((activations > 0) & (activations < 1))
    np.testing.assert_allclose(scores[0], -1.5)
    print(f"single neuron z={z:.2f} relu={activation}")
    print(f"batch sigmoid={activations}")


if __name__ == "__main__":
    main()
