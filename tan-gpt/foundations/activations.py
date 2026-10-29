"""Activation functions and their derivatives.

A neuron computes a linear score z = w·x + b and then squashes it.
The derivative is the factor backpropagation carries to earlier layers.
"""

import numpy as np


def sigmoid(x):
    # Stable for large positive and negative inputs.
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (np.asarray(x) > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    t = np.tanh(x)
    return 1.0 - t * t


def gelu(x):
    """GELU with the tanh approximation used by GPT-2."""
    x = np.asarray(x, dtype=float)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x):
    x = np.asarray(x, dtype=float)
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * np.power(x, 3))
    tanh_u = np.tanh(u)
    sech2 = 1.0 - tanh_u * tanh_u
    du = c * (1.0 + 3.0 * 0.044715 * np.square(x))
    return 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du


def silu(x):
    """SiLU / swish: x * sigmoid(x)."""
    x = np.asarray(x, dtype=float)
    return x * sigmoid(x)


def silu_grad(x):
    x = np.asarray(x, dtype=float)
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _check_grad(fn, grad_fn, x, atol=1e-5):
    numerical = np.empty_like(x, dtype=float)
    eps = 1e-6
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical[index] = (fn(plus) - fn(minus)) / (2 * eps)
    np.testing.assert_allclose(grad_fn(x), numerical, atol=atol, rtol=1e-4)


def main():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.all(relu(x) >= 0)
    assert relu(np.array(-3.0)) == 0.0
    assert np.allclose(sigmoid(0.0), 0.5)
    assert np.all((sigmoid(x) > 0) & (sigmoid(x) < 1))
    assert np.allclose(tanh(0.0), 0.0)
    # GELU and SiLU are smooth ReLU-shaped curves: negative inputs are near 0,
    # positive inputs stay close to the identity.
    assert gelu(np.array(-8.0)) < 0.01
    assert abs(float(gelu(np.array(8.0))) - 8.0) < 0.05
    assert silu(np.array(-8.0)) > -0.01
    samples = np.linspace(-2.0, 2.0, 8)
    for fn, grad_fn in (
        (sigmoid, sigmoid_grad),
        (relu, relu_grad),
        (tanh, tanh_grad),
        (gelu, gelu_grad),
        (silu, silu_grad),
    ):
        _check_grad(lambda value, fn=fn: fn(value).sum(), grad_fn, samples)
    print("activation derivatives match numerical gradients")


if __name__ == "__main__":
    main()
