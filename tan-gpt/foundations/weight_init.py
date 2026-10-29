"""Weight initialization.

The scale of the random weights decides whether a deep stack of layers
starts with healthy activations.

- Zeros leave every neuron identical, so they can never learn different features.
- Xavier (Glorot) keeps the variance stable for linear and tanh layers.
- Kaiming (He) scales up by sqrt(2) to make up for the units ReLU zeroes out.

Weights here use the layout (fan_in, fan_out), matching y = x @ W.
"""

import math

import numpy as np


def zeros(fan_in, fan_out):
    return np.zeros((fan_in, fan_out))


def xavier_normal(fan_in, fan_out, rng):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def xavier_uniform(fan_in, fan_out, rng):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def kaiming_normal(fan_in, fan_out, rng, nonlinearity="relu"):
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / math.sqrt(fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def init_linear_(linear, kind="kaiming"):
    """Initialize a torch.nn.Linear module in place.

    PyTorch stores weights as (fan_out, fan_in).
    """
    import torch

    fan_in = linear.weight.shape[1]
    fan_out = linear.weight.shape[0]
    rng = np.random.default_rng(torch.initial_seed() % (2**32))
    if kind == "kaiming":
        weight = kaiming_normal(fan_in, fan_out, rng)
    elif kind == "xavier":
        weight = xavier_normal(fan_in, fan_out, rng)
    elif kind == "zeros":
        weight = zeros(fan_in, fan_out)
    else:
        raise ValueError(f"unknown init {kind}")
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(weight.T, dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


def _relu_stack_std(init_fn, layers=12, width=128, batch=400, seed=0):
    rng = np.random.default_rng(seed)
    activation = rng.normal(size=(batch, width))
    preactivation = activation
    for _ in range(layers):
        weight = init_fn(width, width, rng)
        preactivation = activation @ weight
        activation = np.maximum(preactivation, 0.0)
    return float(preactivation.std())


def main():
    rng = np.random.default_rng(0)
    fan_in, fan_out = 64, 32
    xavier = xavier_normal(fan_in, fan_out, rng)
    he = kaiming_normal(fan_in, fan_out, rng)
    # Var(x @ W) ≈ fan_in * var(W) when x has unit variance.
    assert abs(fan_in * xavier.var() - 2 * fan_in / (fan_in + fan_out)) < 0.15
    assert abs(fan_in * he.var() - 2.0) < 0.3
    assert np.all(zeros(3, 3) == 0)

    he_std = _relu_stack_std(kaiming_normal)
    xavier_std = _relu_stack_std(xavier_normal)
    zero_std = _relu_stack_std(lambda n_in, n_out, _rng: zeros(n_in, n_out))
    assert he_std > 0.3
    assert xavier_std < 0.05
    assert zero_std == 0.0
    print(f"final preactivation std  he={he_std:.3f}  xavier={xavier_std:.3f}  zeros={zero_std:.3f}")


if __name__ == "__main__":
    main()
