"""Softmax turns scores into a probability distribution.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

Subtracting the max score before the exponential does not change the result
and keeps the values from overflowing.
"""

import numpy as np


def softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def main():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert probs.shape == logits.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    assert np.all(probs > 0)
    # Adding a constant to every score leaves the distribution unchanged.
    np.testing.assert_allclose(softmax(logits), softmax(logits + 10.0))
    uniform = softmax(np.zeros(5))
    np.testing.assert_allclose(uniform, np.full(5, 0.2))
    # A large gap concentrates almost all of the mass on the winning class.
    peaked = softmax(np.array([0.0, 0.0, 40.0]))
    assert peaked.argmax() == 2
    assert peaked[2] > 0.999
    np.testing.assert_allclose(np.log(probs), log_softmax(logits), atol=1e-6)
    # Huge scores must stay finite because of the max subtraction.
    huge = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(huge).all()
    print("softmax rows", np.round(probs, 4))


if __name__ == "__main__":
    main()
