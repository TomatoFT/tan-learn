"""Cross-entropy loss.

For one example with logits z and true class y:

    L = -log softmax(z)_y

Averaged over a batch, this is the loss a classifier and a language model
both minimize. Computing it from log-softmax avoids taking log(0).
"""

import numpy as np

from foundations.softmax import log_softmax


def cross_entropy(logits, targets):
    """Mean negative log-likelihood of integer class labels.

    logits: (batch, classes)
    targets: (batch,) integer class ids
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    if targets.shape != (logits.shape[0],):
        raise ValueError("targets must have shape (batch,)")
    log_probs = log_softmax(logits, axis=-1)
    picked = log_probs[np.arange(logits.shape[0]), targets]
    return -float(np.mean(picked))


def main():
    classes = 4
    batch = 6
    perfect = np.full((batch, classes), -20.0)
    targets = np.arange(batch) % classes
    perfect[np.arange(batch), targets] = 20.0
    assert cross_entropy(perfect, targets) < 1e-6

    uniform = np.zeros((batch, classes))
    np.testing.assert_allclose(cross_entropy(uniform, targets), np.log(classes))

    # Shifting every logit by the same amount does not change the loss.
    logits = np.array([[2.0, 0.5, -1.0], [0.0, 3.0, 1.0]])
    labels = np.array([0, 1])
    np.testing.assert_allclose(
        cross_entropy(logits, labels),
        cross_entropy(logits + 5.0, labels),
    )
    try:
        cross_entropy(logits, np.array([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched targets should fail")
    print(f"uniform {classes}-way loss {cross_entropy(uniform, targets):.4f}")


if __name__ == "__main__":
    main()
