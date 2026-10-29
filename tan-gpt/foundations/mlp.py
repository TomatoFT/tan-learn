"""Multilayer perceptron trained with backpropagation.

Hidden layers use ReLU. The last layer emits logits, and the loss is
cross-entropy. One training step is a forward pass, a backward pass, and
a parameter update:

    W <- W - learning_rate * dL/dW
"""

import numpy as np

from foundations.activations import relu
from foundations.loss import cross_entropy
from foundations.softmax import softmax


class MLP:
    def __init__(self, layer_sizes, seed=0):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs an input size and an output size")
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization keeps ReLU activations from shrinking.
            scale = np.sqrt(2.0 / n_in)
            self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        activations = [np.asarray(x, dtype=float)]
        preacts = []
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ weight + bias
            preacts.append(z)
            activations.append(z if index == last else relu(z))
        return activations[-1], preacts, activations

    def predict(self, x):
        logits, _preacts, _activations = self.forward(x)
        return softmax(logits, axis=-1).argmax(axis=-1)

    def backward(self, x, targets):
        logits, preacts, activations = self.forward(x)
        targets = np.asarray(targets, dtype=int)
        loss = cross_entropy(logits, targets)
        residual = softmax(logits, axis=-1)
        residual[np.arange(len(targets)), targets] -= 1.0
        residual /= len(targets)

        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        for index in reversed(range(len(self.weights))):
            grad_w[index] = activations[index].T @ residual
            grad_b[index] = residual.sum(axis=0)
            residual = residual @ self.weights[index].T
            if index > 0:
                residual = residual * (preacts[index - 1] > 0)
        return loss, grad_w, grad_b

    def step(self, x, targets, lr):
        loss, grad_w, grad_b = self.backward(x, targets)
        for weight, bias, dw, db in zip(self.weights, self.biases, grad_w, grad_b):
            weight -= lr * dw
            bias -= lr * db
        return loss


def _numerical_param_grad(model, x, y, eps=1e-5):
    grads = []
    for param in model.weights + model.biases:
        grad = np.empty_like(param)
        for coordinate in np.ndindex(param.shape):
            original = param[coordinate]
            param[coordinate] = original + eps
            loss_plus = cross_entropy(model.forward(x)[0], y)
            param[coordinate] = original - eps
            loss_minus = cross_entropy(model.forward(x)[0], y)
            param[coordinate] = original
            grad[coordinate] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(grad)
    return grads


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3))
    y = rng.integers(0, 2, size=8)
    model = MLP([3, 4, 2], seed=0)
    _loss, grad_w, grad_b = model.backward(x, y)
    numerical = _numerical_param_grad(model, x, y)
    analytic = grad_w + grad_b
    for got, expected in zip(analytic, numerical):
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-4)

    # XOR needs the hidden layer: the classes are not linearly separable.
    xor_x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_y = np.array([0, 1, 1, 0])
    xor = MLP([2, 8, 2], seed=1)
    losses = [xor.step(xor_x, xor_y, lr=0.5) for _ in range(800)]
    predictions = xor.predict(xor_x)
    assert losses[-1] < losses[0]
    assert np.array_equal(predictions, xor_y)
    print(f"xor loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print("xor predictions", predictions.tolist())


if __name__ == "__main__":
    main()
