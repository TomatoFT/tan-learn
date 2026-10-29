"""Gradient descent.

Each step moves the parameters opposite the gradient:

    theta <- theta - learning_rate * dL/dtheta

For a smooth bowl, a small enough step size walks downhill to the minimum.
A step size that is too large overshoots and diverges.
"""

import numpy as np


def gradient_descent(grad_fn, x0, lr, steps):
    """Minimize a function given its gradient.

    grad_fn maps the current parameter array to an array of the same shape.
    Returns the final parameters and the parameter value after every step,
    including the start.
    """
    x = np.array(x0, dtype=float, copy=True)
    history = [x.copy()]
    for _ in range(steps):
        x = x - lr * np.asarray(grad_fn(x), dtype=float)
        history.append(x.copy())
    return x, history


def _parabola_grad(x):
    # f(x) = (x - 3)^2, so f'(x) = 2(x - 3). Minimum is x = 3.
    return 2.0 * (x - 3.0)


def main():
    converged, history = gradient_descent(_parabola_grad, x0=0.0, lr=0.1, steps=40)
    assert abs(float(converged) - 3.0) < 1e-3
    assert abs(float(history[-1]) - 3.0) < abs(float(history[0]) - 3.0)

    # The update is x <- (1 - 2*lr)*x + 6*lr. It contracts only for 0 < lr < 1.
    diverged, div_history = gradient_descent(_parabola_grad, x0=0.0, lr=1.1, steps=8)
    assert abs(float(diverged)) > abs(float(div_history[0]))

    # Same routine in two dimensions: f(w) = ||w - target||^2.
    target = np.array([2.0, -4.0])
    found, _ = gradient_descent(lambda w: 2.0 * (w - target), x0=np.zeros(2), lr=0.2, steps=30)
    assert np.allclose(found, target, atol=1e-4)
    print(f"converged to {float(converged):.4f} (target 3)")
    print(f"large step size ended at {float(diverged):.4f}")
    print(f"2d solution {found}")


if __name__ == "__main__":
    main()
