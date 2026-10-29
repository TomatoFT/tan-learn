"""Dead ReLU detector.

A ReLU unit is dead on a batch when its output is zero for every example.
That happens when the pre-activation stays negative, so the gradient through
the unit is zero and the weights feeding it stop moving.

The detector attaches a forward hook and reports the fraction of examples
that produced a zero for each unit.
"""

import torch
from torch import nn


class DeadReLUDetector:
    def __init__(self):
        self.fractions = {}
        self._hooks = []

    def watch(self, module, name):
        def hook(_module, _inputs, output):
            dead = output.detach() <= 0
            reduce_dims = tuple(range(dead.ndim - 1))
            self.fractions[name] = dead.float().mean(dim=reduce_dims)

        self._hooks.append(module.register_forward_hook(hook))

    def dead_units(self, name):
        return self.fractions[name] == 1

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def main():
    torch.manual_seed(0)
    layer = nn.Linear(8, 4)
    torch.nn.init.zeros_(layer.weight)
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([-5.0, -5.0, 1.0, 0.5]))
    relu = nn.ReLU()
    detector = DeadReLUDetector()
    detector.watch(relu, "hidden")
    features = torch.randn(32, 8)
    _ = relu(layer(features))
    dead = detector.dead_units("hidden")
    assert dead.tolist() == [True, True, False, False]
    assert torch.all(detector.fractions["hidden"][:2] == 1)
    detector.close()
    print("dead units", [index for index, flag in enumerate(dead.tolist()) if flag])
    print("zero fraction", detector.fractions["hidden"].tolist())


if __name__ == "__main__":
    main()
