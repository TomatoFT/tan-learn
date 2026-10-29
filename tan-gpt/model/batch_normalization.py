"""Batch normalization.

Normalize each feature using the mean and variance across the batch:

    y = weight * (x - batch_mean) / sqrt(batch_var + eps) + bias

During training the batch statistics update running estimates. At eval time
those running estimates replace the batch, so a single example still has a
defined normalization. The feature axis is dim 1, matching (N, C) or (N, C, L).
"""

import torch
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if x.ndim < 2:
            raise ValueError("expected x shaped (N, C, *)")
        reduce_dims = (0,) + tuple(range(2, x.ndim))
        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, correction=0)
            count = x.numel() // x.shape[1]
            var_unbiased = var * (count / (count - 1)) if count > 1 else var
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var_unbiased, alpha=self.momentum)
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var
        shape = (1, -1) + (1,) * (x.ndim - 2)
        normalized = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self.weight.view(shape) * normalized + self.bias.view(shape)


def _match_reference(shape):
    features = shape[1]
    ours = BatchNorm(features)
    reference = nn.BatchNorm1d(features)
    with torch.no_grad():
        ours.weight.copy_(reference.weight)
        ours.bias.copy_(reference.bias)
    x = torch.randn(*shape)
    torch.testing.assert_close(ours(x), reference(x), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ours.running_mean, reference.running_mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ours.running_var, reference.running_var, atol=1e-5, rtol=1e-4)
    ours.eval()
    reference.eval()
    held_out = torch.randn(*shape)
    torch.testing.assert_close(ours(held_out), reference(held_out), atol=1e-5, rtol=1e-4)


def main():
    torch.manual_seed(0)
    _match_reference((8, 4))
    _match_reference((5, 4, 7))
    print("batch norm matches torch.nn.BatchNorm1d")


if __name__ == "__main__":
    main()
