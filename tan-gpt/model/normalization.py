"""Layer normalization.

For each token, normalize its feature vector to zero mean and unit variance,
then apply a learned scale and shift:

    y = weight * (x - mean) / sqrt(var + eps) + bias

Mean and variance are taken over the feature axis only, so the result does
not depend on the other tokens or the batch size.
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def main():
    torch.manual_seed(0)
    norm = LayerNorm(8)
    reference = nn.LayerNorm(8)
    with torch.no_grad():
        reference.weight.copy_(norm.weight)
        reference.bias.copy_(norm.bias)
    x = torch.randn(4, 6, 8)
    torch.testing.assert_close(norm(x), reference(x), atol=1e-5, rtol=1e-4)

    plain = LayerNorm(8)
    y = plain(torch.randn(3, 5, 8))
    assert torch.allclose(y.mean(dim=-1), torch.zeros(3, 5), atol=1e-4)
    assert torch.allclose(y.std(dim=-1, correction=0), torch.ones(3, 5), atol=1e-4)
    print("layer norm matches torch.nn.LayerNorm")


if __name__ == "__main__":
    main()
