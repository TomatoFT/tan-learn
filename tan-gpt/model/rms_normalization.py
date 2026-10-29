"""Root-mean-square normalization.

RMSNorm scales each token by the root mean square of its features. It drops
the mean subtraction that LayerNorm does, which removes a reduction and a
bias parameter:

    y = weight * x / sqrt(mean(x^2) + eps)

Recent language models use this in place of LayerNorm. The GPT class can
select it with norm='rmsnorm'.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def main():
    torch.manual_seed(0)
    norm = RMSNorm(8)
    x = torch.randn(4, 5, 8)
    y = norm(x)
    assert y.shape == x.shape
    # With a unit scale, each token's root mean square is about 1.
    measured = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(measured, torch.ones(4, 5), atol=1e-4)
    if hasattr(nn, "RMSNorm"):
        reference = nn.RMSNorm(8, eps=1e-5)
        with torch.no_grad():
            reference.weight.copy_(norm.weight)
        torch.testing.assert_close(y, reference(x), atol=1e-5, rtol=1e-4)
        print("rms norm matches torch.nn.RMSNorm")
    else:
        print("rms norm root-mean-square is 1")


if __name__ == "__main__":
    main()
