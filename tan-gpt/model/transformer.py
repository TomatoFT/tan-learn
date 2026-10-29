"""A pre-norm transformer block.

The block is two residual branches. Each branch normalizes, transforms,
and adds the result back:

    x = x + Attention(Norm(x))
    x = x + FeedForward(Norm(x))

The feed-forward network expands the features by 4, applies GELU, and
projects back. Stacking this block is the transformer.
"""

import torch
from torch import nn
import torch.nn.functional as F

from model.multi_head_attention import MultiHeadAttention
from model.normalization import LayerNorm
from model.rms_normalization import RMSNorm


def make_norm(kind, n_embd):
    if kind == "layernorm":
        return LayerNorm(n_embd)
    if kind == "rmsnorm":
        return RMSNorm(n_embd)
    raise ValueError(f"unknown norm {kind}")


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.fc2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = make_norm(config.norm, config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = make_norm(config.norm, config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x, cache=None, layer_idx=0):
        x = x + self.attn(self.ln1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.ffn(self.ln2(x))
        return x


def main():
    from types import SimpleNamespace

    torch.manual_seed(0)
    config = SimpleNamespace(
        n_embd=32,
        n_head=4,
        dropout=0.0,
        bias=True,
        norm="layernorm",
    )
    block = TransformerBlock(config)
    block.eval()
    x = torch.randn(2, 6, 32)
    y = block(x)
    assert y.shape == x.shape
    changed = x.clone()
    changed[:, -1] += 3
    torch.testing.assert_close(y[:, :-1], block(changed)[:, :-1])

    config.norm = "rmsnorm"
    rms_block = TransformerBlock(config)
    assert rms_block(x).shape == x.shape
    print("transformer block ok")


if __name__ == "__main__":
    main()
