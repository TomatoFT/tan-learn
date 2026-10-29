"""One causal self-attention head.

Each token builds a query, a key, and a value. Attention weights are the
softmax of query-key dot products, scaled so they do not grow with the head
size. The causal mask sets every future score to -inf, so a token can read
itself and earlier tokens only.

    weights = softmax(Q K^T / sqrt(d))
    output  = weights V
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, causal=True, dropout_p=0.0, training=False):
    """Attention for q, k, v shaped (batch, heads, time, dim)."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        query_len = q.size(-2)
        key_len = k.size(-2)
        if query_len > key_len:
            raise ValueError("causal attention needs at least as many keys as queries")
        key_index = torch.arange(key_len, device=q.device)
        query_index = torch.arange(query_len, device=q.device)[:, None]
        # Queries are the last query_len positions of the key sequence.
        allowed = key_index <= (key_len - query_len + query_index)
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_dim, block_size, dropout=0.0, bias=True):
        super().__init__()
        self.head_dim = head_dim
        self.query = nn.Linear(n_embd, head_dim, bias=bias)
        self.key = nn.Linear(n_embd, head_dim, bias=bias)
        self.value = nn.Linear(n_embd, head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _batch, time, _channels = x.shape
        if time > self.mask.size(0):
            raise ValueError("sequence is longer than block_size")
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ v


def main():
    torch.manual_seed(0)
    head = CausalSelfAttentionHead(n_embd=16, head_dim=8, block_size=12, dropout=0.0, bias=True)
    x = torch.randn(2, 6, 16)
    y = head(x)
    assert y.shape == (2, 6, 8)
    changed = x.clone()
    changed[:, -1, :] += 4
    y_changed = head(changed)
    torch.testing.assert_close(y[:, :-1], y_changed[:, :-1])
    assert not torch.allclose(y[:, -1], y_changed[:, -1])

    q = k = v = torch.randn(2, 1, 5, 4)
    manual = scaled_dot_product_attention(q, k, v, causal=True)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    torch.testing.assert_close(manual, torch.softmax(scores, dim=-1) @ v)
    print("causal self-attention head ok")


if __name__ == "__main__":
    main()
