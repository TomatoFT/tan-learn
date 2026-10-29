"""Multi-head self-attention.

One head can only write a narrow vector. Multi-head attention runs several
heads and concatenates them, then mixes the result with an output projection.
The heads are computed together: one linear layer emits every query, key,
and value, and a reshape splits them into heads.

Pass a KV cache at inference time. The cache stores past keys and values so
a new token does not recompute them. The caller advances the cache after the
whole transformer layer stack has read it.
"""

import torch
from torch import nn

from model.attention import CausalSelfAttentionHead, scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, cache=None, layer_idx=0):
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        mixed = scaled_dot_product_attention(
            q,
            k,
            v,
            causal=True,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_drop(self.proj(mixed))


def _copy_heads(module):
    """Rebuild the fused projections as separate heads with the same weights."""
    channels = module.qkv.in_features
    heads = nn.ModuleList()
    bias = module.qkv.bias is not None
    for index in range(module.n_head):
        head = CausalSelfAttentionHead(
            channels,
            module.head_dim,
            block_size=16,
            dropout=0.0,
            bias=bias,
        )
        start = index * module.head_dim
        stop = start + module.head_dim
        with torch.no_grad():
            head.query.weight.copy_(module.qkv.weight[start:stop])
            head.key.weight.copy_(module.qkv.weight[channels + start:channels + stop])
            head.value.weight.copy_(module.qkv.weight[2 * channels + start:2 * channels + stop])
            if bias:
                head.query.bias.copy_(module.qkv.bias[start:stop])
                head.key.bias.copy_(module.qkv.bias[channels + start:channels + stop])
                head.value.bias.copy_(module.qkv.bias[2 * channels + start:2 * channels + stop])
        heads.append(head)
    return heads


def main():
    from types import SimpleNamespace

    torch.manual_seed(0)
    config = SimpleNamespace(n_embd=32, n_head=4, dropout=0.0, bias=True)
    module = MultiHeadAttention(config)
    module.eval()
    x = torch.randn(2, 6, 32)
    y = module(x)
    assert y.shape == x.shape

    changed = x.clone()
    changed[:, -1, :] += 3
    y_changed = module(changed)
    torch.testing.assert_close(y[:, :-1], y_changed[:, :-1])

    separate = _copy_heads(module)
    for head in separate:
        head.eval()
    concatenated = torch.cat([head(x) for head in separate], dim=-1)
    with torch.no_grad():
        expected = module.proj(concatenated)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-4)
    print("multi-head attention matches separate heads")


if __name__ == "__main__":
    main()
