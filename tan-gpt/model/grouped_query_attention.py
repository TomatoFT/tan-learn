"""Grouped-query attention.

Multi-head attention stores a key and a value for every query head. Grouped-query
attention lets several query heads share one key/value head. The cache shrinks
by the group size. When the number of key/value heads equals the number of
query heads, the layer is ordinary multi-head attention.
"""

import torch
from torch import nn

from model.attention import scaled_dot_product_attention


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    batch, heads, time, dim = x.shape
    x = x.unsqueeze(2).expand(batch, heads, n_rep, time, dim)
    return x.reshape(batch, heads * n_rep, time, dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, dropout=0.0, bias=True):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_rep = n_head // n_kv_head
        self.dropout_p = dropout
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, cache=None, layer_idx=0):
        batch, time, channels = x.shape
        q = self.q_proj(x).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
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


def main():
    from model.kv_cache import KVCache
    from model.multi_head_attention import MultiHeadAttention
    from types import SimpleNamespace

    torch.manual_seed(0)
    n_embd, n_head = 32, 4
    gqa = GroupedQueryAttention(n_embd, n_head=n_head, n_kv_head=n_head, dropout=0.0, bias=False)
    mha = MultiHeadAttention(SimpleNamespace(n_embd=n_embd, n_head=n_head, dropout=0.0, bias=False))
    with torch.no_grad():
        gqa.q_proj.weight.copy_(mha.qkv.weight[:n_embd])
        gqa.k_proj.weight.copy_(mha.qkv.weight[n_embd:2 * n_embd])
        gqa.v_proj.weight.copy_(mha.qkv.weight[2 * n_embd:])
        gqa.proj.weight.copy_(mha.proj.weight)
    x = torch.randn(2, 5, n_embd)
    gqa.eval()
    mha.eval()
    torch.testing.assert_close(gqa(x), mha(x), atol=1e-5, rtol=1e-4)

    shared = GroupedQueryAttention(n_embd, n_head=4, n_kv_head=2, dropout=0.0, bias=True)
    shared.eval()
    y = shared(x)
    changed = x.clone()
    changed[:, -1] += 2
    torch.testing.assert_close(y[:, :-1], shared(changed)[:, :-1])

    cache = KVCache(
        n_layers=1,
        batch_size=2,
        max_seq=8,
        n_kv_heads=2,
        head_dim=n_embd // 4,
        device=x.device,
    )
    with torch.no_grad():
        full = shared(x)
        first = shared(x[:, :3], cache=cache, layer_idx=0)
        cache.advance(3)
        second = shared(x[:, 3:], cache=cache, layer_idx=0)
    torch.testing.assert_close(full, torch.cat([first, second], dim=1), atol=1e-5, rtol=1e-4)
    assert cache.k.shape[2] == 2
    print("grouped-query attention matches multi-head when every head has its own key")


if __name__ == "__main__":
    main()
