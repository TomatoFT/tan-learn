"""Key-value cache for fast inference.

A causal model produces the next token from the tokens it has already seen.
Without a cache, every new token repeats the attention work for the whole
prefix. The cache stores each layer's keys and values. A new token appends
one key and one value, then attends to the stored prefix.

`update` writes into the cache and returns the prefix including the new
tokens. It does not move the cursor. The model advances the cursor once,
after every layer has written its row for this step.
"""

import torch


class KVCache:
    def __init__(self, n_layers, batch_size, max_seq, n_kv_heads, head_dim, device, dtype=None):
        if dtype is None:
            dtype = torch.float32
        shape = (n_layers, batch_size, n_kv_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len = 0
        self.max_seq = max_seq

    def update(self, layer_idx, k, v):
        batch, _heads, time, _dim = k.shape
        start = self.seq_len
        end = start + time
        if end > self.max_seq:
            raise RuntimeError(f"KV cache length {end} exceeds max_seq {self.max_seq}")
        self.k[layer_idx, :batch, :, start:end, :] = k
        self.v[layer_idx, :batch, :, start:end, :] = v
        return self.k[layer_idx, :batch, :, :end, :], self.v[layer_idx, :batch, :, :end, :]

    def advance(self, n_tokens):
        self.seq_len += n_tokens

    def reset(self):
        self.seq_len = 0
        self.k.zero_()
        self.v.zero_()


def main():
    from model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    with torch.no_grad():
        full, _loss = model(idx)
        cache = KVCache(
            n_layers=config.n_layer,
            batch_size=idx.size(0),
            max_seq=config.block_size,
            n_kv_heads=config.n_head,
            head_dim=config.n_embd // config.n_head,
            device=idx.device,
            dtype=full.dtype,
        )
        first, _loss = model(idx[:, :6], cache=cache)
        second, _loss = model(idx[:, 6:], cache=cache)
    cached = torch.cat([first, second], dim=1)
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-4)
    assert cache.seq_len == idx.size(1)

    cache.reset()
    assert cache.seq_len == 0
    try:
        with torch.no_grad():
            too_long = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
            model(too_long, cache=cache)
    except ValueError:
        pass
    else:
        raise AssertionError("the cache must refuse a context past block_size")
    print("cached logits match the full forward pass")


if __name__ == "__main__":
    main()
