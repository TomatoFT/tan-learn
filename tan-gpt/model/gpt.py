"""A small GPT.

Tokens and positions are embedded, added, and passed through a stack of
pre-norm transformer blocks. A final layer norm and a linear head produce
a logit for every vocabulary id at every position. The head shares its
weight matrix with the token embedding.

Training passes the next-token targets and gets back the mean cross-entropy.
Inference can pass a KV cache; the forward pass appends to it and then
advances the cache cursor.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model.embeddings import TokenEmbedding
from model.positional_encoding import LearnedPositionalEncoding
from model.transformer import TransformerBlock, make_norm


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True
    norm: str = "layernorm"


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'")
        self.config = config
        self.tok_emb = TokenEmbedding(config.vocab_size, config.n_embd)
        self.pos_emb = LearnedPositionalEncoding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = make_norm(config.norm, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def trainable_parameters(self):
        seen = set()
        chosen = []
        for param in self.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                chosen.append(param)
        return chosen

    def num_parameters(self):
        return sum(param.numel() for param in self.trainable_parameters())

    def forward(self, idx, targets=None, cache=None):
        _batch, time = idx.shape
        start = 0 if cache is None else cache.seq_len
        if start + time > self.config.block_size:
            raise ValueError(
                f"context length {start + time} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(start, start + time, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb.at(positions))
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx)
        if cache is not None:
            cache.advance(time)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def main():
    import io

    torch.manual_seed(0)
    config = GPTConfig(vocab_size=40, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.randint(0, config.vocab_size, (4, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 8, config.vocab_size)
    manual = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    torch.testing.assert_close(loss, manual)
    assert torch.isfinite(loss)

    model.eval()
    with torch.no_grad():
        base, _loss = model(idx)
        changed = idx.clone()
        changed[:, -1] = (changed[:, -1] + 1) % config.vocab_size
        other, _loss = model(changed)
    torch.testing.assert_close(base[:, :-1], other[:, :-1])

    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    first = None
    last = None
    for _ in range(25):
        _logits, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert model.tok_emb.embedding.weight.grad is not None
        optimizer.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last < first

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = GPT(config)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    clone.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(idx)[0], clone(idx)[0])

    rms = GPT(GPTConfig(vocab_size=40, block_size=16, n_layer=1, n_head=4, n_embd=32, norm="rmsnorm"))
    rms_loss = rms(idx, targets)[1]
    assert torch.isfinite(rms_loss)
    assert model.lm_head.weight is model.tok_emb.embedding.weight
    state = model.state_dict()
    assert torch.equal(state["lm_head.weight"], state["tok_emb.embedding.weight"])
    print(f"parameters {model.num_parameters()}  overfit {first:.3f} -> {last:.3f}")


if __name__ == "__main__":
    main()
