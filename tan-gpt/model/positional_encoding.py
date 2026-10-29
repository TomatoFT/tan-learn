"""Positional encodings.

Self-attention is a set operation: it does not know which token came first
unless the vectors carry a position. Two versions are here.

Sinusoidal encodings are fixed sines and cosines of different frequencies,
the scheme from the original transformer:

    pe[pos, 2i]   = sin(pos / 10000^(2i/d))
    pe[pos, 2i+1] = cos(pos / 10000^(2i/d))

Learned encodings are an embedding table indexed by absolute position.
GPT uses the learned table. `at(positions)` supports a KV cache, where the
next token's position is not zero.
"""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError("n_embd must be even")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd)
        )
        pe = torch.zeros(max_len, n_embd)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", pe)

    def at(self, positions):
        return self.pe[positions]

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.at(positions)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(block_size, n_embd)

    def at(self, positions):
        return self.embedding(positions)

    def forward(self, x):
        if x.size(1) > self.embedding.num_embeddings:
            raise ValueError("sequence is longer than block_size")
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.at(positions)


def main():
    n_embd = 8
    sinusoidal = SinusoidalPositionalEncoding(n_embd, max_len=20)
    positions = torch.arange(4)
    manual = []
    for pos in range(4):
        row = []
        for i in range(0, n_embd, 2):
            angle = pos / (10000 ** (i / n_embd))
            row.extend((math.sin(angle), math.cos(angle)))
        manual.append(row)
    torch.testing.assert_close(sinusoidal.at(positions), torch.tensor(manual))

    tokens = torch.zeros(2, 4, n_embd)
    added = sinusoidal(tokens)
    assert torch.equal(added[0], sinusoidal.at(positions))

    learned = LearnedPositionalEncoding(block_size=6, n_embd=n_embd)
    encoded = learned.at(torch.tensor([0, 2, 5]))
    assert encoded.shape == (3, n_embd)
    assert torch.equal(encoded[0], learned.embedding.weight[0])
    summed = learned(torch.zeros(2, 3, n_embd))
    assert summed.shape == (2, 3, n_embd)
    print("sinusoidal and learned positional encodings ok")


if __name__ == "__main__":
    main()
