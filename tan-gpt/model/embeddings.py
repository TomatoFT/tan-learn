"""Token embeddings.

An embedding is a lookup table. Token id i reads row i, which is a learned
vector of size n_embd. Rows that occur in similar contexts are pushed
together by the language-model loss.
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx):
        return self.embedding(idx)


def main():
    torch.manual_seed(0)
    table = TokenEmbedding(11, 4)
    idx = torch.tensor([[1, 2, 1], [0, 10, 3]])
    out = table(idx)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, table.embedding.weight[idx])
    # The same id always reads the same vector.
    assert torch.equal(out[0, 0], out[0, 2])
    print("embedding shape", tuple(out.shape))


if __name__ == "__main__":
    main()
