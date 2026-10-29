"""Sample text from a trained GPT.

The next-token distribution is the softmax of the last position's logits.
Temperature flattens or sharpens it. Top-k keeps the k largest logits.
Top-p keeps the smallest set of tokens whose probability exceeds p.
A KV cache makes each new token attend to the stored prefix instead of
recomputing it.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from foundations.train import CHECKPOINT_PATH, load_checkpoint
from model.kv_cache import KVCache


def sample_logits(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(ordered, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, index, ordered)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    was_training = model.training
    model.eval()
    try:
        if max_new_tokens == 0:
            return idx
        if use_cache:
            return _generate_cached(model, idx, max_new_tokens, temperature, top_k, top_p)
        return _generate_sliding(model, idx, max_new_tokens, temperature, top_k, top_p)
    finally:
        model.train(was_training)


def _generate_cached(model, idx, max_new_tokens, temperature, top_k, top_p):
    room = model.config.block_size
    if idx.size(1) >= room:
        idx = idx[:, -(room - 1):]
    max_new_tokens = min(max_new_tokens, room - idx.size(1))
    cache = KVCache(
        n_layers=model.config.n_layer,
        batch_size=idx.size(0),
        max_seq=model.config.block_size,
        n_kv_heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
        device=idx.device,
        dtype=next(model.parameters()).dtype,
    )
    logits, _loss = model(idx, cache=cache)
    for _ in range(max_new_tokens):
        nxt = sample_logits(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, nxt], dim=1)
        logits, _loss = model(nxt, cache=cache)
    return idx


def _generate_sliding(model, idx, max_new_tokens, temperature, top_k, top_p):
    for _ in range(max_new_tokens):
        window = idx[:, -model.config.block_size:]
        logits, _loss = model(window)
        nxt = sample_logits(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def _check_sampler():
    logits = torch.tensor([[0.0, 1.0, 3.0]])
    assert int(sample_logits(logits, temperature=0).item()) == 2
    torch.manual_seed(0)
    drawn = sample_logits(logits.expand(16, -1), temperature=1.0, top_k=1)
    assert torch.all(drawn == 2)
    top_p = sample_logits(logits, temperature=1.0, top_p=0.9)
    assert top_p.shape == (1, 1)


def main():
    _check_sampler()
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--prompt", type=str, default="the ")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    args = parser.parse_args()
    path = args.checkpoint
    if not Path(path).exists():
        raise SystemExit(f"no checkpoint at {path}. Train one with: python -m foundations.train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_checkpoint(path, device=device)
    prompt = torch.tensor([vocab.encode(args.prompt)], dtype=torch.long, device=device)
    torch.manual_seed(0)
    out = generate(
        model,
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=True,
    )
    text = vocab.decode(out[0].tolist())
    assert text.startswith(args.prompt)
    print(text)


if __name__ == "__main__":
    main()
