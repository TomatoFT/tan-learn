"""GPT training loop.

Each step draws a batch of character windows, predicts the next character,
and takes an AdamW step on the cross-entropy. Gradient clipping and a
gradient-norm check keep a bad step from blowing up the weights. The
checkpoint stores the model, the config, and the character vocabulary.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from data.dataset import load_corpus, prepare_dataset
from data.loader import get_batch
from data.vocab import CharVocab
from foundations.training_diagnostics import flag_issues, gradient_norm, learning_rate
from model.gpt import GPT, GPTConfig

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "tiny-gpt.pt"


def save_checkpoint(model, vocab, path=CHECKPOINT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
            "stoi": vocab.stoi,
        },
        path,
    )
    return path


def load_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**blob["config"])
    model = GPT(config).to(device)
    model.load_state_dict(blob["model_state"])
    vocab = CharVocab(stoi=blob["stoi"])
    return model, vocab


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, batches=4):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        contexts, targets = get_batch(data, block_size, batch_size, device=device)
        _logits, loss = model(contexts, targets)
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)


def train(
    steps=300,
    batch_size=16,
    lr=3e-3,
    block_size=32,
    n_layer=2,
    n_head=4,
    n_embd=64,
    text_path=None,
    device=None,
    seed=0,
    checkpoint_path=CHECKPOINT_PATH,
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_corpus(text_path)
    vocab, train_ids, val_ids = prepare_dataset(text)
    if min(len(train_ids), len(val_ids)) <= block_size:
        raise ValueError("the corpus is shorter than block_size")
    config = GPTConfig(
        vocab_size=vocab.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    interval = max(steps // 4, 1)
    recorded = []
    print(f"device {device}  parameters {model.num_parameters()}  vocab {vocab.vocab_size}")
    for step in range(steps):
        contexts, targets = get_batch(train_ids, block_size, batch_size, device=device)
        _logits, loss = model(contexts, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        issues = flag_issues(gradient_norm(model), [], grad_limit=1e4)
        if issues:
            print(f"step {step}: {', '.join(issues)}")
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        if step % interval == 0 or step == steps - 1:
            train_loss = estimate_loss(model, train_ids, block_size, batch_size, device)
            val_loss = estimate_loss(model, val_ids, block_size, batch_size, device)
            recorded.append(train_loss)
            print(
                f"step {step:4d}  train {train_loss:.3f}  val {val_loss:.3f}  "
                f"lr {learning_rate(optimizer):.1e}"
            )
    path = save_checkpoint(model, vocab, checkpoint_path)
    print(f"saved {path}")
    return model, vocab, recorded


def main():
    parser = argparse.ArgumentParser(description="Train the tiny character GPT")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--text", type=str, default=None, help="optional UTF-8 training file")
    args = parser.parse_args()
    _model, _vocab, recorded = train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        block_size=args.block_size,
        text_path=args.text,
    )
    if args.steps >= 30 and len(recorded) >= 2:
        if not recorded[-1] < recorded[0]:
            raise SystemExit(f"expected the loss to fall, saw {recorded[0]:.3f} -> {recorded[-1]:.3f}")


if __name__ == "__main__":
    main()
