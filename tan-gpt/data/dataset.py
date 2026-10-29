"""GPT dataset preparation.

A language model trains on the next character. The whole text is encoded
into one long id tensor, then split into a training prefix and a held-out
suffix. Batches are random windows cut from those tensors.
"""

from pathlib import Path

import torch

from data.vocab import CharVocab

_PASSAGE = """
the cat sat on the mat. the cat sat on the hat.
the dog ran in the sun. the dog ran in the rain.
attention reads the past and writes a new vector.
layer norm rescales each token after every block.
the model predicts the next character from the context.
a small network learns this text one batch at a time.
hello world. hello world. the learner says hello.
gradient descent walks downhill on the loss.
"""

SAMPLE_TEXT = (_PASSAGE.strip() + "\n") * 40


def load_corpus(path=None):
    if path is None:
        return SAMPLE_TEXT
    return Path(path).read_text(encoding="utf-8")


def prepare_dataset(text, train_frac=0.9):
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    vocab = CharVocab(text)
    ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    if ids.numel() < 2:
        raise ValueError("need at least two characters")
    split = int(ids.numel() * train_frac)
    split = min(max(split, 1), ids.numel() - 1)
    return vocab, ids[:split], ids[split:]


def main():
    vocab, train_ids, val_ids = prepare_dataset(SAMPLE_TEXT)
    assert vocab.decode(train_ids.tolist() + val_ids.tolist()) == SAMPLE_TEXT
    assert len(train_ids) > len(val_ids)
    assert train_ids.dtype == torch.long
    custom = "abcabcabc"
    custom_vocab, custom_train, custom_val = prepare_dataset(custom, train_frac=0.5)
    assert custom_vocab.vocab_size == 3
    assert len(custom_train) + len(custom_val) == len(custom)
    print(f"vocab {vocab.vocab_size}  train {len(train_ids)}  val {len(val_ids)}")


if __name__ == "__main__":
    main()
