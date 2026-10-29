"""Sentiment analysis with a bag of words.

Preprocessing turns each sentence into words. A word vocabulary maps those
words to ids, dropping nothing important on this tiny set. The classifier
is one linear layer over word counts. Positive and negative sentences use
different words, so the linear layer can separate them.
"""

import torch
from torch import nn

from data.nlp_preprocessing import preprocess
from data.tokenizer_utils import WordVocab
from foundations.training_loop import training_loop

TRAIN = [
    ("i love this wonderful movie", 1),
    ("a good and excellent story", 1),
    ("happy joyful and love the ending", 1),
    ("wonderful acting and a good script", 1),
    ("what a delightful and excellent book", 1),
    ("i hate this terrible movie", 0),
    ("a bad and awful story", 0),
    ("sad boring and hate the ending", 0),
    ("terrible acting and a bad script", 0),
    ("what a poor and awful book", 0),
]
TEST = [
    ("a wonderful excellent movie", 1),
    ("i love the good ending", 1),
    ("a terrible awful movie", 0),
    ("i hate the bad ending", 0),
]


def _bow_matrix(sentences, vocab):
    rows = []
    for sentence in sentences:
        ids, _truncated = vocab.encode(preprocess(sentence))
        row = torch.zeros(len(vocab))
        for index in ids:
            row[index] += 1.0
        rows.append(row)
    return torch.stack(rows)


def accuracy(model, features, labels):
    model.eval()
    with torch.no_grad():
        preds = model(features).argmax(dim=-1)
    return float((preds == labels).float().mean())


def main():
    torch.manual_seed(0)
    train_tokens = [preprocess(text) for text, _label in TRAIN]
    vocab = WordVocab(train_tokens, min_freq=1)
    train_x = _bow_matrix([text for text, _label in TRAIN], vocab)
    train_y = torch.tensor([label for _text, label in TRAIN])
    test_x = _bow_matrix([text for text, _label in TEST], vocab)
    test_y = torch.tensor([label for _text, label in TEST])

    model = nn.Linear(len(vocab), 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    history = training_loop(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        [(train_x, train_y)],
        epochs=40,
    )
    score = accuracy(model, test_x, test_y)
    assert history[-1] < history[0]
    assert score == 1.0
    print(f"vocab {len(vocab)}  loss {history[0]:.3f} -> {history[-1]:.3f}")
    print(f"test accuracy {score:.3f}")


if __name__ == "__main__":
    main()
