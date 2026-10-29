"""Tokenization edge cases.

Whitespace splitting looks fine until the input is empty, a word was never
seen, a contraction contains an apostrophe, or a batch has to be rectangular.
Special tokens reserve ids for padding, unknown words, and sequence boundaries.
Padding makes lengths match. The attention mask marks those pads so a model
can ignore them.
"""

import re

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)

_CONTRACTION = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.!?]", re.IGNORECASE)


def tokenize_keeping_apostrophes(text):
    """Split on whitespace and punctuation, but keep internal apostrophes."""
    return [match.group(0).lower() for match in _CONTRACTION.finditer(text)]


def fit_length(ids, max_len, pad_id):
    """Truncate on the right, or pad on the right, to exactly max_len ids."""
    if max_len < 0:
        raise ValueError("max_len must be non-negative")
    if len(ids) > max_len:
        return ids[:max_len], True
    return ids + [pad_id] * (max_len - len(ids)), False


def padding_mask(batch, pad_id):
    """1 for a real token, 0 for padding. Shape matches the batch of id lists."""
    return [[0 if token == pad_id else 1 for token in sequence] for sequence in batch]


class WordVocab:
    def __init__(self, tokenized_texts, min_freq=1):
        counts = {}
        for tokens in tokenized_texts:
            for token in tokens:
                if token in SPECIALS:
                    continue
                counts[token] = counts.get(token, 0) + 1
        words = sorted(token for token, count in counts.items() if count >= min_freq)
        self.itos = list(SPECIALS) + words
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def encode(self, tokens, max_len=None, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.stoi[BOS])
        for token in tokens:
            ids.append(self.stoi.get(token, self.unk_id))
        if add_eos:
            ids.append(self.stoi[EOS])
        truncated = False
        if max_len is not None:
            ids, truncated = fit_length(ids, max_len, self.pad_id)
        return ids, truncated

    def decode(self, ids, skip_special=False):
        words = []
        for index in ids:
            token = self.itos[index]
            if skip_special and token in SPECIALS:
                continue
            words.append(token)
        return words


def collate(sequences, pad_id):
    width = max((len(sequence) for sequence in sequences), default=0)
    padded = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    return padded, padding_mask(padded, pad_id)


def main():
    assert tokenize_keeping_apostrophes("Don't stop.") == ["don't", "stop", "."]
    assert tokenize_keeping_apostrophes("") == []

    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "ran"],
        ["a", "rareword"],
    ]
    vocab = WordVocab(corpus, min_freq=2)
    assert "rareword" not in vocab.stoi
    assert "the" in vocab.stoi
    assert "cat" in vocab.stoi

    ids, truncated = vocab.encode(["the", "zebra", "cat"], max_len=10, add_bos=True, add_eos=True)
    assert truncated is False
    assert ids[0] == vocab.stoi[BOS]
    assert ids[4] == vocab.stoi[EOS]
    assert vocab.unk_id in ids
    assert len(ids) == 10
    assert ids[-1] == vocab.pad_id

    long_ids, was_truncated = vocab.encode(["the", "cat", "sat", "the"], max_len=2)
    assert was_truncated is True
    assert long_ids == [vocab.stoi["the"], vocab.stoi["cat"]]

    empty, empty_truncated = vocab.encode([], max_len=3)
    assert empty == [vocab.pad_id] * 3
    assert empty_truncated is False

    batch, mask = collate([[1, 2], [], [3]], pad_id=0)
    assert batch == [[1, 2], [0, 0], [3, 0]]
    assert mask == [[1, 1], [0, 0], [1, 0]]
    assert padding_mask([[vocab.pad_id]], vocab.pad_id) == [[0]]

    # A literal special-token string in the text keeps the reserved id.
    special_ids, _ = vocab.encode([PAD, "the"])
    assert special_ids[0] == vocab.pad_id
    print(f"word vocab size {len(vocab)}")
    print("encoded", ids)
    print("mask", mask)


if __name__ == "__main__":
    main()
