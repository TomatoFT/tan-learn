"""Character-level vocabulary.

Every character in the training text gets an integer id. Encoding and
decoding are lookups, so a string round-trips exactly when every character
was seen while building the vocabulary.
"""


class CharVocab:
    def __init__(self, text=None, stoi=None):
        if stoi is None:
            if text is None:
                raise ValueError("provide text or an existing stoi map")
            chars = sorted(set(text))
            stoi = {char: index for index, char in enumerate(chars)}
        self.stoi = dict(stoi)
        self.itos = {index: char for char, index in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the vocabulary") from exc

    def decode(self, ids):
        try:
            return "".join(self.itos[int(index)] for index in ids)
        except KeyError as exc:
            raise ValueError(f"id {exc.args[0]!r} is not in the vocabulary") from exc


def main():
    text = "abacaba\nhello"
    vocab = CharVocab(text)
    assert vocab.decode(vocab.encode(text)) == text
    assert vocab.vocab_size == len(set(text))
    assert vocab.encode("") == []
    assert vocab.decode([]) == ""
    try:
        vocab.encode("z")
    except ValueError:
        pass
    else:
        raise AssertionError("unseen characters should fail")
    restored = CharVocab(stoi=vocab.stoi)
    assert restored.decode(restored.encode(text)) == text
    print(f"vocab size {vocab.vocab_size}")
    print("ids", vocab.encode("aba"))


if __name__ == "__main__":
    main()
