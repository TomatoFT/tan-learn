"""Byte-level byte-pair encoding.

Training starts from the 256 raw bytes. Each step counts adjacent pairs,
merges the most frequent pair into a new token, and repeats until the
vocabulary reaches the requested size. Encoding applies those merges from
oldest to newest. Decoding concatenates the bytes each id stands for.

Because the base vocabulary is bytes, any Unicode string round-trips.
"""

from collections import Counter


def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids, pair, new_id):
    merged = []
    index = 0
    first, second = pair
    while index < len(ids):
        if index < len(ids) - 1 and ids[index] == first and ids[index + 1] == second:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {index: bytes([index]) for index in range(256)}

    def train(self, text, vocab_size):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 so every byte exists")
        ids = list(text.encode("utf-8"))
        for offset in range(vocab_size - 256):
            counts = pair_counts(ids)
            if not counts:
                break
            best = max(counts.values())
            pair = min(pair for pair, count in counts.items() if count == best)
            new_id = 256 + offset
            ids = merge_pair(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return ids

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = pair_counts(ids)
            pair = min(counts, key=lambda item: self.merges.get(item, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_pair(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[index] for index in ids)
        return data.decode("utf-8", errors="replace")


def main():
    text = "aaabdaaabac"
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=260)
    first_pair = next(iter(tokenizer.merges))
    assert first_pair == (ord("a"), ord("a"))
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode("")) == ""
    other = "hello 你好 — café"
    assert tokenizer.decode(tokenizer.encode(other)) == other
    # Training on a repeated word should create tokens above the byte range.
    words = BPETokenizer()
    words.train("banana " * 20, vocab_size=280)
    encoded = words.encode("banana ")
    assert any(token >= 256 for token in encoded)
    assert words.decode(encoded) == "banana "
    try:
        BPETokenizer().train("ab", vocab_size=10)
    except ValueError:
        pass
    else:
        raise AssertionError("a byte vocabulary cannot shrink below 256")
    print("first merge", first_pair, "->", tokenizer.merges[first_pair])
    print("banana ids", encoded)


if __name__ == "__main__":
    main()
