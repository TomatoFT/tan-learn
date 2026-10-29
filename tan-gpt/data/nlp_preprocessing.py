"""NLP preprocessing.

Raw text becomes a list of tokens through a short pipeline: lowercase,
strip punctuation, collapse whitespace, then split. Sentence splitting is
separate, because a model that classifies sentences needs the boundaries
that word tokenization throws away.
"""

import re

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return _PUNCT.sub(" ", text)


def collapse_whitespace(text):
    return _SPACE.sub(" ", text).strip()


def word_tokenize(text):
    if not text:
        return []
    return text.split(" ")


def sentence_tokenize(text):
    parts = [part.strip() for part in _SENTENCE.findall(text)]
    return [part for part in parts if part]


def preprocess(text, lowercase_text=True, drop_punctuation=True):
    """Return word tokens from a raw string."""
    if lowercase_text:
        text = lowercase(text)
    if drop_punctuation:
        text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return word_tokenize(text)


def main():
    raw = "  Don't   stop. Hello, WORLD!  "
    tokens = preprocess(raw)
    assert tokens == ["don", "t", "stop", "hello", "world"]
    assert preprocess("") == []
    assert preprocess("   ") == []
    sentences = sentence_tokenize("Hello there. How are you? Fine!")
    assert sentences == ["Hello there.", "How are you?", "Fine!"]
    assert sentence_tokenize("") == []
    # Case and extra spaces are normalized before the split.
    assert preprocess("A  B\tC") == ["a", "b", "c"]
    print("tokens", tokens)
    print("sentences", sentences)


if __name__ == "__main__":
    main()
