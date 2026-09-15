"""Document statistics using rule-based sentence splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .eos import split_sentences
from .preprocess import tokenize


@dataclass
class DocStats:
    num_characters: int
    num_words: int
    num_sentences: int
    vocabulary_size: int
    avg_sentence_length: float
    sentences: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep sentences available but UI can decide whether to show all
        return d


def compute_stats(text: str) -> DocStats:
    """Compute character / word / sentence statistics for a document."""
    text = text or ""
    sentences = split_sentences(text)
    words = tokenize(text)
    vocab = set(words)
    avg = (len(words) / len(sentences)) if sentences else 0.0
    return DocStats(
        num_characters=len(text),
        num_words=len(words),
        num_sentences=len(sentences),
        vocabulary_size=len(vocab),
        avg_sentence_length=round(avg, 2),
        sentences=sentences,
    )
