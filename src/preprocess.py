"""Text preprocessing: tokenize, stop-words, Porter stemming."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from .engine import tokenize_words



def ensure_nltk_data() -> None:
    """Only stopwords are needed: tokenization and EOS use our own rules."""
    local_data = Path(__file__).resolve().parents[1] / ".venv" / "nltk_data"
    if local_data.is_dir() and str(local_data) not in nltk.data.path:
        nltk.data.path.insert(0, str(local_data))
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise LookupError("English stopwords could not be downloaded. Check your connection and rebuild the index.")


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Return final-specification tokens for statistics and retrieval."""
    return tokenize_words(text)


def preprocess(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> list[str]:
    """
    Full IR preprocessing pipeline shared by documents and queries.

    Steps: lowercase → tokenize → optional stop-word removal → optional Porter stem.
    Porter stemmer: public-domain algorithm via NLTK (Martin Porter, 1980).
    """
    tokens = tokenize(text)
    if remove_stopwords:
        stops = _stop_words()
        tokens = [t for t in tokens if t not in stops]
    if stem:
        stemmer = _stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens
