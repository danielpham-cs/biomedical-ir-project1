"""Text preprocessing: tokenize, stop-words, Porter stemming."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)


def ensure_nltk_data() -> None:
    """Download NLTK resources if missing."""
    for resource, path in (
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric / hyphenated tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


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
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    if stem:
        stemmer = _stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens
