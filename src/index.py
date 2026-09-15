"""Inverted index construction and persistence."""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parse_pmc import Document
from .preprocess import preprocess


@dataclass
class Posting:
    doc_id: str
    tf: int
    positions: list[int]


@dataclass
class InvertedIndex:
    """term → list of postings; plus document lengths and IDF."""

    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[str, int] = field(default_factory=dict)
    doc_token_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    N: int = 0
    idf: dict[str, float] = field(default_factory=dict)
    stem: bool = True
    remove_stopwords: bool = True

    def df(self, term: str) -> int:
        return len(self.postings.get(term, []))

    def vocabulary_size(self) -> int:
        return len(self.postings)


def build_index(
    docs: list[Document],
    *,
    stem: bool = True,
    remove_stopwords: bool = True,
) -> InvertedIndex:
    """Build an inverted index with TF and positions; compute IDF."""
    raw: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[str, int] = {}
    doc_token_counts: dict[str, dict[str, int]] = {}

    for doc in docs:
        tokens = preprocess(
            doc.full_text,
            remove_stopwords=remove_stopwords,
            stem=stem,
        )
        doc_lengths[doc.doc_id] = len(tokens)
        tf_map: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            tf_map[term].append(pos)
        counts = {t: len(positions) for t, positions in tf_map.items()}
        doc_token_counts[doc.doc_id] = counts
        for term, positions in tf_map.items():
            raw[term].append(Posting(doc.doc_id, len(positions), positions))

    N = len(docs)
    idf = {
        term: math.log((N + 1) / (len(posts) + 1)) + 1.0
        for term, posts in raw.items()
    }
    return InvertedIndex(
        postings=dict(raw),
        doc_lengths=doc_lengths,
        doc_token_counts=doc_token_counts,
        N=N,
        idf=idf,
        stem=stem,
        remove_stopwords=remove_stopwords,
    )


def save_index(index: InvertedIndex, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path | str) -> InvertedIndex:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_docs(docs: list[Document], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def load_docs(path: Path | str) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.from_dict(json.loads(line)))
    return docs


def index_summary(index: InvertedIndex) -> dict[str, Any]:
    return {
        "num_documents": index.N,
        "vocabulary_size": index.vocabulary_size(),
        "stem": index.stem,
        "remove_stopwords": index.remove_stopwords,
        "avg_doc_length": (
            round(sum(index.doc_lengths.values()) / index.N, 2) if index.N else 0
        ),
    }
