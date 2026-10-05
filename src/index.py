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
    avg_doc_length: float = 0.0
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
            doc.search_text,
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
    # Robertson/Sparck Jones BM25 IDF.
    idf = {
        term: math.log(1 + (N - len(posts) + 0.5) / (len(posts) + 0.5))
        for term, posts in raw.items()
    }
    return InvertedIndex(
        postings=dict(raw),
        doc_lengths=doc_lengths,
        doc_token_counts=doc_token_counts,
        N=N,
        idf=idf,
        avg_doc_length=(sum(doc_lengths.values()) / N) if N else 0.0,
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
    from .library import active_documents
    return active_documents(docs, Path(path).parent.parent / "library.sqlite3")


def index_summary(index: InvertedIndex) -> dict[str, Any]:
    return {
        "num_documents": index.N,
        "vocabulary_size": index.vocabulary_size(),
        "stem": index.stem,
        "remove_stopwords": index.remove_stopwords,
        "avg_doc_length": (
            round(index.avg_doc_length, 2) if index.N else 0
        ),
    }


def rebuild_local_index(raw_dir: Path, processed_dir: Path, *,
                        stem: bool | None = None, remove_stopwords: bool | None = None) -> int:
    """Merge saved XMLs into the collection while preserving existing settings."""
    from .parse_pmc import parse_directory

    docs_path = processed_dir / "docs.jsonl"
    index_path = processed_dir / "index.pkl"
    existing = load_index(index_path) if index_path.exists() else None
    docs = {doc.doc_id: doc for doc in load_docs(docs_path)} if docs_path.exists() else {}
    for doc in parse_directory(raw_dir):
        if not doc.pmid and doc.pmcid:
            known = next((value for value in docs.values()
                          if value.pmcid == doc.pmcid and value.pmid), None)
            if known is not None:
                doc.doc_id = doc.pmid = known.pmid
        # A previously indexed PMC-only document may now have a known PMID.
        aliases = [key for key, value in docs.items()
                   if key != doc.doc_id and doc.pmcid and value.pmcid == doc.pmcid]
        for key in aliases:
            previous_alias = docs.pop(key)
            if len(previous_alias.body) > len(doc.body):
                previous_alias.doc_id = previous_alias.pmid = doc.doc_id
                doc = previous_alias
        previous = docs.get(doc.doc_id)
        if previous is None or len(doc.body) >= len(previous.body):
            docs[doc.doc_id] = doc
    from .library import hidden_article_ids
    if not docs and not hidden_article_ids(raw_dir.parent / "library.sqlite3"):
        raise ValueError("No readable article XML files were found.")
    from .library import active_documents
    documents = active_documents(list(docs.values()), raw_dir.parent / "library.sqlite3")
    index = build_index(
        documents,
        stem=stem if stem is not None else (existing.stem if existing else True),
        remove_stopwords=remove_stopwords if remove_stopwords is not None else (existing.remove_stopwords if existing else True),
    )
    # Complete parsing and preprocessing before replacing either output file.
    save_docs(documents, docs_path)
    save_index(index, index_path)
    return len(documents)
