"""Boolean and TF-IDF keyword matching over an inverted index."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .index import InvertedIndex
from .parse_pmc import Document
from .preprocess import preprocess


@dataclass
class SearchHit:
    doc_id: str
    score: float
    title: str
    pmid: str
    snippet: str
    matched_terms: list[str]


def _doc_map(docs: list[Document]) -> dict[str, Document]:
    return {d.doc_id: d for d in docs}


def _snippet(text: str, query_terms: list[str], window: int = 180) -> str:
    lower = text.lower()
    pos = -1
    for term in query_terms:
        # match original-ish forms in raw text
        raw = term.rstrip("s")
        idx = lower.find(raw)
        if idx >= 0:
            pos = idx
            break
        idx = lower.find(term)
        if idx >= 0:
            pos = idx
            break
    if pos < 0:
        return (text[:window] + "…") if len(text) > window else text
    start = max(0, pos - window // 3)
    end = min(len(text), start + window)
    snip = text[start:end].strip()
    if start > 0:
        snip = "…" + snip
    if end < len(text):
        snip = snip + "…"
    return snip


def highlight(text: str, terms: list[str]) -> str:
    """Wrap query term occurrences in markdown-bold markers."""
    if not text or not terms:
        return text
    # Prefer longer terms first
    unique = sorted({t for t in terms if t}, key=len, reverse=True)
    pattern = re.compile(
        r"(" + "|".join(re.escape(t) for t in unique) + r")",
        re.IGNORECASE,
    )
    return pattern.sub(r"**\1**", text)


def boolean_search(
    query: str,
    index: InvertedIndex,
    docs: list[Document],
    *,
    mode: str = "AND",
    top_k: int = 20,
) -> list[SearchHit]:
    """Basic Boolean AND/OR matching using the inverted index."""
    terms = preprocess(
        query,
        remove_stopwords=index.remove_stopwords,
        stem=index.stem,
    )
    if not terms:
        return []

    mode = mode.upper()
    postings_sets = []
    matched_per_term: dict[str, set[str]] = {}
    for term in terms:
        doc_ids = {p.doc_id for p in index.postings.get(term, [])}
        matched_per_term[term] = doc_ids
        postings_sets.append(doc_ids)

    if not postings_sets:
        return []

    if mode == "OR":
        result_ids = set.union(*postings_sets) if postings_sets else set()
    else:
        result_ids = set.intersection(*postings_sets) if postings_sets else set()

    dmap = _doc_map(docs)
    hits: list[SearchHit] = []
    for doc_id in result_ids:
        doc = dmap.get(doc_id)
        if doc is None:
            continue
        matched = [t for t, ids in matched_per_term.items() if doc_id in ids]
        # Score = number of distinct query terms matched (for ranking within Boolean)
        score = float(len(matched))
        hits.append(
            SearchHit(
                doc_id=doc_id,
                score=score,
                title=doc.title,
                pmid=doc.pmid,
                snippet=_snippet(doc.abstract or doc.full_text, terms),
                matched_terms=matched,
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:top_k]


def tfidf_search(
    query: str,
    index: InvertedIndex,
    docs: list[Document],
    *,
    top_k: int = 20,
) -> list[SearchHit]:
    """Rank documents by cosine similarity of TF-IDF vectors."""
    q_terms = preprocess(
        query,
        remove_stopwords=index.remove_stopwords,
        stem=index.stem,
    )
    if not q_terms:
        return []

    # Query TF
    q_tf: dict[str, int] = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_weights: dict[str, float] = {}
    for term, tf in q_tf.items():
        if term not in index.idf:
            continue
        q_weights[term] = (1 + math.log(tf)) * index.idf[term]

    if not q_weights:
        return []

    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm == 0:
        return []

    # Accumulate scores via inverted index
    scores: dict[str, float] = {}
    matched: dict[str, set[str]] = {}
    for term, qw in q_weights.items():
        for posting in index.postings.get(term, []):
            dw = (1 + math.log(posting.tf)) * index.idf[term]
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + qw * dw
            matched.setdefault(posting.doc_id, set()).add(term)

    # Cosine: divide by |d| * |q|
    # Approximate |d| from stored token tf-idf weights
    dmap = _doc_map(docs)
    hits: list[SearchHit] = []
    for doc_id, dot in scores.items():
        counts = index.doc_token_counts.get(doc_id, {})
        d_norm_sq = 0.0
        for term, tf in counts.items():
            idf = index.idf.get(term)
            if idf is None:
                continue
            w = (1 + math.log(tf)) * idf
            d_norm_sq += w * w
        d_norm = math.sqrt(d_norm_sq) if d_norm_sq else 1.0
        score = dot / (d_norm * q_norm)
        doc = dmap.get(doc_id)
        if doc is None:
            continue
        hits.append(
            SearchHit(
                doc_id=doc_id,
                score=round(score, 4),
                title=doc.title,
                pmid=doc.pmid,
                snippet=_snippet(doc.abstract or doc.full_text, q_terms),
                matched_terms=sorted(matched.get(doc_id, [])),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:top_k]


def search(
    query: str,
    index: InvertedIndex,
    docs: list[Document],
    *,
    method: str = "tfidf",
    boolean_mode: str = "AND",
    top_k: int = 20,
) -> list[SearchHit]:
    method = method.lower()
    if method == "boolean":
        return boolean_search(
            query, index, docs, mode=boolean_mode, top_k=top_k
        )
    return tfidf_search(query, index, docs, top_k=top_k)
