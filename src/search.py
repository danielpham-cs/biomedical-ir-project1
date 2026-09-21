"""AND, OR, and phrase retrieval ranked with the final-specification BM25."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html import escape

from .index import InvertedIndex
from .parse_pmc import Document
from .preprocess import preprocess, tokenize

K1 = 1.2
B = 0.75
_DISPLAY_TOKEN = re.compile(r"[^\W_]+(?:[.\-‐‑–'’][^\W_]+)*%?", re.UNICODE)


@dataclass
class SearchHit:
    doc_id: str
    score: float
    title: str
    pmid: str
    pmcid: str
    snippet: str
    matched_terms: list[str]
    term_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _doc_map(docs: list[Document]) -> dict[str, Document]:
    return {document.doc_id: document for document in docs}


def _snippet(
    text: str,
    terms: list[str],
    *,
    stem: bool,
    phrase_span: tuple[int, int] | None = None,
    window: int = 150,
) -> str:
    """Show context for every matched query term, even when terms are far apart."""
    if phrase_span is not None:
        start, end = phrase_span
        window_start = max(0, start - window // 2)
        window_end = min(len(text), end + window // 2)
        return (
            ("..." if window_start else "")
            + text[window_start:window_end].strip()
            + ("..." if window_end < len(text) else "")
        )
    targets = set(terms)
    first_match: dict[str, tuple[int, int]] = {}
    for match in _DISPLAY_TOKEN.finditer(text):
        normalized = preprocess(match.group(), remove_stopwords=False, stem=stem)
        if len(normalized) == 1 and normalized[0] in targets:
            first_match.setdefault(normalized[0], (match.start(), match.end()))
    if not first_match:
        return text[:window].strip() + ("..." if len(text) > window else "")

    windows = sorted(
        (max(0, start - window // 2), min(len(text), end + window // 2))
        for start, end in first_match.values()
    )
    merged: list[list[int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 24:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    fragments = [
        ("..." if start else "") + text[start:end].strip() + ("..." if end < len(text) else "")
        for start, end in merged
    ]
    return " … ".join(fragments)


def _query_terms(query: str, mode: str, index: InvertedIndex) -> list[str]:
    """AND/OR discard stopwords; PHRASE preserves them. All modes stem."""
    return preprocess(query, remove_stopwords=mode != "PHRASE", stem=index.stem)


def highlight_text(text: str, terms: list[str], index: InvertedIndex) -> str:
    """Return escaped text with every retrieval-equivalent token marked.

    Display highlighting follows the same tokenizer and stemming setting as the
    inverted index, so a query for ``test`` also marks source forms such as
    ``testing`` and ``tested``.
    """
    targets = set(terms)
    if not text or not targets:
        return escape(text or "")

    rendered: list[str] = []
    cursor = 0
    for match in _DISPLAY_TOKEN.finditer(text):
        rendered.append(escape(text[cursor:match.start()]))
        token = match.group()
        normalized = preprocess(token, remove_stopwords=False, stem=index.stem)
        if len(normalized) == 1 and normalized[0] in targets:
            rendered.append(f"<mark>{escape(token)}</mark>")
        else:
            rendered.append(escape(token))
        cursor = match.end()
    rendered.append(escape(text[cursor:]))
    return "".join(rendered)


def build_spelling_vocabulary(docs: list[Document]) -> dict[str, int]:
    """Collect English surface words and their corpus frequencies.

    The retrieval index stores stems, while spelling suggestions must show the
    original readable word (for example, ``diabetes`` rather than ``diabet``).
    """
    counts: Counter[str] = Counter()
    for document in docs:
        counts.update(token for token in tokenize(document.search_text) if token.isascii() and token.isalpha())
    return dict(counts)


def _damerau_levenshtein(left: str, right: str, cutoff: int) -> int:
    """Edit distance with adjacent transpositions and an early cutoff."""
    if abs(len(left) - len(right)) > cutoff:
        return cutoff + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            cost = left_char != right_char
            value = min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + cost,
            )
            if (
                previous_previous is not None
                and row > 1
                and column > 1
                and left_char == right[column - 2]
                and left[row - 2] == right_char
            ):
                value = min(value, previous_previous[column - 2] + 1)
            current.append(value)
        if min(current) > cutoff:
            return cutoff + 1
        previous_previous, previous = previous, current
    return previous[-1]


def suggest_query_correction(
    query: str,
    index: InvertedIndex,
    vocabulary: dict[str, int],
) -> str | None:
    """Suggest an English typo correction without changing the submitted query.

    Candidate words are ranked by Damerau-Levenshtein distance first and corpus
    frequency second. Non-English tokens, short words, and all-caps biomedical
    abbreviations are deliberately left unchanged.
    """
    replacements: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[A-Za-z]+", query):
        typed = match.group()
        folded = typed.casefold()
        if len(typed) < 4 or typed.isupper() or folded in vocabulary:
            continue
        stems = preprocess(typed, remove_stopwords=False, stem=index.stem)
        if stems and stems[0] in index.postings:
            continue
        cutoff = 1 if len(folded) <= 4 else 2
        candidates = (
            (distance, -frequency, word)
            for word, frequency in vocabulary.items()
            if abs(len(word) - len(folded)) <= cutoff
            if (distance := _damerau_levenshtein(folded, word, cutoff)) <= cutoff
        )
        try:
            _, _, replacement = min(candidates)
        except ValueError:
            continue
        replacements.append((match.start(), match.end(), replacement))

    if not replacements:
        return None
    corrected: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        corrected.append(query[cursor:start])
        corrected.append(replacement)
        cursor = end
    corrected.append(query[cursor:])
    suggestion = "".join(corrected)
    return suggestion if suggestion != query else None


def _phrase_matches(document: Document, terms: list[str], index: InvertedIndex) -> bool:
    if not terms:
        return False
    # A phrase must remain inside one original paragraph.
    for paragraph in re.split(r"\n\s*\n", document.search_text):
        tokens = preprocess(paragraph, remove_stopwords=False, stem=index.stem)
        width = len(terms)
        if any(tokens[offset : offset + width] == terms for offset in range(len(tokens) - width + 1)):
            return True
    return False


def _phrase_span(text: str, terms: list[str], index: InvertedIndex) -> tuple[int, int] | None:
    """Locate the first phrase match so the result snippet can prove the match."""
    if not terms:
        return None
    for paragraph in re.finditer(r"(?:^|\n\s*\n)(.*?)(?=$|\n\s*\n)", text, re.DOTALL):
        offset = paragraph.start(1)
        token_spans: list[tuple[str, int, int]] = []
        for raw_token in _DISPLAY_TOKEN.finditer(paragraph.group(1)):
            normalized = preprocess(raw_token.group(), remove_stopwords=False, stem=index.stem)
            if len(normalized) == 1:
                token_spans.append((normalized[0], offset + raw_token.start(), offset + raw_token.end()))
        width = len(terms)
        for position in range(len(token_spans) - width + 1):
            if [token for token, _, _ in token_spans[position : position + width]] == terms:
                return token_spans[position][1], token_spans[position + width - 1][2]
    return None


def _candidate_ids(terms: list[str], index: InvertedIndex, mode: str, docs: list[Document]) -> set[str]:
    if not terms:
        return set()
    posting_sets = [{posting.doc_id for posting in index.postings.get(term, [])} for term in terms]
    if mode == "OR":
        return set.union(*posting_sets) if posting_sets else set()
    if mode == "PHRASE":
        preliminary = set.intersection(*posting_sets) if posting_sets else set()
        by_id = _doc_map(docs)
        return {doc_id for doc_id in preliminary if doc_id in by_id and _phrase_matches(by_id[doc_id], terms, index)}
    return set.intersection(*posting_sets) if posting_sets else set()


def _count_terms(document: Document, terms: list[str]) -> dict[str, dict[str, int]]:
    """Show stemmed query occurrence totals by the three required scopes."""
    unique = list(dict.fromkeys(terms))
    abstract = Counter(preprocess(document.abstract, remove_stopwords=False, stem=True))
    body = Counter(preprocess(document.body, remove_stopwords=False, stem=True))
    return {
        term: {
            "abstract": abstract[term],
            "body": body[term],
            "full": abstract[term] + body[term],
        }
        for term in unique
    }


def _bm25_score(doc_id: str, terms: list[str], index: InvertedIndex) -> float:
    length = index.doc_lengths.get(doc_id, 0)
    average = index.avg_doc_length or 1.0
    score = 0.0
    for term in set(terms):
        posting = next((item for item in index.postings.get(term, []) if item.doc_id == doc_id), None)
        if posting is None:
            continue
        denominator = posting.tf + K1 * (1 - B + B * length / average)
        score += index.idf[term] * (posting.tf * (K1 + 1) / denominator)
    return score


def search(
    query: str,
    index: InvertedIndex,
    docs: list[Document],
    *,
    method: str = "and",
    boolean_mode: str = "AND",
    top_k: int = 20,
) -> list[SearchHit]:
    """Find AND/OR/PHRASE candidates and rank every candidate using BM25."""
    requested = method.upper()
    if requested in {"BOOLEAN", "TFIDF"}:
        requested = boolean_mode.upper() if requested == "BOOLEAN" else "OR"
    mode = requested if requested in {"AND", "OR", "PHRASE"} else "AND"
    terms = _query_terms(query, mode, index)
    ids = _candidate_ids(terms, index, mode, docs)
    by_id = _doc_map(docs)
    hits: list[SearchHit] = []
    for doc_id in ids:
        document = by_id.get(doc_id)
        if document is None:
            continue
        matched = [term for term in dict.fromkeys(terms) if doc_id in {p.doc_id for p in index.postings.get(term, [])}]
        phrase_span = _phrase_span(document.search_text, terms, index) if mode == "PHRASE" else None
        hits.append(
            SearchHit(
                doc_id=doc_id,
                score=round(_bm25_score(doc_id, terms, index), 4),
                title=document.title,
                pmid=document.pmid,
                pmcid=document.pmcid,
                snippet=_snippet(document.search_text, matched, stem=index.stem, phrase_span=phrase_span),
                matched_terms=matched,
                term_counts=_count_terms(document, terms),
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.title.casefold()))
    return hits[:top_k]


def boolean_search(query: str, index: InvertedIndex, docs: list[Document], *, mode: str = "AND", top_k: int = 20) -> list[SearchHit]:
    """Compatibility wrapper for existing callers."""
    return search(query, index, docs, method=mode, top_k=top_k)


def tfidf_search(query: str, index: InvertedIndex, docs: list[Document], *, top_k: int = 20) -> list[SearchHit]:
    """Deprecated compatibility wrapper; final project ranking is BM25."""
    return search(query, index, docs, method="OR", top_k=top_k)
