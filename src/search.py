"""AND, OR, and phrase retrieval ranked with the final-specification BM25."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .index import InvertedIndex
from .matching import (MatchPlan, Occurrence, document_matches, edit_distance as _damerau_levenshtein,
                       highlight_plan, make_match_plan, match_snippet, searchable_text)
from .markup import plain_text
from .parse_pmc import Document
from .preprocess import preprocess, tokenize

K1 = 1.2
B = 0.75


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
    occurrences: list[Occurrence] = field(default_factory=list)
    match_mode: str = "STEM"
    snippet_html: str = ""


def highlight_text(text: str, terms: list[str], index: InvertedIndex) -> str:
    """Escaped highlighting with the same tokenizer and stemmer as retrieval."""
    return highlight_plan(text, MatchPlan(tuple(terms), stem=index.stem))


def build_spelling_vocabulary(docs: list[Document]) -> dict[str, int]:
    """Collect English surface words and their corpus frequencies.

    The retrieval index stores stems, while spelling suggestions must show the
    original readable word (for example, ``diabetes`` rather than ``diabet``).
    """
    counts: Counter[str] = Counter()
    for document in docs:
        counts.update(token for token in tokenize(document.search_text) if token.isascii() and token.isalpha())
    return dict(counts)


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


def spelling_candidates(word: str, vocabulary: dict[str, int], limit: int = 10) -> list[dict]:
    """The ranked candidates behind a suggestion: distance first, then corpus frequency."""
    folded = word.casefold()
    cutoff = 1 if len(folded) <= 4 else 2
    ranked = sorted(
        (distance, -frequency, candidate)
        for candidate, frequency in vocabulary.items()
        if abs(len(candidate) - len(folded)) <= cutoff
        if (distance := _damerau_levenshtein(folded, candidate, cutoff)) <= cutoff
    )
    return [{"candidate": candidate, "edit_distance": distance, "corpus_frequency": -negative}
            for distance, negative, candidate in ranked[:limit]]


def _candidate_ids(terms: list[str], index: InvertedIndex, mode: str, docs: list[Document]) -> set[str]:
    if not terms:
        return set()
    # PHRASE is validated on the original token stream, including stopwords
    # omitted from the index. They must not empty the preliminary candidate set.
    if mode == "PHRASE":
        terms = [term for term in terms if term in index.postings]
        if not terms:
            return {document.doc_id for document in docs}
    posting_sets = [{posting.doc_id for posting in index.postings.get(term, [])} for term in terms]
    return set.union(*posting_sets) if mode == "OR" else set.intersection(*posting_sets)


def _count_terms(document: Document, terms: list[str], *, stem: bool) -> dict[str, dict[str, int]]:
    """Count query terms in each scope, respecting the index's stem setting."""
    unique = list(dict.fromkeys(terms))
    abstract = Counter(preprocess(plain_text(document.abstract), remove_stopwords=False, stem=stem))
    body = Counter(preprocess(plain_text(document.body), remove_stopwords=False, stem=stem))
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
    match_mode: str = "STEM",
    max_edits: int = 1,
) -> list[SearchHit]:
    """Rank standard, substring or fuzzy matches; record every source location.

    Partial/fuzzy modes match casefolded surface tokens, without stemming or
    stopword removal. AND requires each query token, OR requires any token.
    BM25 ranks the actual matched vocabulary terms using the existing index.
    """
    requested = method.upper()
    if requested in {"BOOLEAN", "TFIDF"}:
        requested = boolean_mode.upper() if requested == "BOOLEAN" else "OR"
    mode = requested if requested in {"AND", "OR", "PHRASE"} else "AND"
    plan = make_match_plan(query, stem=index.stem, method=mode, match_mode=match_mode, max_edits=max_edits)
    if not plan.terms:
        return []
    terms = list(plan.terms)
    ids = (_candidate_ids(terms, index, mode, docs) if plan.kind == "STEM"
           else {document.doc_id for document in docs})
    hits: list[SearchHit] = []
    for document in docs:
        if document.doc_id not in ids:
            continue
        matches = document_matches(document, plan)
        found = {match.term for match in matches}
        if not matches or (mode == "AND" and not set(terms).issubset(found)):
            continue
        matched = terms if mode == "PHRASE" else [term for term in terms if term in found]
        if plan.kind == "STEM":
            counts = _count_terms(document, terms, stem=index.stem)
            ranking_terms = terms
        else:
            counts = {term: {"abstract": 0, "body": 0, "full": 0} for term in terms}
            for match in matches:
                counts[match.term][match.section.lower()] += 1
                counts[match.term]["full"] += 1
            ranking_terms = preprocess(" ".join({match.word for match in matches}),
                                       remove_stopwords=False, stem=index.stem)
        hits.append(SearchHit(
            doc_id=document.doc_id, score=round(_bm25_score(document.doc_id, ranking_terms, index), 4),
            title=document.title, pmid=document.pmid, pmcid=document.pmcid,
            snippet=match_snippet(searchable_text(document), matches), matched_terms=matched,
            snippet_html=match_snippet(searchable_text(document), matches, highlighted=True),
            term_counts=counts, occurrences=matches, match_mode=plan.kind,
        ))
    hits.sort(key=lambda hit: (-hit.score, hit.title.casefold()))
    return hits[:max(0, top_k)]


def boolean_search(query: str, index: InvertedIndex, docs: list[Document], *, mode: str = "AND", top_k: int = 20) -> list[SearchHit]:
    """Compatibility wrapper for existing callers."""
    return search(query, index, docs, method=mode, top_k=top_k)


def tfidf_search(query: str, index: InvertedIndex, docs: list[Document], *, top_k: int = 20) -> list[SearchHit]:
    """Deprecated compatibility wrapper; final project ranking is BM25."""
    return search(query, index, docs, method="OR", top_k=top_k)
