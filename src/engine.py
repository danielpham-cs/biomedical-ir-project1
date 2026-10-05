"""Authoritative text rules for Biomedical IR Project 1.

This module is the single implementation of the final specification for
abstract scope, character normalization, word tokenization, and EOS splitting.
"""

from __future__ import annotations

import re

_HYPHENS = frozenset({"-", "‐", "‑", "–"})
_APOSTROPHES = frozenset({"'", "’"})
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "fig", "figs", "eq", "eqs",
        "vs", "approx", "no", "vol", "et al", "i.e", "e.g",
    }
)
_SENTENCE_END = re.compile(r'([.!?。！？]+)(["\'”’»\)\]\}]*)((?:\s+)|$)')


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and strip its ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def compose_abstract(paragraphs: list[str], keywords: list[str]) -> str:
    """Create the project abstract field, including one final Keywords line.

    A pre-existing Keywords line is retained and ends abstract collection, so
    trailing XML content cannot be misclassified as abstract prose.
    """
    kept: list[str] = []
    has_keywords_line = False
    for paragraph in paragraphs:
        clean = normalize_whitespace(paragraph)
        if not clean:
            continue
        if re.match(r"^keywords\s*:", clean, re.IGNORECASE):
            kept.append(clean)
            has_keywords_line = True
            break
        kept.append(clean)
    cleaned_keywords = [normalize_whitespace(keyword) for keyword in keywords]
    cleaned_keywords = [keyword for keyword in cleaned_keywords if keyword]
    if cleaned_keywords and not has_keywords_line:
        kept.append("Keywords: " + ", ".join(cleaned_keywords))
    return "\n\n".join(kept)


def _abbreviation_at_end(token: str) -> bool:
    value = token.rstrip(".").casefold()
    if value in _ABBREVIATIONS:
        return True
    if len(value) == 1 and value.isalpha():
        return True
    parts = value.split(".")
    return len(parts) > 1 and all(len(part) == 1 and part.isalpha() for part in parts)


def tokenize_words(text: str) -> list[str]:
    """Tokenize words exactly as defined by the final project specification."""
    return [token for token, _, _ in tokenize_word_spans(text)]


def tokenize_word_spans(text: str) -> list[tuple[str, int, int]]:
    """Return tokens and zero-based [start, end) offsets in the input text.

    Do not casefold or normalize the input before scanning: Unicode casefolding
    and whitespace normalization can change character offsets.
    """
    text = text or ""
    tokens: list[tuple[str, int, int]] = []
    index = 0
    while index < len(text):
        if not text[index].isalnum():
            index += 1
            continue
        start = index
        value: list[str] = []
        saw_abbreviation_dot = False
        while index < len(text):
            current = text[index]
            previous = text[index - 1] if index > start else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if current.isalnum():
                value.append(current)
                index += 1
                continue
            if current == "." and previous.isdigit() and following.isdigit():
                value.append(current)
                index += 1
                continue
            if current == "%" and value and value[-1].isdigit():
                value.append(current)
                index += 1
                break
            if current in _HYPHENS | _APOSTROPHES:
                # Numeric ranges split (12-15, 0.44–0.97); a hyphenated
                # biomedical form such as HIV-1 or 5-stage remains one word.
                if previous.isalnum() and following.isalnum() and not (
                    current in _HYPHENS and previous.isdigit() and following.isdigit()
                ):
                    value.append(current)
                    index += 1
                    continue
            if current == "." and previous.isalpha() and following.isalpha():
                # Preserve only conventional multi-dot initials/abbreviations.
                segment = "".join(value).split(".")[-1]
                if len(segment) == 1:
                    value.append(current)
                    saw_abbreviation_dot = True
                    index += 1
                    continue
            if current == "." and saw_abbreviation_dot:
                value.append(current)
                index += 1
            break
        token = "".join(value).rstrip(".") if not saw_abbreviation_dot else "".join(value)
        if token:
            tokens.append((token.casefold(), start, start + len(token)))
        if index == start:
            index += 1
    return tokens


def _last_token_before_boundary(text: str, boundary: int) -> str:
    prefix = text[:boundary].rstrip()
    return prefix.split(" ")[-1] if prefix else ""


def _split_paragraph(paragraph: str) -> list[str]:
    """Apply abbreviation-aware EOS rules to one non-empty paragraph."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        candidate_end = match.end(2)
        punctuation = match.group(1)
        before_punctuation = paragraph[start:match.start(1)].rstrip()
        last = _last_token_before_boundary(before_punctuation, len(before_punctuation))
        is_multiword_abbreviation = bool(re.search(r"\bet\s+al$", before_punctuation, re.IGNORECASE))
        if punctuation == "." and (_abbreviation_at_end(last) or is_multiword_abbreviation):
            continue
        candidate = paragraph[start:candidate_end].strip()
        if candidate:
            sentences.append(candidate)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[str]:
    """Split each non-empty paragraph into heuristic EOS sentences."""
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", text or "")]
    return [sentence for paragraph in paragraphs if paragraph for sentence in _split_paragraph(paragraph)]
