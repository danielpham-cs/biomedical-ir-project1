"""Surface, stem, substring and fuzzy matching with inspectable text offsets."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from html import escape

from .engine import normalize_whitespace, split_sentences, tokenize_word_spans
from .markup import plain_text
from .preprocess import preprocess


def edit_distance(left: str, right: str, cutoff: int) -> int:
    """Dynamic-programming optimal string alignment; cutoff + 1 means too far."""
    if abs(len(left) - len(right)) > cutoff:
        return cutoff + 1
    previous_previous = None
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            value = min(previous[column] + 1, current[column - 1] + 1,
                        previous[column - 1] + (left_char != right_char))
            if (previous_previous is not None and row > 1 and column > 1
                    and left_char == right[column - 2] and left[row - 2] == right_char):
                value = min(value, previous_previous[column - 2] + 1)
            current.append(value)
        if min(current) > cutoff:
            return cutoff + 1
        previous_previous, previous = previous, current
    return min(previous[-1], cutoff + 1)


def edit_distance_table(left: str, right: str) -> tuple[list[list[int]], list[dict]]:
    """Full OSA table D[i][j] plus one optimal backtrace, for the demo.

    Same recurrence as edit_distance without the cutoff: insertion, deletion,
    substitution (cost 0 on a match) and adjacent transposition, each cost 1.
    """
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        table[i][0] = i
    for j in range(len(right) + 1):
        table[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + (left[i - 1] != right[j - 1]))
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + 1)
    steps, i, j = [], len(left), len(right)
    while i or j:
        if (i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]
                and table[i][j] == table[i - 2][j - 2] + 1 and left[i - 1] != right[j - 1]):
            steps.append({"operation": "transpose", "from": left[i - 2:i], "to": right[j - 2:j], "cost": 1, "cell": (i, j)})
            i, j = i - 2, j - 2
        elif i and j and table[i][j] == table[i - 1][j - 1] + (left[i - 1] != right[j - 1]):
            same = left[i - 1] == right[j - 1]
            steps.append({"operation": "match" if same else "substitute", "from": left[i - 1], "to": right[j - 1],
                          "cost": 0 if same else 1, "cell": (i, j)})
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            steps.append({"operation": "delete", "from": left[i - 1], "to": "", "cost": 1, "cell": (i, j)})
            i -= 1
        else:
            steps.append({"operation": "insert", "from": "", "to": right[j - 1], "cost": 1, "cell": (i, j)})
            j -= 1
    return table, steps[::-1]


@lru_cache(maxsize=50000)
def normalized_word(word: str, stem: bool) -> str:
    tokens = preprocess(word, remove_stopwords=False, stem=stem)
    return tokens[0] if len(tokens) == 1 else word


@dataclass(frozen=True)
class MatchPlan:
    terms: tuple[str, ...]
    kind: str = "STEM"
    method: str = "AND"
    stem: bool = True
    max_edits: int = 1

    @lru_cache(maxsize=100000)
    def token_matches(self, word: str) -> tuple[tuple[str, int], ...]:
        if self.kind == "STEM":
            normalized = normalized_word(word, self.stem)
            return ((normalized, 0),) if normalized in self.terms else ()
        if self.kind == "PARTIAL":
            return tuple((term, 0) for term in self.terms if term in word)
        return tuple((term, distance) for term in self.terms
                     if (distance := edit_distance(term, word, self.max_edits)) <= self.max_edits)


def make_match_plan(query: str, *, stem: bool = True, method: str = "AND",
                    match_mode: str = "STEM", max_edits: int = 1) -> MatchPlan:
    kind, method = match_mode.upper(), method.upper()
    if kind not in {"STEM", "PARTIAL", "FUZZY"}:
        raise ValueError("Match mode must be STEM, PARTIAL or FUZZY.")
    if kind != "STEM" and method == "PHRASE":
        raise ValueError("PHRASE uses standard token matching; choose AND or OR for partial/fuzzy matching.")
    if max_edits not in {1, 2}:
        raise ValueError("Fuzzy edit distance must be 1 or 2.")
    terms = preprocess(query, remove_stopwords=kind == "STEM" and method != "PHRASE",
                       stem=kind == "STEM" and stem)
    return MatchPlan(tuple(terms if method == "PHRASE" else dict.fromkeys(terms)), kind, method, stem, max_edits)


@dataclass(frozen=True)
class Occurrence:
    term: str
    word: str
    start: int
    end: int
    section: str
    sentence: int
    sentence_start: int
    sentence_end: int
    distance: int = 0

    def row(self, text: str) -> dict:
        return {"Query term": self.term, "Matched text": text[self.start:self.end], "Word / phrase": self.word,
                "Section": self.section, "Sentence": self.sentence,
                "Char start (0-based)": self.start, "Char end (exclusive)": self.end,
                "Start in sentence": self.start - self.sentence_start,
                "End in sentence": self.end - self.sentence_start,
                "Edit distance": self.distance, "Context": text[self.sentence_start:self.sentence_end]}


@lru_cache(maxsize=256)
def _scope(abstract: str, body: str):
    sections, pieces, cursor = [], [], 0
    for name, raw in (("Abstract", abstract), ("Body", body)):
        paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", plain_text(raw))]
        text = "\n\n".join(part for part in paragraphs if part)
        if not text:
            continue
        if pieces:
            cursor += 2
        sections.append((cursor, cursor + len(text), name))
        pieces.append(text)
        cursor += len(text)
    text = "\n\n".join(pieces)
    sentences, cursor = [], 0
    for sentence in split_sentences(text):
        start = text.index(sentence, cursor)
        cursor = start + len(sentence)
        sentences.append((start, cursor))
    return text, sections, sentences, tokenize_word_spans(text)


def searchable_text(document) -> str:
    """Canonical displayed plaintext: abstract/keywords + body, title excluded."""
    return _scope(document.abstract, document.body)[0]


def _substring_spans(source: str, term: str):
    # Map casefolded positions back to source characters (e.g. ß -> ss).
    folded, offsets = [], []
    for index, char in enumerate(source):
        folded.append(char.casefold())
        offsets.extend([index] * len(char.casefold()))
    value, cursor = "".join(folded), 0
    while (position := value.find(term, cursor)) >= 0:
        yield offsets[position], offsets[position + len(term) - 1] + 1
        cursor = position + 1


def _phrase_spans(text, tokens, plan):
    normalized = [normalized_word(word, plan.stem) for word, _, _ in tokens]
    width = len(plan.terms)
    for offset in range(len(tokens) - width + 1):
        if tuple(normalized[offset:offset + width]) == plan.terms:
            start, end = tokens[offset][1], tokens[offset + width - 1][2]
            if "\n\n" not in text[start:end]:
                yield start, end


def document_matches(document, plan: MatchPlan) -> list[Occurrence]:
    text, sections, sentences, tokens = _scope(document.abstract, document.body)
    if not plan.terms or not sentences:
        return []
    spans = []
    if plan.method == "PHRASE":
        spans = [(" ".join(plan.terms), text[start:end], start, end, 0)
                 for start, end in _phrase_spans(text, tokens, plan)]
    else:
        for word, start, end in tokens:
            for term, distance in plan.token_matches(word):
                if plan.kind == "PARTIAL":
                    spans.extend((term, text[start:end], start + a, start + b, distance)
                                 for a, b in set(_substring_spans(text[start:end], term)))
                else:
                    spans.append((term, text[start:end], start, end, distance))
    starts = [start for start, _ in sentences]
    section_starts = [start for start, _, _ in sections]
    matches = []
    for term, word, start, end, distance in sorted(spans, key=lambda item: (item[2], item[3], item[0])):
        number = bisect_right(starts, start) - 1
        sentence_start, sentence_end = sentences[number]
        # A phrase may span heuristic sentences, but never paragraphs.
        sentence_end = max(sentence_end, sentences[bisect_right(starts, end - 1) - 1][1])
        section = sections[bisect_right(section_starts, start) - 1][2]
        matches.append(Occurrence(term, word, start, end, section, number + 1,
                                  sentence_start, sentence_end, distance))
    return matches


def highlight_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Escape source HTML and merge overlapping matches before marking them."""
    merged = []
    for start, end in sorted(set(spans)):
        if not 0 <= start < end <= len(text):
            raise ValueError("Highlight span is outside the text.")
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    output, cursor = [], 0
    for start, end in merged:
        output.extend((escape(text[cursor:start]), f"<mark>{escape(text[start:end])}</mark>"))
        cursor = end
    output.append(escape(text[cursor:]))
    return "".join(output)


def match_snippet(text: str, matches: list[Occurrence], window: int = 150, *, highlighted: bool = False) -> str:
    """Include the first occurrence of every distinct query term."""
    first = {}
    for match in matches:
        first.setdefault(match.term, match)
    windows = []
    for match in sorted(first.values(), key=lambda item: item.start):
        start, end = max(0, match.start - window // 2), min(len(text), match.end + window // 2)
        if windows and start <= windows[-1][1] + 24:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    fragments = []
    for start, end in windows:
        fragment = text[start:end]
        if highlighted:
            fragment = highlight_spans(fragment, [(max(m.start, start) - start, min(m.end, end) - start)
                                                  for m in matches if m.start < end and m.end > start])
        fragments.append(("…" if start else "") + fragment + ("…" if end < len(text) else ""))
    return " … ".join(fragments) or (escape(text[:window]) if highlighted else text[:window])


def highlight_plan(text: str, plan: MatchPlan) -> str:
    """Highlight a snippet using precisely the same surface matching rules."""
    tokens = tokenize_word_spans(text)
    if plan.method == "PHRASE" and plan.terms:
        return highlight_spans(text, list(_phrase_spans(text, tokens, plan)))
    spans = []
    for word, start, end in tokens:
        for term, _ in plan.token_matches(word):
            if plan.kind == "PARTIAL":
                spans.extend((start + a, start + b) for a, b in _substring_spans(text[start:end], term))
            else:
                spans.append((start, end))
    return highlight_spans(text, spans)
