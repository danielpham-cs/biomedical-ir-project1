"""Rule-based end-of-sentence (EOS) detection for biomedical text."""

from __future__ import annotations

import re

# Common abbreviations that should NOT trigger sentence breaks.
_ABBREVS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "fig",
        "figs",
        "eq",
        "eqs",
        "no",
        "nos",
        "vol",
        "vs",
        "approx",
        "dept",
        "est",
        "al",
        "etc",
        "i.e",
        "e.g",
        "cf",
        "ref",
        "refs",
        "st",
        "nd",
        "rd",
        "th",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "u.s",
        "u.k",
        "ph.d",
        "m.d",
        "b.s",
        "m.s",
        "a.m",
        "p.m",
    }
)

_SENT_END = re.compile(r"([.!?]+)(\s+|$)")
_INITIAL = re.compile(r"^[A-Z]$")
# True decimals like 1.2 / 3.14 — not bare integers (Fig. 2.)
_DECIMAL = re.compile(r".*\d\.\d+$")
_ELLIPSIS = re.compile(r"\.{2,}$")


def _is_abbrev_token(token: str) -> bool:
    t = token.rstrip(".").lower()
    if t in _ABBREVS:
        return True
    # Single-letter initials: "A." / "J."
    if _INITIAL.match(t):
        return True
    # Multi-initial like "U.S"
    if "." in token.rstrip(".") and all(len(p) <= 1 for p in t.split(".")):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences with abbreviation-aware rules.

    Rules:
    - Split on . ? ! followed by whitespace (or end)
    - Do not split after known abbreviations (Dr., Fig., et al., i.e., e.g.)
    - Do not split after decimals (1.2) or ellipses (...)
    - Do not split after single-letter initials
    """
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text.strip())
    sentences: list[str] = []
    start = 0

    for match in _SENT_END.finditer(text):
        end = match.end(1)  # include punctuation
        candidate = text[start:end].strip()
        if not candidate:
            start = match.end()
            continue

        # Look at last word before the terminator
        before = text[start : match.start(1)].rstrip()
        last_word = before.split(" ")[-1] if before else ""
        punct = match.group(1)

        if punct == "." and (
            _is_abbrev_token(last_word)
            or _DECIMAL.match(last_word.rstrip("."))
            or _ELLIPSIS.match(last_word + punct)
        ):
            continue

        # Require following capital / digit / end for soft confirmation
        after = text[match.end() : match.end() + 1]
        if after and after[0].islower():
            continue

        sentences.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
