"""Porter's original 1980 suffix-stripping rules, implemented for Homework 2.

Specification: https://tartarus.org/martin/PorterStemmer/def.txt
This is the original ABLI variant, not NLTK's extended Porter or Snowball.
Only ASCII alphabetic words are stemmed; biomedical identifiers are preserved.
Each step returns the fired rule so the demo can show the full derivation.
"""

from functools import lru_cache


def _consonants(word: str) -> list[bool]:
    flags: list[bool] = []
    for char in word:
        flags.append(not flags[-1] if char == "y" and flags else char not in "aeiou")
    return flags


def measure(word: str) -> int:
    """Count VC transitions in [C](VC)^m[V]."""
    flags = _consonants(word)
    return sum(not left and right for left, right in zip(flags, flags[1:]))


def _has_vowel(word: str) -> bool:
    return any(not flag for flag in _consonants(word))


def _cvc(word: str) -> bool:
    return len(word) >= 3 and _consonants(word)[-3:] == [True, False, True] and word[-1] not in "wxy"


def form(word: str) -> str:
    """Collapse the word into its [C](VC)^m[V] consonant/vowel pattern."""
    pattern = ""
    for consonant in _consonants(word):
        letter = "C" if consonant else "V"
        if not pattern.endswith(letter):
            pattern += letter
    return pattern


STEP2 = (("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
         ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
         ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
         ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
         ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"))
STEP3 = (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
         ("ical", "ic"), ("ful", ""), ("ness", ""))
STEP4 = ("ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ion",
         "ism", "ate", "iti", "ous", "ive", "ize", "al", "er", "ic", "ou")


def _replace(word: str, rules: tuple[tuple[str, str], ...], minimum: int) -> tuple[str, str]:
    # Stop after the longest matching suffix, even when its condition fails.
    for suffix, replacement in sorted(rules, key=lambda pair: -len(pair[0])):
        if word.endswith(suffix):
            root = word[:-len(suffix)]
            if measure(root) > minimum:
                return root + replacement, f"(m>{minimum}) {suffix.upper()} → {replacement.upper() or '∅'}"
            return word, f"{suffix.upper()} matched, but m({root})={measure(root)} ≤ {minimum}"
    return word, ""


def _step1a(word):
    for suffix, replacement in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement, f"{suffix.upper()} → {replacement.upper() or '∅'}"
    return word, ""


def _step1b(word):
    if word.endswith("eed"):
        if measure(word[:-3]) > 0:
            return word[:-1], "(m>0) EED → EE"
        return word, f"EED matched, but m({word[:-3]})=0"
    for suffix in ("ed", "ing"):
        if word.endswith(suffix) and _has_vowel(word[:-len(suffix)]):
            word = word[:-len(suffix)]
            rule = f"(*v*) {suffix.upper()} → ∅"
            if word.endswith(("at", "bl", "iz")):
                return word + "e", rule + f"; {word[-2:].upper()} → {word[-2:].upper()}E"
            if len(word) > 1 and word[-1] == word[-2] and _consonants(word)[-1] and word[-1] not in "lsz":
                return word[:-1], rule + "; (*d and not *L/*S/*Z) → single letter"
            if measure(word) == 1 and _cvc(word):
                return word + "e", rule + "; (m=1 and *o) → E"
            return word, rule
    return word, ""


def _step1c(word):
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i", "(*v*) Y → I"
    return word, ""


def _step2(word):
    return _replace(word, STEP2, 0)


def _step3(word):
    return _replace(word, STEP3, 0)


def _step4(word):
    # Remove suffixes only for m > 1; ION additionally requires S/T.
    for suffix in STEP4:
        if word.endswith(suffix):
            root = word[:-len(suffix)]
            if measure(root) > 1 and (suffix != "ion" or root.endswith(("s", "t"))):
                return root, f"(m>1{' and (*S or *T)' if suffix == 'ion' else ''}) {suffix.upper()} → ∅"
            return word, f"{suffix.upper()} matched, but condition fails (m({root})={measure(root)})"
    return word, ""


def _step5a(word):
    if word.endswith("e"):
        root = word[:-1]
        m = measure(root)
        if m > 1 or (m == 1 and not _cvc(root)):
            return root, "(m>1) E → ∅" if m > 1 else "(m=1 and not *o) E → ∅"
    return word, ""


def _step5b(word):
    if word.endswith("ll") and measure(word) > 1:
        return word[:-1], "(m>1 and *d and *L) → single letter"
    return word, ""


STEPS = (("1a", "Plurals", _step1a), ("1b", "-ed / -ing", _step1b), ("1c", "Y → I", _step1c),
         ("2", "Double suffixes", _step2), ("3", "-ic-, -full, -ness", _step3),
         ("4", "Remove suffixes (m>1)", _step4), ("5a", "Final E", _step5a), ("5b", "Double L", _step5b))


def porter_steps(word: str) -> list[dict]:
    """Every Porter step with its input, output and fired rule (empty if none matched)."""
    word = word.lower()
    if not word or not word.isascii() or not word.isalpha():
        return []
    rows = []
    for step, title, function in STEPS:
        after, rule = function(word)
        rows.append({"step": step, "title": title, "before": word, "after": after,
                     "rule": rule or "no suffix matched", "changed": after != word,
                     "form": form(after), "m": measure(after)})
        word = after
    return rows


@lru_cache(maxsize=50000)
def porter_stem(word: str) -> str:
    word = word.lower()
    if not word or not word.isascii() or not word.isalpha():
        return word
    for _, _, function in STEPS:
        word = function(word)[0]
    return word
