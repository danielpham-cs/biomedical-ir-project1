"""Per-question answers for the detailed brief, computed from one analysis result.

Each page shows the questions it is responsible for. The text is built from the
measured numbers, so a different corpus produces different (but honest) answers.
"""
from __future__ import annotations

from .project2_report import findings, fmt, ir_essay

REGIONS = ("Head (0–10%)", "Middle (10–80%)", "Tail (80–100%)")


def _fit(result, condition="B", region="All"):
    return result["conditions"][condition]["fits"][region]


def part5_answers(result) -> dict[str, str]:
    """Part V questions 1–4 (condition B, the brief's punctuation-cleaned baseline)."""
    fit = _fit(result)
    regions = {name: _fit(result, "B", name) for name in REGIONS}
    usable = {n: f for n, f in regions.items() if f["r_squared"] is not None}
    best = min(usable, key=lambda n: usable[n]["rmse_log10"]) if usable else None
    region_text = "; ".join(f"{n}: k={fmt(f['zipf_exponent'], 3)}, R²={fmt(f['r_squared'], 3)}, RMSE={fmt(f['rmse_log10'], 3)}"
                            for n, f in regions.items())
    exponents = ", ".join(f"{c}: {v['fits']['All']['zipf_exponent']:.3f}" for c, v in result["conditions"].items())
    return {
        "Q1 · Does your collection follow Zipf's Law?":
            f"Approximately. On log-log axes the rank-frequency curve is close to a straight line (condition B: R²={fit['r_squared']:.3f}, "
            f"RMSE={fit['rmse_log10']:.3f} in log10 CF). A few terms dominate and a long tail of rare terms follows. "
            f"It is Zipf-like rather than exactly 1/r: the fitted exponent is {fit['zipf_exponent']:.3f}, not 1, and the curve bends at both ends.",
        "Q2 · What is your estimated Zipf exponent?":
            f"Condition B: log10(CF) = {fit['intercept']:.3f} {fit['slope']:+.3f}·log10(r), so k = −slope = {fit['zipf_exponent']:.3f}. "
            f"Other conditions: {exponents}. The value depends on preprocessing and on which ranks are fitted.",
        "Q3 · Is R² sufficient to prove Zipf's Law?":
            "No. R² only says the points are close to some straight line in log space. Ranked frequencies are sorted, so neighbouring points are "
            "strongly dependent and any decreasing curve gets a high R². Thousands of tied tail frequencies dominate the fit, R² does not compare "
            "against alternatives (log-normal, power law with cutoff), and it is not a goodness-of-fit test. Residuals and regional fits are needed; "
            "a constant tail even has undefined R².",
        "Q4 · Which portion fits best?":
            (f"Best (lowest log-space RMSE): {best}. " if best else "") + f"Regional fits for B — {region_text}. "
            "The head is shaped by a few function words, and the tail is a staircase of tied counts (many terms with CF=1, 2, …), "
            "so neither end is a clean power law. The middle ranks carry most of the linear trend.",
    }


def part6_answer(result) -> dict[str, str]:
    """Part VI: how preprocessing affects the distribution (four required angles)."""
    stats = {c: v["stats"] for c, v in result["conditions"].items()}
    heads = {c: ", ".join(r["term"] for r in v["rows"][:5]) for c, v in result["conditions"].items()}
    k = {c: v["fits"]["All"]["zipf_exponent"] for c, v in result["conditions"].items()}
    return {
        "1 · Vocabulary size":
            " → ".join(f"{c}: {s['vocabulary']:,}" for c, s in stats.items()) +
            f". Removing boundary punctuation merges 'data,' / 'data.' into one term (A→B, "
            f"{1 - stats['B']['vocabulary'] / stats['A']['vocabulary']:.1%} fewer terms). Stopword removal drops few types but many tokens "
            f"({stats['B']['tokens']:,} → {stats['C']['tokens']:,}). Porter merges inflected forms (C→D, "
            f"{1 - stats['D']['vocabulary'] / stats['C']['vocabulary']:.1%} fewer terms).",
        "2 · High-frequency terms":
            " | ".join(f"{c}: {h}" for c, h in heads.items()) +
            ". Until stopwords are removed the head is function words; after C the head shows the collection's subject.",
        "3 · Zipf exponent":
            ", ".join(f"{c}: {v:.3f}" for c, v in k.items()) + ". Step changes: " +
            ", ".join(f"{x}→{y} {k[y] - k[x]:+.3f} ({'steeper' if k[y] > k[x] else 'flatter'})" for x, y in ("AB", "BC", "CD")) +
            ". Steps that merge surface variants (punctuation, stemming) shrink the tail relative to the head; "
            "removing stopwords removes the top of the head.",
        "4 · Shape of the distribution":
            "All four curves keep the same heavy-tailed shape. Changes are local: stopword removal lowers and flattens the head, "
            "punctuation cleaning and stemming shorten the tail (fewer hapax: "
            + ", ".join(f"{c}: {s['hapax']:,}" for c, s in stats.items()) + ").",
    }


def part7_answers(result, condition="B") -> dict[str, str]:
    terms = result["conditions"][condition]["selected_terms"]
    n = result["conditions"][condition]["stats"]["documents"]
    common = max(terms, key=lambda r: r["DF"])
    concentrated = max(terms, key=lambda r: r["CF_per_document"])
    return {
        "Q1 · Why can a term have high CF but relatively low DF?":
            f"CF counts every occurrence, DF counts each document once. A term that is repeated inside a few papers accumulates CF without "
            f"spreading: '{concentrated['term']}' has CF={concentrated['CF']} but DF={concentrated['DF']} "
            f"({concentrated['CF_per_document']:.1f} occurrences per containing abstract) — typically an abbreviation or method name that one "
            "paper repeats throughout its abstract.",
        "Q2 · Which measure shows whether a term is widely distributed?":
            f"DF (or DF/N). '{common['term']}' appears in {common['DF']:,} of {n:,} abstracts (DF/N={common['DF'] / n:.2f}); "
            "CF alone cannot tell repetition in one document from presence in many.",
        "Q3 · Why are CF and DF both important in IR?":
            "CF measures total volume: it drives token counts, positional index size and term-frequency statistics (e.g. collection language models). "
            "DF measures spread: it is the postings-list length and the basis of IDF, i.e. how well a term separates documents. "
            "Ranking functions use both kinds of evidence — TF within a document and DF across the collection.",
    }


def part8_answer(result, condition="B") -> dict[str, str]:
    data = result["conditions"][condition]
    n = data["stats"]["documents"]
    head = data["rows"][:10]
    singletons = sum(r["DF"] == 1 for r in data["rows"])
    import math
    return {
        "Why do very common terms get lower IDF?":
            f"IDF = ln(N/DF). As DF approaches N the ratio approaches 1 and IDF approaches 0. The ten highest-CF terms in {condition} have mean IDF "
            f"{sum(r['IDF'] for r in head) / len(head):.3f}, while the {singletons:,} terms found in a single abstract all receive the maximum "
            f"ln({n}) = {math.log(n):.3f}. A term present in almost every document says little about any one of them.",
        "How does Zipf relate to TF-IDF weighting?":
            "Zipf says a small head of terms accounts for most occurrences. Those head terms also have the largest DF, so IDF pushes their weight "
            "toward zero, while the long tail gets high IDF. TF-IDF therefore counteracts the skew that Zipf describes: raw TF would be dominated "
            "by the head, IDF shifts weight to the discriminating middle and tail. CF and DF are correlated but not identical, so IDF must be "
            "computed from DF, not inferred from rank.",
    }


def report_answers(result) -> dict[str, str]:
    """RQ1–RQ5 and the final challenge, identical to the exported report."""
    return findings(result)


def ir_implications(result) -> str:
    return ir_essay(result)
