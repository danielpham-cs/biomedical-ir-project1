"""Optional challenge: compare Zipf statistics of two equally sized domain corpora."""
from __future__ import annotations

from collections import Counter

from .zipf_experiment import AbstractCorpus, CONDITIONS, fit_zipf, region_fits, tokens


def domain_profile(corpus: AbstractCorpus, size: int, condition: str, stops=None) -> dict:
    """Frequency profile of the first `size` abstracts (manifest order, not a new sample)."""
    if condition not in CONDITIONS:
        raise ValueError("Condition must be A, B, C or D.")
    if not 1 <= size <= len(corpus.documents):
        raise ValueError(f"Choose 1 to {len(corpus.documents)} documents.")
    documents = corpus.documents[:size]
    cf, df = Counter(), Counter()
    for doc in documents:
        counts = Counter(tokens(doc.abstract, condition, stops))
        cf.update(counts)
        df.update(counts.keys())
    total = sum(cf.values())
    rows = [{"rank": rank, "term": term, "CF": frequency, "DF": df[term]}
            for rank, (term, frequency) in enumerate(sorted(cf.items(), key=lambda x: (-x[1], x[0])), 1)]
    fit = fit_zipf([r["rank"] for r in rows], [r["CF"] for r in rows])
    return {"query": corpus.query, "fingerprint": corpus.fingerprint, "documents": size, "tokens": total,
            "vocabulary": len(cf), "average_tokens_per_document": total / size,
            "hapax": sum(v == 1 for v in cf.values()), "hapax_share": sum(v == 1 for v in cf.values()) / len(cf),
            "zipf_exponent": fit["zipf_exponent"], "slope": fit["slope"], "intercept": fit["intercept"],
            "r_squared": fit["r_squared"], "rmse_log10": fit["rmse_log10"],
            "regions": region_fits(rows), "top_terms": [r["term"] for r in rows[:10]], "rows": rows}


def compare_domains(corpus_a: AbstractCorpus, corpus_b: AbstractCorpus, *, condition: str = "C",
                    size: int | None = None, stops=None) -> dict:
    """Profiles of two domains at equal size, plus vocabulary overlap and distinctive terms."""
    size = min(len(corpus_a.documents), len(corpus_b.documents)) if size is None else size
    a, b = (domain_profile(c, size, condition, stops) for c in (corpus_a, corpus_b))
    terms_a, terms_b = ({r["term"] for r in p["rows"]} for p in (a, b))
    shared = terms_a & terms_b
    top_a, top_b = set(a["top_terms"]), set(b["top_terms"])

    def distinctive(profile, other_terms):
        return [r["term"] for r in profile["rows"] if r["term"] not in other_terms][:10]

    return {"condition": condition, "size": size, "A": a, "B": b,
            "shared_vocabulary": len(shared), "jaccard": len(shared) / len(terms_a | terms_b),
            "shared_top10": sorted(top_a & top_b, key=a["top_terms"].index),
            "distinctive_A": distinctive(a, terms_b), "distinctive_B": distinctive(b, terms_a)}


def domain_answer(comparison: dict, names=("A", "B")) -> str:
    """Data-grounded answer to 'Does domain specificity affect the Zipf distribution?'."""
    a, b = comparison["A"], comparison["B"]
    na, nb = names
    k_gap = abs(a["zipf_exponent"] - b["zipf_exponent"])
    return (
        f"With {comparison['size']:,} abstracts per domain under condition {comparison['condition']}, both rank-frequency "
        f"curves remain Zipf-like: k = {a['zipf_exponent']:.3f} (R² {a['r_squared']:.3f}) for {na} and "
        f"k = {b['zipf_exponent']:.3f} (R² {b['r_squared']:.3f}) for {nb}, a difference of {k_gap:.3f}. "
        f"The overall shape is therefore similar, but the details are domain-specific. {na} has {a['vocabulary']:,} distinct terms "
        f"versus {b['vocabulary']:,} for {nb} ({a['hapax_share']:.1%} vs {b['hapax_share']:.1%} hapax), and only "
        f"{comparison['jaccard']:.1%} of the combined vocabulary is shared. The top-10 lists share "
        f"{len(comparison['shared_top10'])} terms ({', '.join(comparison['shared_top10']) or 'none'}); the other head terms "
        f"name each domain's subject ({', '.join(t for t in a['top_terms'] if t not in comparison['shared_top10'])} "
        f"vs {', '.join(t for t in b['top_terms'] if t not in comparison['shared_top10'])}). "
        "Domain specificity thus changes which terms occupy the head and how large the tail is, while the heavy-tailed "
        "form persists. Two single-topic samples cannot establish a general effect size; the exponent also depends on "
        "preprocessing and the fitted rank range."
    )
