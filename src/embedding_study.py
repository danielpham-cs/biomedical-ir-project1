"""Project 2, part 2: CBOW vs Skip-gram experiments and embedding-space measurements.

Every measurement here uses the original (unprojected) word vectors unless the
function name says otherwise; 2-D projections are only evaluated, never trusted.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .word2vec import Word2VecConfig

# Main setting from the brief: 100 dimensions; every variant changes exactly one parameter.
BASE = Word2VecConfig(dimensions=100, window=5, min_count=2, epochs=15)
EXPERIMENTS = {
    "main": BASE,
    "window2": replace(BASE, window=2),
    "dim50": replace(BASE, dimensions=50),
    "mincount5": replace(BASE, min_count=5),
    "seed7": replace(BASE, seed=7),  # same setting, different initialization/sampling: stability
}
ARCHITECTURES = ("skipgram", "cbow")
LABELS = {"skipgram": "Skip-gram", "cbow": "CBOW"}

# Semantic groups chosen from the GLP-1 domain before looking at any neighbor list.
GROUPS = {
    "GLP-1 drugs": ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "orforglipron", "retatrutide"],
    "Weight / obesity": ["obesity", "overweight", "weight", "bmi", "adiposity", "fat"],
    "Glycemia": ["insulin", "glucose", "glycemic", "hba1c", "hyperglycemia"],
    "GI adverse events": ["nausea", "vomiting", "gastrointestinal", "pancreatitis", "constipation"],
    "Cardiovascular": ["cardiovascular", "cardiac", "heart", "mace", "stroke", "myocardial"],
    "Kidney": ["kidney", "renal", "ckd"],
    "Liver": ["liver", "hepatic", "masld", "mash", "steatosis"],
    "Oncology": ["cancer", "tumor", "carcinoma", "thyroid"],
    "Animal models": ["mice", "rats", "mouse", "rat"],
}
# The brief's own example words plus GLP-1 terms; >= 10 words, high- and low-frequency.
QUERIES = ["glp-1", "semaglutide", "tirzepatide", "liraglutide", "exenatide", "obesity", "insulin",
           "nausea", "cardiovascular", "kidney", "liver", "cancer", "mice", "treatment", "diagnosis"]
RARE_QUERIES = ["lixisenatide", "diarrhea", "nafld", "constipation", "rat"]
PAIRS = [("cancer", "tumor"), ("treatment", "therapy"), ("treatment", "diagnosis"), ("cancer", "treatment"),
         ("semaglutide", "liraglutide"), ("semaglutide", "tirzepatide"), ("obesity", "overweight"),
         ("obesity", "weight"), ("insulin", "glucose"), ("glycemic", "hba1c"), ("nausea", "vomiting"),
         ("kidney", "renal"), ("liver", "hepatic"), ("mice", "rats"), ("cardiovascular", "mace"),
         ("nausea", "kidney"), ("mice", "semaglutide"), ("cancer", "weight")]
FREQUENCY_BANDS = [(2, 4), (5, 9), (10, 49), (50, 199), (200, None)]


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def cosine(model, a: str, b: str) -> float | None:
    if a not in model.words or b not in model.words:
        return None
    u, v = model.vectors[model.words.index(a)], model.vectors[model.words.index(b)]
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))


def random_pair_baseline(model, *, min_count: int = 5, samples: int = 20000, seed: int = 0) -> np.ndarray:
    """Cosines of random distinct word pairs: the reference a 'related' cosine must beat."""
    eligible = np.flatnonzero(model.counts >= min_count)
    rng = np.random.default_rng(seed)
    a, b = rng.choice(eligible, samples), rng.choice(eligible, samples)
    keep = a != b
    vectors = unit(model.vectors)
    return np.einsum("ij,ij->i", vectors[a[keep]], vectors[b[keep]])


def percentile(value: float | None, baseline: np.ndarray) -> float | None:
    return None if value is None else float(np.mean(baseline < value) * 100)


def group_cosines(model, groups: dict = GROUPS) -> list[dict]:
    """Mean cosine inside each group vs. between that group and every other group."""
    vectors = unit(model.vectors)
    present = {name: [model.words.index(w) for w in words if w in model.words] for name, words in groups.items()}
    rows = []
    for name, ids in present.items():
        if len(ids) < 2:
            continue
        inside = vectors[ids] @ vectors[ids].T
        within = float(inside[np.triu_indices(len(ids), 1)].mean())
        others = [i for other, other_ids in present.items() if other != name for i in other_ids]
        between = float((vectors[ids] @ vectors[others].T).mean())
        rows.append({"group": name, "words": len(ids), "within_mean_cosine": within,
                     "between_mean_cosine": between, "margin": within - between})
    return rows


def neighbor_ids(model, top_n: int = 10, chunk: int = 2048) -> np.ndarray:
    """Top-n cosine neighbors of every vocabulary word (self excluded), computed in chunks."""
    vectors = unit(model.vectors).astype(np.float32)
    result = np.empty((len(vectors), top_n), dtype=np.int64)
    for start in range(0, len(vectors), chunk):
        scores = vectors[start:start + chunk] @ vectors.T
        scores[np.arange(len(scores)), np.arange(start, start + len(scores))] = -np.inf
        top = np.argpartition(-scores, top_n, axis=1)[:, :top_n]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        result[start:start + chunk] = np.take_along_axis(top, order, axis=1)
    return result


def neighbor_overlap(first, second, top_n: int = 10, bands=FREQUENCY_BANDS) -> list[dict]:
    """Mean overlap@n of two models' neighbor lists over their shared words, per frequency band."""
    shared = [w for w in first.words if w in set(second.words)]
    index_a, index_b = {w: i for i, w in enumerate(first.words)}, {w: i for i, w in enumerate(second.words)}
    neighbors_a, neighbors_b = neighbor_ids(first, top_n), neighbor_ids(second, top_n)
    overlaps, counts = [], []
    for word in shared:
        a = {first.words[i] for i in neighbors_a[index_a[word]]}
        b = {second.words[i] for i in neighbors_b[index_b[word]]}
        overlaps.append(len(a & b) / top_n)
        counts.append(int(first.counts[index_a[word]]))
    overlaps, counts = np.array(overlaps), np.array(counts)
    rows = []
    for low, high in list(bands) + [(None, None)]:
        mask = np.ones(len(counts), bool) if low is None else (counts >= low) & ((counts <= high) if high else True)
        label = "all" if low is None else (f"{low}-{high}" if high else f">={low}")
        rows.append({"frequency_band": label, "words": int(mask.sum()),
                     "mean_overlap": float(overlaps[mask].mean()) if mask.any() else None})
    return rows


def cooccurrence_share(model, documents: list[set[str]], top_n: int = 10, bands=FREQUENCY_BANDS) -> list[dict]:
    """Share of each word's top-n neighbors that appear in at least one abstract together with it.

    A low share means a neighbor was learned from similar contexts in different
    abstracts, not merely memorized from one shared document.
    """
    index = {w: i for i, w in enumerate(model.words)}
    postings: dict[int, set[int]] = {}
    for doc_id, words in enumerate(documents):
        for word in words:
            if word in index:
                postings.setdefault(index[word], set()).add(doc_id)
    neighbors = neighbor_ids(model, top_n)
    shares = np.array([np.mean([bool(postings.get(i, set()) & postings.get(int(j), set())) for j in neighbors[i]])
                       for i in range(len(model.words))])
    rows = []
    for low, high in bands:
        mask = (model.counts >= low) & ((model.counts <= high) if high else True)
        rows.append({"frequency_band": f"{low}-{high}" if high else f">={low}", "words": int(mask.sum()),
                     "mean_share": float(shares[mask].mean()) if mask.any() else None})
    return rows


def chance_purity(labels: list[str], k: int = 3) -> float:
    """Expected group purity if the k nearest labeled words were drawn at random."""
    labels = np.array(labels)
    sizes = {g: int(np.sum(labels == g)) for g in set(labels)}
    return float(np.mean([(sizes[g] - 1) / (len(labels) - 1) for g in labels]))


def labeled_words(model, groups: dict = GROUPS) -> list[tuple[str, str]]:
    return [(word, name) for name, words in groups.items() for word in words if word in model.words]


def group_purity(points: np.ndarray, labels: list[str], k: int = 3, *, cosine_space: bool = False) -> float:
    """Share of each labeled word's k nearest labeled words that belong to its own group."""
    values = unit(points) if cosine_space else np.asarray(points, float)
    if cosine_space:
        distances = -(values @ values.T)
    else:
        distances = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    labels = np.array(labels)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return float(np.mean(labels[nearest] == labels[:, None]))


def projection_subset(model, frequent: int = 300) -> list[int]:
    """Indices plotted/embedded in 2-D: the most frequent words plus every labeled group word."""
    ids = list(np.argsort(-model.counts, kind="stable")[:frequent])
    for word, _ in labeled_words(model):
        index = model.words.index(word)
        if index not in ids:
            ids.append(index)
    return [int(i) for i in ids]
