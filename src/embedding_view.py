"""Deterministic PCA and display rows for the trained Word2Vec vocabulary."""

from __future__ import annotations

import numpy as np


def selected_map_word(selection) -> str | None:
    """Read Vega-Lite's field-array selection (or selected-row payload)."""
    if isinstance(selection, dict):
        words = selection.get("word", [])
        if isinstance(words, str):
            return words
        return words[0] if isinstance(words, list) and words else None
    if isinstance(selection, list) and selection and isinstance(selection[0], dict):
        return selection[0].get("word")
    return None


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit two components on ALL L2-normalized, mean-centered word vectors.

    Return coordinates and explained-variance fractions. Changing the plotted
    subset does not refit PCA. A constant or rank-one input has zero spare axes.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("PCA requires a nonempty, finite matrix of word vectors.")
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    coordinates, ratios = np.zeros((len(values), 2)), np.zeros(2)
    total = float(np.sum(centered ** 2))
    if total <= np.finfo(float).eps * max(values.shape):
        return coordinates, ratios
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    width = min(2, len(singular))
    axes = components[:width].copy()
    # Resolve SVD sign ambiguity for repeatable labels and exported coordinates.
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    coordinates[:, :width] = centered @ axes.T
    ratios[:width] = singular[:width] ** 2 / total
    return coordinates, ratios


def projection_rows(model, coordinates: np.ndarray, *, query: str | None = None,
                    mode: str = "neighbors", limit: int = 40) -> list[dict]:
    """Select points after projection; cosine values always use original vectors."""
    if mode not in {"neighbors", "frequent"} or limit < 1:
        raise ValueError("Choose neighbors or frequent, and a positive point limit.")
    if coordinates.shape != (len(model.words), 2):
        raise ValueError("Coordinates must align with the complete model vocabulary.")
    if query is not None and query not in model.words:
        raise ValueError("The query word is outside the trained vocabulary.")
    if mode == "neighbors" and query is None:
        raise ValueError("Choose a vocabulary word to show its neighbors.")
    query_id = model.words.index(query) if query is not None else None
    similarities = None
    if query_id is not None:
        unit = model.vectors / np.maximum(np.linalg.norm(model.vectors, axis=1, keepdims=True), 1e-12)
        similarities = unit @ unit[query_id]
    ordered = (np.argsort(-similarities, kind="stable") if mode == "neighbors"
               else np.argsort(-model.counts, kind="stable"))
    selected = ([] if query_id is None else [query_id]) + [int(i) for i in ordered if i != query_id]
    return [{"word": model.words[i], "pc1": float(coordinates[i, 0]), "pc2": float(coordinates[i, 1]),
             "corpus_count": int(model.counts[i]),
             "cosine_similarity": float(similarities[i]) if similarities is not None else None,
             "group": "Query" if i == query_id else ("Neighbor" if mode == "neighbors" else "Frequent word")}
            for i in selected[:limit]]
