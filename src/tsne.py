"""Exact t-SNE in NumPy for a few hundred word vectors (Project 2, part 2).

Follows van der Maaten & Hinton (2008): Gaussian input affinities calibrated to a
target perplexity by binary search, Student-t output affinities, early
exaggeration, momentum and per-parameter gains. Exact O(n^2) gradients are fine
for the <1,000 points plotted here. PCA initialization keeps runs repeatable.
Reference: https://jmlr.org/papers/v9/vandermaaten08a.html
"""

from __future__ import annotations

import numpy as np


def _conditional_affinities(distances: np.ndarray, perplexity: float, tolerance: float = 1e-5) -> np.ndarray:
    """Row-wise Gaussian P(j|i) whose entropy matches log(perplexity)."""
    n = len(distances)
    target = np.log(perplexity)
    result = np.zeros((n, n))
    for i in range(n):
        row = np.delete(distances[i], i)
        beta, low, high = 1.0, 0.0, np.inf
        for _ in range(100):
            weights = np.exp(-(row - row.min()) * beta)
            total = weights.sum()
            probabilities = weights / total
            entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            if abs(entropy - target) < tolerance:
                break
            if entropy > target:  # too flat: narrow the Gaussian
                low = beta
                beta = beta * 2 if high == np.inf else (beta + high) / 2
            else:
                high = beta
                beta = (beta + low) / 2
        result[i, np.arange(n) != i] = probabilities
    return result


def tsne_2d(vectors: np.ndarray, *, perplexity: float = 30.0, iterations: int = 1000,
            learning_rate: float | None = None, exaggeration: float = 12.0, seed: int = 42) -> tuple[np.ndarray, float]:
    """Embed L2-normalized vectors in 2-D; return coordinates and the final KL divergence."""
    values = np.asarray(vectors, dtype=np.float64)
    n = len(values)
    if values.ndim != 2 or n < 4 or not np.isfinite(values).all():
        raise ValueError("t-SNE requires a finite matrix with at least four rows.")
    if not 1 <= perplexity < n:
        raise ValueError("perplexity must be at least 1 and smaller than the number of points.")
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    squared = np.sum(normalized ** 2, axis=1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * normalized @ normalized.T, 0)
    conditional = _conditional_affinities(distances, perplexity)
    p = np.maximum((conditional + conditional.T) / (2 * n), 1e-12)

    centered = normalized - normalized.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    y = centered @ components[:2].T
    y = y / np.std(y[:, 0]) * 1e-4  # PCA initialization, rescaled as recommended for t-SNE
    rng = np.random.default_rng(seed)
    y += rng.normal(0, 1e-6, y.shape)  # breaks exact ties only
    rate = learning_rate or max(n / exaggeration / 4, 50.0)
    velocity, gains = np.zeros_like(y), np.ones_like(y)
    for iteration in range(iterations):
        early = iteration < 250
        scale = exaggeration if early else 1.0
        momentum = 0.5 if early else 0.8
        diff = y[:, None, :] - y[None, :, :]
        inverse = 1 / (1 + np.sum(diff ** 2, axis=2))
        np.fill_diagonal(inverse, 0)
        q = np.maximum(inverse / inverse.sum(), 1e-12)
        gradient = 4 * np.einsum("ij,ijk->ik", (scale * p - q) * inverse, diff)
        same_sign = np.sign(gradient) == np.sign(velocity)
        gains = np.maximum(np.where(same_sign, gains * 0.8, gains + 0.2), 0.01)
        velocity = momentum * velocity - rate * gains * gradient
        y = y + velocity
        y -= y.mean(axis=0)
    kl = float(np.sum(p * np.log(p / q)))
    return y, kl
