"""Checks for the NumPy t-SNE and the Project 2 part 2 embedding measurements."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedding_study as study
from src.tsne import _conditional_affinities, tsne_2d
from src.word2vec import Word2VecModel


def toy_model(vectors, words=None, counts=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    words = words or [f"w{i}" for i in range(len(vectors))]
    counts = np.asarray(counts if counts is not None else [10] * len(vectors))
    return Word2VecModel(words, counts, vectors, np.zeros_like(vectors))


class TsneTests(unittest.TestCase):
    def test_affinities_match_target_perplexity(self):
        points = np.random.default_rng(1).normal(size=(40, 5))
        distances = np.sum((points[:, None] - points[None]) ** 2, axis=2)
        p = _conditional_affinities(distances, perplexity=10)
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertTrue(np.allclose(np.diag(p), 0))
        entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), axis=1)
        self.assertTrue(np.allclose(np.exp(entropy), 10, rtol=1e-3))

    def test_separated_clusters_stay_separated_and_runs_repeat(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(3, 12)) * 4
        points = np.vstack([c + rng.normal(size=(30, 12)) for c in centers])
        labels = list(np.repeat([0, 1, 2], 30))
        first, kl = tsne_2d(points, perplexity=10, iterations=400)
        second, _ = tsne_2d(points, perplexity=10, iterations=400)
        self.assertEqual(first.shape, (90, 2))
        self.assertTrue(np.isfinite(kl) and kl >= 0)
        self.assertTrue(np.allclose(first, second))
        self.assertGreater(study.group_purity(first, labels), 0.95)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            tsne_2d(np.ones((3, 4)))
        with self.assertRaises(ValueError):
            tsne_2d(np.random.default_rng(0).normal(size=(10, 3)), perplexity=10)


class MeasurementTests(unittest.TestCase):
    def test_cosine_and_missing_words(self):
        model = toy_model([[1, 0], [0, 1], [1, 1]], ["a", "b", "c"])
        self.assertAlmostEqual(study.cosine(model, "a", "b"), 0.0)
        self.assertAlmostEqual(study.cosine(model, "a", "c"), 1 / np.sqrt(2), places=6)
        self.assertIsNone(study.cosine(model, "a", "zzz"))

    def test_group_cosines_margin(self):
        model = toy_model([[1, 0.1], [1, -0.1], [0.1, 1], [-0.1, 1]], ["a1", "a2", "b1", "b2"])
        rows = {r["group"]: r for r in study.group_cosines(model, {"A": ["a1", "a2"], "B": ["b1", "b2", "missing"]})}
        self.assertGreater(rows["A"]["within_mean_cosine"], 0.95)
        self.assertLess(rows["A"]["between_mean_cosine"], 0.2)
        self.assertEqual(rows["B"]["words"], 2)

    def test_neighbor_ids_excludes_self_and_orders_by_cosine(self):
        model = toy_model([[1, 0], [0.9, 0.1], [0, 1], [0.5, 0.5]])
        neighbors = study.neighbor_ids(model, top_n=2, chunk=3)
        self.assertEqual(neighbors[0].tolist(), [1, 3])
        self.assertTrue(all(i not in row for i, row in enumerate(neighbors)))

    def test_overlap_is_one_for_identical_models_and_banded(self):
        vectors = np.random.default_rng(2).normal(size=(30, 6))
        counts = [3] * 10 + [20] * 10 + [300] * 10
        model = toy_model(vectors, counts=counts)
        rows = {r["frequency_band"]: r for r in study.neighbor_overlap(model, model, top_n=5)}
        self.assertEqual(rows["all"]["words"], 30)
        self.assertAlmostEqual(rows["all"]["mean_overlap"], 1.0)
        self.assertEqual(rows["2-4"]["words"], 10)
        self.assertIsNone(rows["5-9"]["mean_overlap"])

    def test_purity_in_cosine_and_plane(self):
        labels = ["x", "x", "y", "y"]
        self.assertEqual(study.group_purity(np.array([[0, 0], [0, 1], [9, 9], [9, 10]]), labels, k=1), 1.0)
        self.assertEqual(study.group_purity(np.array([[1, 0], [2, 0.1], [0, 1], [0.1, 3]]), labels, k=1, cosine_space=True), 1.0)

    def test_random_baseline_excludes_self_pairs(self):
        model = toy_model(np.random.default_rng(3).normal(size=(50, 8)))
        baseline = study.random_pair_baseline(model, samples=500)
        self.assertTrue(np.all(baseline < 0.999))
        self.assertEqual(study.percentile(10.0, baseline), 100.0)

    def test_cooccurrence_share_and_chance_purity(self):
        model = toy_model([[1, 0], [0.99, 0.1], [0, 1]], ["a", "b", "c"], counts=[3, 3, 300])
        rows = {r["frequency_band"]: r for r in study.cooccurrence_share(model, [{"a", "b"}, {"c"}], top_n=1)}
        self.assertEqual(rows["2-4"]["words"], 2)
        self.assertAlmostEqual(rows["2-4"]["mean_share"], 1.0)  # a and b are each other's neighbor and share doc 0
        self.assertAlmostEqual(rows[">=200"]["mean_share"], 0.0)
        self.assertAlmostEqual(study.chance_purity(["x", "x", "y", "y"]), 1 / 3)

    def test_experiments_change_one_parameter(self):
        base = study.EXPERIMENTS["main"]
        self.assertEqual(base.dimensions, 100)
        for name, config in study.EXPERIMENTS.items():
            changed = [f for f in ("dimensions", "window", "min_count", "seed") if getattr(config, f) != getattr(base, f)]
            self.assertLessEqual(len(changed), 1, name)
        self.assertGreaterEqual(len(study.QUERIES), 10)
        self.assertGreaterEqual(len(study.GROUPS), 5)


if __name__ == "__main__":
    unittest.main()
