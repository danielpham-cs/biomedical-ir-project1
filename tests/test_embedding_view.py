"""Numerical properties of PCA and original-space neighbor selection."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding_view import pca_2d, projection_rows, selected_map_word
from src.word2vec import Word2VecModel


class EmbeddingViewTests(unittest.TestCase):
    def test_chart_selection_decodes_field_arrays_and_reset(self):
        self.assertEqual(selected_map_word({"word": ["insulin"], "vlPoint": {"or": [{"word": "insulin"}]}}), "insulin")
        self.assertEqual(selected_map_word([{"word": "insulin"}]), "insulin")
        for selection in ({}, {"word": []}, [], None):
            self.assertIsNone(selected_map_word(selection))

    def test_two_dimensional_projection_preserves_normalized_distances(self):
        vectors = np.array([[4., 0.], [1., 2.], [-3., 1.], [0., -2.]])
        points, ratios = pca_2d(vectors)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        np.testing.assert_allclose(np.linalg.norm(points[:, None] - points[None, :], axis=2),
                                   np.linalg.norm(unit[:, None] - unit[None, :], axis=2), atol=1e-12)
        np.testing.assert_allclose(points.mean(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(float(sum(ratios)), 1)
        self.assertGreaterEqual(ratios[0], ratios[1])

    def test_rank_one_and_degenerate_matrices_are_finite(self):
        points, ratios = pca_2d(np.array([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]))
        np.testing.assert_allclose(ratios, [1, 0])
        np.testing.assert_allclose(points[:, 1], 0)
        for matrix in (np.zeros((3, 4)), np.ones((3, 4)), np.ones((1, 4))):
            points, ratios = pca_2d(matrix)
            np.testing.assert_array_equal(points, np.zeros((len(matrix), 2)))
            np.testing.assert_array_equal(ratios, [0, 0])

    def test_invalid_pca_input(self):
        for value in (np.array([1, 2]), np.empty((0, 2)), np.array([[np.nan, 0]]), np.array([[np.inf, 1]])):
            with self.assertRaises(ValueError):
                pca_2d(value)

    def test_point_selection_matches_cosine_neighbors_and_keeps_coordinates(self):
        vectors = np.array([[1., 0., 0.], [.9, .1, 0.], [-1., 0., 0.], [0., 1., .5]])
        model = Word2VecModel(["diabetes", "insulin", "cancer", "care"], np.array([10, 8, 40, 20]), vectors, vectors)
        coordinates, _ = pca_2d(vectors)
        close = projection_rows(model, coordinates, query="diabetes", limit=3)
        self.assertEqual([row["word"] for row in close[1:]], [row["word"] for row in model.most_similar("diabetes", 2)])
        frequent = projection_rows(model, coordinates, query="diabetes", mode="frequent", limit=3)
        self.assertEqual([row["word"] for row in frequent], ["diabetes", "cancer", "care"])
        for row in close + frequent:
            i = model.words.index(row["word"])
            self.assertEqual(row["corpus_count"], model.counts[i])
            np.testing.assert_allclose([row["pc1"], row["pc2"]], coordinates[i])
        self.assertEqual(len(projection_rows(model, coordinates, query="insulin", limit=100)), 4)
        global_map = projection_rows(model, coordinates, mode="frequent", limit=2)
        self.assertEqual([row["word"] for row in global_map], ["cancer", "care"])
        self.assertTrue(all(row["cosine_similarity"] is None for row in global_map))
        with self.assertRaises(ValueError):
            projection_rows(model, coordinates, query="missing")


if __name__ == "__main__":
    unittest.main()
