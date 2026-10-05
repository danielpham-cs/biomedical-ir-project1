"""Correctness checks for the local Porter stemmer and Skip-gram embeddings."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.porter import form, measure, porter_stem, porter_steps
from src.matching import edit_distance, edit_distance_table
from src.word2vec import (Word2VecConfig, Word2VecModel, cbow_examples, context_window, model_key, pair_scores,
                          skipgram_pairs, train_skipgram)


class PorterTests(unittest.TestCase):
    def test_measure_and_original_rules_against_independent_reference(self):
        for word, expected in {"tree": 0, "trouble": 1, "oats": 1, "ivy": 1, "troubles": 2, "private": 2, "oaten": 2}.items():
            self.assertEqual(measure(word), expected)
        # Examples cover every original suffix rule, failures, Y and short words.
        words = """caresses ponies ties caress cats feed agreed plastered bled motoring
        sing conflated troubled sized hopping tanned falling hissing fizzed failing
        filing happy sky relational conditional rational valenci hesitanci digitizer
        conformabli radicalli differentli vileli analogousli vietnamization predication
        operator feudalism decisiveness hopefulness callousness formaliti sensitiviti
        sensibiliti triplicate formative formalize electriciti electrical hopeful
        goodness revival allowance inference airliner gyroscopic adjustable defensible
        irritant replacement adjustment dependent adoption homologou communism activate
        angulariti homologous effective bowdlerize probate rate cease controll roll
        generalizations oscillators y yy by cry toy syzygy a i s as is""".split()
        reference = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
        for word in words:
            with self.subTest(word=word):
                self.assertEqual(porter_stem(word), reference.stem(word))

    def test_biomedical_identifiers_do_not_lose_their_structure(self):
        for token in ("GLP-1", "HIV-1", "0.65", "24.9%", "BRCA1", "β-cells", "Alzheimer's"):
            self.assertEqual(porter_stem(token), token.lower())
            self.assertEqual(porter_steps(token), [])

    def test_trace_ends_in_the_stem_and_names_each_fired_rule(self):
        for word in ("generalizations", "hopping", "agreed", "relational", "caresses", "sky", "a"):
            with self.subTest(word=word):
                rows = porter_steps(word)
                self.assertEqual([r["step"] for r in rows], ["1a", "1b", "1c", "2", "3", "4", "5a", "5b"])
                self.assertEqual(rows[-1]["after"], porter_stem(word))
                for before, after in zip(rows, rows[1:]):
                    self.assertEqual(before["after"], after["before"])
        rows = porter_steps("generalizations")
        self.assertEqual([r["rule"] for r in rows if r["changed"]], ["S → ∅", "(m>0) IZATION → IZE", "(m>0) ALIZE → AL", "(m>1) AL → ∅"])
        self.assertEqual(form("trouble"), "CVCV")


class EditDistanceTests(unittest.TestCase):
    def test_table_agrees_with_cutoff_version_and_backtrace_costs(self):
        pairs = [("semaglutde", "semaglutide"), ("diabtes", "diabetes"), ("ca", "ac"), ("kitten", "sitting"), ("", "abc"), ("abc", "")]
        for left, right in pairs:
            with self.subTest(pair=(left, right)):
                table, steps = edit_distance_table(left, right)
                self.assertEqual(table[-1][-1], edit_distance(left, right, 10))
                self.assertEqual(sum(s["cost"] for s in steps), table[-1][-1])
                self.assertEqual("".join(s["to"] for s in steps), right)
        self.assertEqual([s["operation"] for s in edit_distance_table("ca", "ac")[1]], ["transpose"])


class Word2VecTests(unittest.TestCase):
    def test_context_window_lists_the_pairs_of_one_center(self):
        rows = context_window(["a", "b", "c", "d", "e"], 2, 1)
        self.assertEqual([(r["center"], r["context"], r["offset"]) for r in rows], [("c", "b", -1), ("c", "d", 1)])
        self.assertEqual(len(context_window(["a", "b"], 0, 5)), 1)
        with self.assertRaises(ValueError):
            context_window(["a"], 3, 1)

    def test_cbow_examples_pad_the_window_of_each_position(self):
        contexts, centers = cbow_examples(np.array([7, 8, 9]), 1)
        self.assertEqual(centers.tolist(), [7, 8, 9])
        self.assertEqual(contexts.tolist(), [[-1, 8], [7, 9], [8, -1]])
        self.assertEqual(cbow_examples(np.array([5]), 2)[0].shape, (0, 4))

    def test_cbow_learns_groups_and_keeps_skipgram_keys(self):
        # 8 topics x 6 words: with a tiny vocabulary, negative samples would mostly be same-topic words.
        rng = np.random.default_rng(0)
        groups = [[f"t{g}w{i}" for i in range(6)] for g in range(8)]
        sentences = [[str(w) for w in rng.permutation(g)] for _ in range(40) for g in groups]
        config = Word2VecConfig(dimensions=16, epochs=8, window=2, min_count=1, sample=0, seed=5, architecture="cbow")
        model = train_skipgram(sentences, config)
        history = model.metadata["history"]
        self.assertLess(history[-1]["probe_loss"], history[0]["probe_loss"])
        self.assertEqual(model.metadata["architecture"], "CBOW with negative sampling")
        same_topic = sum(model.most_similar(w, 1)[0]["word"] in g for g in groups for w in g) / 48
        self.assertGreater(same_topic, 0.9)  # chance is 5/47 ≈ 11%
        np.testing.assert_array_equal(model.vectors, train_skipgram(sentences, config).vectors)
        # Skip-gram keys hash the config without the new field, so saved models stay valid; CBOW gets its own key.
        import hashlib, json
        from dataclasses import asdict
        legacy = {k: v for k, v in asdict(Word2VecConfig()).items() if k != "architecture"}
        payload = {"version": "numpy-sgns-v1", "config": legacy, "corpus": "fp", "preprocessing": {}}
        self.assertEqual(model_key(Word2VecConfig(), "fp", {}),
                         hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        self.assertNotEqual(model_key(Word2VecConfig(architecture="cbow"), "fp", {}), model_key(Word2VecConfig(), "fp", {}))
        with self.assertRaises(ValueError):
            train_skipgram(sentences, Word2VecConfig(architecture="glove"))

    def test_directed_window_pairs_do_not_cross_sentences(self):
        pairs = skipgram_pairs([np.array([0, 1, 2]), np.array([3, 4])], 1)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3)})
        self.assertEqual(skipgram_pairs([np.array([0])], 5).shape, (0, 2))

    def test_training_learns_is_seeded_and_survives_save_load(self):
        sentences = ([["diabetes", "insulin", "glucose", "treatment"]] * 12
                     + [["cancer", "tumor", "oncology", "therapy"]] * 12)
        config = Word2VecConfig(dimensions=12, epochs=8, window=2, min_count=1, sample=0, seed=7)
        model = train_skipgram(sentences, config)
        other = train_skipgram(sentences, config)
        np.testing.assert_array_equal(model.vectors, other.vectors)
        self.assertTrue(np.any(model.context_vectors != 0))
        history = model.metadata["history"]
        self.assertLess(history[-1]["probe_loss"], history[0]["probe_loss"])
        self.assertGreater(model.metadata["training_pairs"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = Word2VecModel.load(path)
            np.testing.assert_array_equal(restored.vectors, model.vectors)
            np.testing.assert_array_equal(restored.context_vectors, model.context_vectors)
            self.assertEqual(restored.metadata, model.metadata)
            self.assertEqual(restored.most_similar("insulin"), model.most_similar("insulin"))
        scores = {r["context"]: r["sigmoid"] for r in pair_scores(model, "insulin", ["glucose", "tumor", "unknown"])}
        self.assertEqual(set(scores), {"glucose", "tumor"})
        self.assertGreater(scores["glucose"], scores["tumor"])

    def test_cosine_neighbors_exclude_query_and_handle_oov(self):
        model = Word2VecModel(["a", "b", "c"], np.array([3, 2, 1]), np.array([[1., 0.], [2., 0.], [0., 1.]]), np.zeros((3, 2)))
        neighbors = model.most_similar("a")
        self.assertEqual([row["word"] for row in neighbors], ["b", "c"])
        self.assertEqual([row["cosine_similarity"] for row in neighbors], [1, 0])
        with self.assertRaisesRegex(ValueError, "not in the trained vocabulary"):
            model.most_similar("unknown")

    def test_invalid_training_and_cache_invalidation(self):
        for config in (Word2VecConfig(window=0), Word2VecConfig(epochs=0), Word2VecConfig(sample=-1), Word2VecConfig(learning_rate=float("nan"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                train_skipgram([["a", "b"]], config)
        for sentences in ([], [["a", "a"]], [["a"], ["b"]]):
            with self.subTest(sentences=sentences), self.assertRaises(ValueError):
                train_skipgram(sentences, Word2VecConfig(min_count=1))
        config = Word2VecConfig()
        original = model_key(config, "corpus-a", {"stem": False})
        self.assertNotEqual(original, model_key(config, "corpus-b", {"stem": False}))
        self.assertNotEqual(original, model_key(config, "corpus-a", {"stem": True}))
        self.assertNotEqual(original, model_key(Word2VecConfig(window=3), "corpus-a", {"stem": False}))


if __name__ == "__main__":
    unittest.main()
