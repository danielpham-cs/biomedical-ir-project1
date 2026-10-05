"""Correctness checks for corpus selection, Porter, Zipf and learned embeddings."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from nltk.stem import PorterStemmer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.homework2 import (ExperimentCorpus, compare_stemming, frequency_rows,
                           frequency_spectrum, load_experiment_corpus, training_sentences)
from src.parse_pmc import Document
from src.porter import measure, porter_stem
from src.word2vec import Word2VecConfig, Word2VecModel, model_key, skipgram_pairs, train_skipgram


def document(pmid, text):
    return Document(pmid, pmid, "", "Excluded title", text, "Excluded body", "", "2026")


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


class DistributionTests(unittest.TestCase):
    def test_frequency_rank_ties_and_spectrum(self):
        counts = Counter({"beta": 2, "alpha": 2, "gamma": 1})
        rows = frequency_rows(counts)
        self.assertEqual([row["term"] for row in rows], ["alpha", "beta", "gamma"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertAlmostEqual(sum(row["relative_frequency"] for row in rows), 1)
        self.assertEqual(frequency_spectrum(counts), [{"frequency": 1, "terms": 1}, {"frequency": 2, "terms": 2}])
        self.assertEqual(frequency_rows(Counter()), [])

    def test_stemming_conserves_counts_and_aggregates_entire_corpus(self):
        corpus = ExperimentCorpus([document("1", "Testing tested.\n\nKeywords: tests"), document("2", "Testing insulin.")], "test", "abc")
        result = compare_stemming(corpus, remove_stopwords=False)
        self.assertEqual(result["stats_before"]["tokens"], 5)
        self.assertEqual(result["stats_after"]["tokens"], 5)
        self.assertEqual(result["stats_before"]["vocabulary"], 4)
        self.assertEqual(result["stats_after"]["vocabulary"], 2)
        self.assertEqual(result["after"][0]["term"], "test")
        self.assertEqual(result["after"][0]["frequency"], 4)
        self.assertEqual(result["vocabulary_reduction_percent"], 50)
        self.assertEqual(result["merged_groups"][0]["form_count"], 3)

    @patch("src.preprocess._stop_words", return_value=frozenset({"the"}))
    def test_same_stopword_filter_and_empty_input(self, _):
        corpus = ExperimentCorpus([document("1", "The testing tests")], "test", "abc")
        result = compare_stemming(corpus, remove_stopwords=True)
        self.assertEqual(result["stats_before"]["tokens"], 2)
        self.assertEqual(result["stats_after"]["tokens"], 2)
        with self.assertRaises(ValueError):
            compare_stemming(ExperimentCorpus([document("1", "the")], "test", "empty"))

    def test_sentences_preserve_document_boundaries_and_ignore_body_title(self):
        corpus = ExperimentCorpus([document("1", "Alpha beta. Gamma delta."), document("2", "Epsilon zeta.")], "test", "abc")
        sentences = training_sentences(corpus, remove_stopwords=False)
        self.assertEqual(sentences, [["alpha", "beta"], ["gamma", "delta"], ["epsilon", "zeta"]])

    def test_manifest_isolates_subject_checks_sources_and_fingerprints_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"pmid": str(i), "path": "obsolete/untrusted.xml"} for i in range(1, 11)]
            path = root / "pubmed_abstract_meta.json"
            path.write_text(json.dumps({"query": "diabetes", "saved": records}))
            def xml(i, text):
                return f'<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article><ArticleTitle>Study</ArticleTitle><Abstract><AbstractText>{text}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>'
            for i in range(1, 12):
                (root / f"PMID{i}.xml").write_text(xml(i, "Diabetes treatment."))
            first = load_experiment_corpus(path, 10)
            self.assertEqual(len(first.documents), 10)
            self.assertNotIn("11", first.metadata()["pmids"])
            self.assertEqual(first.fingerprint, load_experiment_corpus(path, 10).fingerprint)
            (root / "PMID1.xml").write_text(xml(1, "Different diabetes treatment."))
            self.assertNotEqual(first.fingerprint, load_experiment_corpus(path, 10).fingerprint)
            with self.assertRaises(ValueError):
                load_experiment_corpus(path, 11)
            (root / "PMID1.xml").unlink()
            with self.assertRaises(FileNotFoundError):
                load_experiment_corpus(path, 10)
            records[0]["pmid"] = "../escape"
            path.write_text(json.dumps({"query": "diabetes", "saved": records}))
            with self.assertRaises(ValueError):
                load_experiment_corpus(path, 10)


class Word2VecTests(unittest.TestCase):
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
