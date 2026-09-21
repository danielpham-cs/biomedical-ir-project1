"""Regression checks for the full-text, preprocessing, and EOS requirements."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import import_xml, parse_importable_xml
from src.eos import split_sentences
from src.index import build_index, load_index, rebuild_local_index, save_docs, save_index
from src.parse_pmc import parse_xml_text, parse_directory
from src.preprocess import preprocess
from src.search import build_spelling_vocabulary, highlight_text, search, suggest_query_correction
from src.stats import compute_stats


def pmc_xml(body: str, pmid: str = "777") -> bytes:
    return (f'<article><front><article-meta><article-id pub-id-type="pmid">{pmid}</article-id>'
            '<article-id pub-id-type="pmc">888</article-id><title-group><article-title>Example study</article-title>'
            '</title-group><abstract><p>An abstract.</p></abstract></article-meta></front>'
            f'<body>{body}</body></article>').encode()


class HomeworkTests(unittest.TestCase):
    def test_result_highlight_uses_the_same_stem_as_retrieval(self):
        doc = parse_xml_text(pmc_xml("<p>Testing was repeated; tested samples passed.</p>"))
        index = build_index([doc], stem=True, remove_stopwords=False)
        self.assertEqual(
            highlight_text("Testing was repeated; tested samples passed.", ["test"], index),
            "<mark>Testing</mark> was repeated; <mark>tested</mark> samples passed.",
        )

    def test_english_typo_suggestion_uses_transposition_distance_and_frequency(self):
        doc = parse_xml_text(pmc_xml("<p>Diabetes treatment is effective. Diabetes care continues.</p>"))
        index = build_index([doc], stem=True, remove_stopwords=False)
        vocabulary = build_spelling_vocabulary([doc])
        self.assertEqual(suggest_query_correction("diabtes treatment", index, vocabulary), "diabetes treatment")
        self.assertIsNone(suggest_query_correction("糖尿病治療", index, vocabulary))

    def test_snippet_shows_each_matched_or_term_when_terms_are_far_apart(self):
        text = "Diabetes was recorded. " + ("intervening evidence " * 40) + "Insulin was prescribed."
        doc = parse_xml_text(pmc_xml(f"<p>{text}</p>"))
        index = build_index([doc], stem=True, remove_stopwords=False)
        hit = search("diabetes insulin", index, [doc], method="OR")[0]
        self.assertIn("Diabetes", hit.snippet)
        self.assertIn("Insulin", hit.snippet)

    def test_terms_after_20000_characters_are_searchable(self):
        body = "common evidence " * 2000 + "xylophosphatase"
        doc = parse_xml_text(pmc_xml(f"<p>{body}</p>"))
        self.assertEqual(doc.body, body)
        index = build_index([doc], stem=False, remove_stopwords=False)
        hits = search("xylophosphatase", index, [doc])
        self.assertEqual(len(hits), 1)
        self.assertIn("xylophosphatase", hits[0].snippet)
        self.assertGreater(compute_stats(doc.full_text).num_characters, 20000)

    def test_mixed_content_order_and_block_separators(self):
        doc = parse_xml_text(pmc_xml('<p>Before <italic>inside <bold>deep</bold> middle</italic> after.</p><p>Next paragraph.</p>'))
        self.assertEqual(doc.body, "Before inside deep middle after. Next paragraph.")

    def test_section_headings_and_table_text_are_indexed(self):
        doc = parse_xml_text(pmc_xml('<sec><title>Neuroinflammation</title><p>Results.</p>'
                                     '<table><tr><td>Biomarker</td><td>TNF</td></tr></table></sec>'))
        index = build_index([doc], stem=False, remove_stopwords=False)
        self.assertTrue(search("neuroinflammation tnf", index, [doc], method="boolean"))

    @patch("src.preprocess._stop_words", return_value=frozenset({"the", "a"}))
    def test_type_one_and_type_two_are_distinct(self, _):
        one, two = preprocess("the type 1 diabetes"), preprocess("the type 2 diabetes")
        self.assertIn("1", one)
        self.assertIn("2", two)
        self.assertNotEqual(one, two)
        self.assertEqual(preprocess("a t 2", remove_stopwords=False, stem=False), ["a", "t", "2"])

    def test_eos_initials_decimals_quotes_and_lowercase(self):
        text = 'Dr. Smith saw 3.14. J. Lee agreed. "Good!" p53 changed.'
        self.assertEqual(split_sentences(text), ['Dr. Smith saw 3.14.', 'J. Lee agreed.', '"Good!"', 'p53 changed.'])
        self.assertEqual(split_sentences("See Fig. 2 for details. Next result."), ["See Fig. 2 for details.", "Next result."])
        self.assertEqual(split_sentences(""), [])

    def test_import_validation_and_safe_filenames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = import_xml(pmc_xml("<p>Safe content.</p>", pmid="../../escape"), root)
            self.assertEqual(path.parent, root)
            self.assertEqual(path.name, "PMC888.xml")
            self.assertEqual(len(parse_directory(root)), 1)
            for invalid in [b"not XML", b"<html><title>Wrong format</title></html>",
                            b"<pmc-articleset><article/><article/></pmc-articleset>"]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    import_xml(invalid, root)

    def test_upload_can_be_previewed_without_writing_the_xml(self):
        content = pmc_xml("<p>Preview before saving.</p>", pmid="12345")
        document = parse_importable_xml(content)
        self.assertEqual(document.pmid, "12345")
        self.assertEqual(document.abstract, "An abstract.")

    @patch("src.preprocess._stop_words", return_value=frozenset({"the"}))
    def test_rebuild_applies_selected_settings(self, _):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, processed = root / "raw", root / "processed"
            import_xml(pmc_xml("<p>The treatments worked.</p>"), raw)
            rebuild_local_index(raw, processed, stem=False, remove_stopwords=False)
            first = load_index(processed / "index.pkl")
            self.assertIn("the", first.postings)
            rebuild_local_index(raw, processed, stem=True, remove_stopwords=True)
            second = load_index(processed / "index.pkl")
            self.assertTrue(second.stem)
            self.assertTrue(second.remove_stopwords)
            self.assertNotIn("the", second.postings)


if __name__ == "__main__":
    unittest.main()
