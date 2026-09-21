"""Regression tests for the final unified statistics and retrieval specification."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import compose_abstract, split_sentences, tokenize_words
from src.index import build_index
from src.parse_pmc import Document, parse_xml_text
from src.search import search
from src.stats import compute_stats


class EngineSpecificationTests(unittest.TestCase):
    def test_characters_collapse_whitespace_and_report_no_space(self):
        stats = compute_stats(" risk\n\nassessment\t")
        self.assertEqual(stats.num_characters, 15)
        self.assertEqual(stats.characters_no_space, 14)

    def test_final_word_token_rules(self):
        text = "0.65 24.9% 12-15% 0.44–0.97 HIV-1 5-stage Alzheimer's U.S. Brca1ΔC/ΔC factors—particularly"
        self.assertEqual(
            tokenize_words(text),
            ["0.65", "24.9%", "12", "15%", "0.44", "0.97", "hiv-1", "5-stage",
             "alzheimer's", "u.s.", "brca1δc", "δc", "factors", "particularly"],
        )

    def test_keywords_are_retained_as_a_final_abstract_line(self):
        abstract = compose_abstract(["Results are stable."], ["diabetes", "insulin"])
        self.assertEqual(abstract, "Results are stable.\n\nKeywords: diabetes, insulin")
        self.assertEqual(split_sentences(abstract), ["Results are stable.", "Keywords: diabetes, insulin"])

    def test_statistics_include_final_keywords_line(self):
        abstract = compose_abstract(["Results are stable."], ["diabetes", "insulin"])
        stats = compute_stats(abstract)
        self.assertEqual(stats.num_words, 6)
        self.assertEqual(stats.num_sentences, 2)

    def test_eos_keeps_multiword_abbreviation_inside_sentence(self):
        self.assertEqual(
            split_sentences("Lee et al. reported improvement. The follow-up confirmed it."),
            ["Lee et al. reported improvement.", "The follow-up confirmed it."],
        )

    def test_pmc_uses_normal_abstract_and_article_keywords(self):
        xml = b'''<article><front><article-meta><title-group><article-title>Title excluded</article-title></title-group>
        <abstract abstract-type="graphical"><p>Do not include.</p></abstract><abstract><title>Methods</title><p>Main text.</p></abstract>
        <kwd-group><kwd>alpha</kwd><kwd>beta</kwd></kwd-group></article-meta></front><body><p>Body evidence.</p></body></article>'''
        document = parse_xml_text(xml)
        self.assertEqual(document.abstract, "Main text.\n\nKeywords: alpha, beta")
        self.assertNotIn("Title excluded", document.search_text)
        self.assertIn("Body evidence", document.search_text)

    def test_phrase_stays_in_one_paragraph_and_bm25_ranks_candidates(self):
        first = Document("1", "1", "12345", "Title", "insulin treatment", "improved insulin treatment", "", "")
        second = Document("2", "2", "", "Title", "insulin", "treatment only", "", "")
        index = build_index([first, second])
        hits = search('"insulin treatment"', index, [first, second], method="phrase")
        self.assertEqual([hit.doc_id for hit in hits], ["1"])
        self.assertEqual(hits[0].pmid, "1")
        self.assertEqual(hits[0].pmcid, "12345")
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(hits[0].term_counts["insulin"]["abstract"], 1)

    def test_phrase_snippet_shows_the_actual_match_in_keywords(self):
        document = Document(
            "1", "1", "", "Title",
            "Cancer was discussed separately.\n\nKeywords: prostate cancer, diabetes",
            "", "", "",
        )
        index = build_index([document])
        hits = search("cancers diabetes", index, [document], method="PHRASE")
        self.assertEqual(len(hits), 1)
        self.assertIn("Keywords: prostate cancer, diabetes", hits[0].snippet)


if __name__ == "__main__":
    unittest.main()
