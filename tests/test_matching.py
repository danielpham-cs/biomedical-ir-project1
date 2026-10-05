"""End-to-end retrieval evidence: query rules, offsets, sentences and escaping."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import tokenize_word_spans, tokenize_words
from src.index import build_index
from src.matching import (document_matches, highlight_plan, highlight_spans,
                          make_match_plan, searchable_text)
from src.parse_pmc import Document
from src.search import highlight_text, search


def doc(identifier, abstract, body="", title="Title excluded"):
    return Document(identifier, identifier, "", title, abstract, body, "", "")


class MatchingTests(unittest.TestCase):
    def test_token_offsets_keep_original_unicode_and_whitespace(self):
        text = "  Straße\tHIV-1 12-15% 0.44–0.97 U.S.\nBrca1ΔC/ΔC"
        spans = tokenize_word_spans(text)
        self.assertEqual([word for word, _, _ in spans], tokenize_words(text))
        self.assertEqual([text[start:end].casefold() for _, start, end in spans], tokenize_words(text))
        index = build_index([], stem=False)
        self.assertEqual(highlight_text("12-15% U.S.", ["12", "u.s."], index),
                         "<mark>12</mark>-15% <mark>U.S.</mark>")

    def test_offsets_sentences_sections_and_normalized_markup(self):
        record = doc("1", "  Dr. Lee tested\tinsulin.\n\nKeywords: insulin", "H<sub>2</sub>O helps. Insulin worked.")
        text = searchable_text(record)
        matches = document_matches(record, make_match_plan("insulin"))
        self.assertEqual([(m.section, m.sentence) for m in matches], [("Abstract", 1), ("Abstract", 2), ("Body", 4)])
        for match in matches:
            self.assertEqual(text[match.start:match.end].casefold(), "insulin")
            row = match.row(text)
            self.assertEqual(row["Context"][row["Start in sentence"]:row["End in sentence"]], text[match.start:match.end])
        self.assertNotIn("<sub>", text)
        self.assertIn("H2O", text)

    def test_partial_is_literal_substring_and_highlights_every_occurrence(self):
        record = doc("1", "Banana diabetes diabetic.")
        index = build_index([record])
        hit = search("ana diabe", index, [record], match_mode="partial")[0]
        text = searchable_text(record)
        self.assertEqual([text[m.start:m.end].casefold() for m in hit.occurrences], ["ana", "ana", "diabe", "diabe"])
        self.assertEqual(hit.term_counts["ana"]["abstract"], 2)
        self.assertEqual(highlight_plan("Banana", make_match_plan("ana", match_mode="PARTIAL")), "B<mark>anana</mark>")
        # Surface matching must not expand to stem-equivalent 'study'.
        record = doc("2", "A study.")
        self.assertFalse(search("studi", build_index([record]), [record], match_mode="PARTIAL"))

    def test_partial_unicode_casefold_does_not_shift_offsets(self):
        record = doc("1", "Straße STRASSE.")
        hits = search("ss", build_index([record]), [record], match_mode="PARTIAL")
        text = searchable_text(record)
        self.assertEqual([text[m.start:m.end] for m in hits[0].occurrences], ["ß", "SS"])

    def test_and_or_require_query_groups_not_all_possible_variants(self):
        records = [doc("1", "Diabetes insulin."), doc("2", "Diabetic care."), doc("3", "Cancer insulin.")]
        index = build_index(records)
        self.assertEqual([h.doc_id for h in search("diab ins", index, records, match_mode="PARTIAL")], ["1"])
        self.assertEqual({h.doc_id for h in search("diab ins", index, records, method="OR", match_mode="PARTIAL")}, {"1", "2", "3"})
        self.assertEqual({h.doc_id for h in search("diab", index, records, match_mode="PARTIAL")}, {"1", "2"})

    def test_fuzzy_threshold_transposition_and_direct_query(self):
        record = doc("1", "Diabetes insulin. Diabetes care.")
        index = build_index([record])
        hit = search("diabets insluin", index, [record], match_mode="FUZZY", max_edits=1)[0]
        self.assertEqual({m.word.lower() for m in hit.occurrences}, {"diabetes", "insulin"})
        self.assertTrue(all(m.distance == 1 for m in hit.occurrences))
        self.assertFalse(search("diabts", index, [record], match_mode="FUZZY", max_edits=1))
        self.assertTrue(search("diabts", index, [record], match_mode="FUZZY", max_edits=2))
        self.assertIn("Diabetes", hit.snippet)

    def test_phrase_locations_retain_stopwords_and_find_all_phrases(self):
        record = doc("1", "Treatment of diabetes. Treatment of diabetes.", "Treatment\n\nof diabetes.")
        index = build_index([record])
        hit = search("treatment of diabetes", index, [record], method="PHRASE")[0]
        self.assertEqual(len(hit.occurrences), 2)
        text = searchable_text(record)
        self.assertTrue(all(text[m.start:m.end].casefold() == "treatment of diabetes" for m in hit.occurrences))

    def test_phrase_keeps_order_duplicates_and_supports_overlapping_matches(self):
        record = doc("1", "Insulin insulin insulin.")
        index = build_index([record])
        matches = search("insulin insulin", index, [record], method="PHRASE")[0].occurrences
        self.assertEqual(len(matches), 2)
        self.assertLess(matches[1].start, matches[0].end)
        self.assertFalse(search("insulin treatment", index, [record], method="PHRASE"))

    def test_phrase_handles_stemmed_stopwords_and_highlights_only_whole_phrases(self):
        record = doc("1", "Treatment alone. Treatment was effective.")
        hit = search("treatment was effective", build_index([record]), [record], method="PHRASE")[0]
        self.assertEqual(len(hit.occurrences), 1)
        self.assertEqual(hit.snippet_html, "Treatment alone. <mark>Treatment was effective</mark>.")
        plan = make_match_plan("treatment was effective", method="PHRASE")
        self.assertEqual(highlight_plan(record.abstract, plan), hit.snippet_html)

    def test_exact_index_counts_do_not_inadvertently_stem(self):
        record = doc("1", "Testing tested.", "Testing again.")
        hit = search("testing", build_index([record], stem=False), [record])[0]
        self.assertEqual(hit.term_counts["testing"], {"abstract": 1, "body": 1, "full": 2})
        self.assertEqual(len(hit.occurrences), 2)

    def test_title_excluded_and_invalid_modes_rejected(self):
        record = doc("1", "Unrelated evidence.", title="Diabetes")
        index = build_index([record])
        for mode in ("STEM", "PARTIAL", "FUZZY"):
            self.assertFalse(search("diabtes" if mode == "FUZZY" else "diabetes", index, [record], match_mode=mode))
        for kwargs in ({"match_mode": "unknown"}, {"match_mode": "PARTIAL", "method": "PHRASE"}, {"max_edits": 3}):
            with self.assertRaises(ValueError):
                search("test", index, [record], **kwargs)

    def test_highlights_escape_source_and_merge_overlapping_spans(self):
        self.assertEqual(highlight_spans("<script>aaa&", [(8, 10), (9, 11)]), "&lt;script&gt;<mark>aaa</mark>&amp;")
        with self.assertRaises(ValueError):
            highlight_spans("short", [(0, 6)])


if __name__ == "__main__":
    unittest.main()
