"""Offline tests for PMID lookup, exports, and collection integration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index import build_index, load_docs, load_index, rebuild_local_index, save_docs, save_index
from src.parse_pmc import Document, parse_directory
from src.pubmed import (PubMedError, export_json, fetch_pmc_fulltext,
                        fetch_pubmed_article, normalize_pmid, normalize_article_id,
                        pmcid_to_pmid, parse_pubmed_xml, save_article)
from src.search import search

PUBMED = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article>
<ArticleTitle>A <i>glucose</i> study</ArticleTitle>
<Journal><Title>Example Journal</Title><JournalIssue><PubDate><MedlineDate>2024 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
<Abstract><AbstractText Label="BACKGROUND">Glucose and <b>insulin</b>.</AbstractText><AbstractText Label="RESULTS">Improved control.</AbstractText></Abstract>
<AuthorList><Author><ForeName>Anna</ForeName><LastName>Lee</LastName></Author><Author><CollectiveName>Study Group</CollectiveName></Author></AuthorList>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC67890</ArticleId><ArticleId IdType="doi">10.1234/example</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>'''
PMC = b'''<pmc-articleset><article><front><journal-meta><journal-title>Example Journal</journal-title></journal-meta><article-meta>
<article-id pub-id-type="pmid">12345</article-id><article-id pub-id-type="pmc">67890</article-id><title-group><article-title>A glucose study</article-title></title-group>
<abstract><p>Glucose and insulin.</p></abstract></article-meta></front><body><p>Metformin treatment details in the full text.</p></body></article></pmc-articleset>'''


class PubMedTests(unittest.TestCase):
    def setUp(self):
        self.article = parse_pubmed_xml(PUBMED, "12345")

    def test_normalize_pmid(self):
        for text in ["12345", " PMID: 12345 ", "https://pubmed.ncbi.nlm.nih.gov/12345/?x=1#abstract"]:
            self.assertEqual(normalize_pmid(text), "12345")
        for text in ["", "0", "123,456", "../../12345", "PMC12345", "https://example.org/12345", "１２３"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_pmid(text)

    def test_structured_metadata_and_ir_json(self):
        self.assertEqual(self.article.title, "A glucose study")
        self.assertEqual(self.article.authors, ["Anna Lee", "Study Group"])
        self.assertEqual(self.article.year, "2024")
        self.assertEqual(self.article.abstract, "Glucose and insulin.\n\nImproved control.")
        data = json.loads(export_json(self.article))
        self.assertEqual(data["doc_id"], "12345")
        self.assertEqual(data["pmcid"], "67890")
        self.assertFalse(data["has_full_text"])
        self.assertEqual(data["body"], "")

    def test_preserves_superscript_for_abstract_display_not_statistics(self):
        xml = PUBMED.replace(
            b"Glucose and <b>insulin</b>.",
            b"Brca1<sup>\xce\x94C/\xce\x94C</sup> is active.",
        )
        article = parse_pubmed_xml(xml, "12345")
        self.assertEqual(article.abstract.split("\n\n")[0], "Brca1<sup>\u0394C/\u0394C</sup> is active.")
        from src.stats import compute_stats
        self.assertEqual(compute_stats(article.abstract).num_characters, len("Brca1\u0394C/\u0394C is active. Improved control."))

    def test_pmc_input_formats_and_numeric_ambiguity(self):
        for text in ["PMC13559554", " pmc13559554 ", "PMCID: PMC13559554",
                     "https://pmc.ncbi.nlm.nih.gov/articles/PMC13559554/",
                     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13559554/?x=1#abstract"]:
            self.assertEqual(normalize_article_id(text), "PMC13559554")
        self.assertEqual(normalize_article_id("13559554"), "13559554")
        for text in ["PMC0", "PMC12/../3", "PMC123,PMC456", "https://example.org/articles/PMC123/"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_article_id(text)

    @patch("src.pubmed._request_ncbi")
    def test_pmc_maps_to_correct_pubmed_id(self, request):
        request.side_effect = [
            b'<pmcids><record pmcid="PMC13559554" pmid="42724776"/></pmcids>',
            PUBMED.replace(b"12345", b"42724776").replace(b"PMC67890", b"PMC13559554"),
        ]
        article = fetch_pubmed_article("PMC13559554")
        self.assertEqual(article.pmid, "42724776")
        self.assertEqual(article.pmcid, "PMC13559554")
        self.assertEqual(request.call_args_list[1].args[1]["id"], "42724776")
        self.assertEqual(request.call_args_list[1].args[1]["db"], "pubmed")

    def test_pmc_mapping_missing_invalid_or_mismatched(self):
        for xml in [b"not XML", b"<pmcids/>",
                    b'<pmcids><record pmcid="PMC67890"/></pmcids>',
                    b'<pmcids><record pmcid="PMC99999" pmid="12345"/></pmcids>']:
            with self.subTest(xml=xml), patch("src.pubmed._request_ncbi", return_value=xml), self.assertRaises(PubMedError):
                pmcid_to_pmid("PMC67890")

    @patch("src.pubmed._request_ncbi")
    def test_pmc_mapping_cannot_return_a_different_paper(self, request):
        request.side_effect = [b'<pmcids><record pmcid="PMC99999" pmid="12345"/></pmcids>', PUBMED]
        with self.assertRaisesRegex(PubMedError, "inconsistent"):
            fetch_pubmed_article("PMC99999")

    def test_missing_abstract_and_pmc(self):
        article = parse_pubmed_xml(b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article><ArticleTitle>Brief report</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>')
        self.assertEqual(article.abstract, "")
        self.assertEqual(article.pmcid, "")
        self.assertEqual(article.authors, [])

    def test_missing_wrong_and_invalid_records(self):
        for xml in [b"<PubmedArticleSet/>", b"<ERROR>bad ID</ERROR>", b"<html>", PUBMED.replace(b"12345", b"99999")]:
            with self.subTest(xml=xml), self.assertRaises(PubMedError):
                parse_pubmed_xml(xml, "12345")

    @patch("src.pubmed.requests.Session.get")
    def test_fetch_uses_pubmed_and_checks_id(self, get):
        get.return_value = Mock(status_code=200, content=PUBMED)
        result = fetch_pubmed_article("PMID: 12345")
        self.assertEqual(result.pmid, "12345")
        self.assertEqual(get.call_args.kwargs["params"]["db"], "pubmed")
        self.assertEqual(get.call_args.kwargs["params"]["id"], "12345")

    @patch("src.pubmed.requests.Session.get")
    def test_invalid_id_never_requests_network(self, get):
        with self.assertRaises(ValueError):
            fetch_pubmed_article("../../bad")
        get.assert_not_called()

    @patch("src.pubmed.requests.Session.get")
    def test_network_errors(self, get):
        for error in [requests.Timeout(), requests.ConnectionError(), requests.HTTPError()]:
            get.side_effect = error
            with self.subTest(error=type(error)), self.assertRaises(PubMedError):
                fetch_pubmed_article("12345")
        get.side_effect = None
        get.return_value = Mock(status_code=429)
        with self.assertRaisesRegex(PubMedError, "too many requests"):
            fetch_pubmed_article("12345")

    @patch("src.pubmed._fetch_xml", return_value=PMC)
    def test_full_text_export(self, fetch):
        xml = fetch_pmc_fulltext(self.article)
        data = json.loads(export_json(self.article, xml))
        self.assertEqual(data["doc_id"], "12345")
        self.assertTrue(data["has_full_text"])
        self.assertIn("Metformin", data["body"])
        fetch.assert_called_once_with("pmc", "67890")

    def test_unavailable_or_wrong_full_text(self):
        for xml in [b"<error>Not available</error>", PMC.replace(b"67890", b"77777"), PMC.replace(b"12345", b"99999"), PMC.replace(b"<body><p>Metformin treatment details in the full text.</p></body>", b"")]:
            with self.subTest(xml=xml), patch("src.pubmed._fetch_xml", return_value=xml), self.assertRaises(PubMedError):
                fetch_pmc_fulltext(self.article)
        with self.assertRaises(PubMedError):
            export_json(self.article, PMC.replace(b"67890", b"77777"))

    def test_save_and_parse_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            save_article(self.article, raw)
            self.assertEqual(len(parse_directory(raw)), 1)
            for xml in [PMC, PMC.replace(b'<article-id pub-id-type="pmid">12345</article-id>', b'')]:
                paths = save_article(self.article, raw, xml)
                self.assertEqual(len(paths), 3)
                self.assertTrue(all(p.exists() for p in paths))
                docs = parse_directory(raw)
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0].doc_id, "12345")
                self.assertIn("Metformin", docs[0].body)
            self.assertFalse(list(raw.glob("*.tmp")))

    def test_merge_preserves_existing_docs_settings_and_fulltext(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, processed = root / "raw", root / "processed"
            existing = Document("old", "old", "", "Old paper", "Existing evidence", "", "", "")
            save_docs([existing], processed / "docs.jsonl")
            save_index(build_index([existing], stem=False, remove_stopwords=False), processed / "index.pkl")
            save_article(self.article, raw, PMC)
            self.assertEqual(rebuild_local_index(raw, processed), 2)
            index = load_index(processed / "index.pkl")
            docs = load_docs(processed / "docs.jsonl")
            self.assertFalse(index.stem)
            self.assertFalse(index.remove_stopwords)
            self.assertEqual(search("metformin", index, docs)[0].doc_id, "12345")
            save_article(self.article, raw)
            self.assertEqual(rebuild_local_index(raw, processed), 2)
            self.assertTrue(next(d for d in load_docs(processed / "docs.jsonl") if d.doc_id == "12345").body)

    def test_pmc_without_pmid_reuses_existing_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, processed = root / "raw", root / "processed"
            raw.mkdir()
            existing = self.article.to_document()
            save_docs([existing], processed / "docs.jsonl")
            save_index(build_index([existing], stem=False, remove_stopwords=False), processed / "index.pkl")
            (raw / "PMC67890.xml").write_bytes(PMC.replace(b'<article-id pub-id-type="pmid">12345</article-id>', b''))
            self.assertEqual(rebuild_local_index(raw, processed), 1)
            doc = load_docs(processed / "docs.jsonl")[0]
            self.assertEqual(doc.pmid, "12345")
            self.assertIn("Metformin", doc.body)


if __name__ == "__main__":
    unittest.main()
