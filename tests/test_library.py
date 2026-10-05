"""Tests for the local SQLite article catalog."""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.library import active_documents, count_articles, list_articles, save_article_record, set_articles_deleted
from src.pubmed import PubMedArticle


class LocalLibraryTests(unittest.TestCase):
    def setUp(self):
        self.article = PubMedArticle(
            pmid="12345", pmcid="PMC67890", title="Local evidence", abstract="An abstract.",
            journal="Test Journal", year="2026", authors=["A. Lee", "B. Chen"], doi="10.1/test",
            xml=b"<PubmedArticleSet />",
        )

    def make_files(self, root: Path, *, fulltext: bool) -> list[Path]:
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        files = [raw / "PMID12345.xml", raw / "PMID12345.json"]
        if fulltext:
            files.append(raw / "PMC67890.xml")
        for file in files:
            file.write_bytes(b"content")
        return files

    def test_catalog_is_created_and_lists_saved_article(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data" / "library.sqlite3"
            save_article_record(database, self.article, paths=self.make_files(root, fulltext=True),
                                project_root=root, has_full_text=True)
            self.assertTrue(database.exists())
            self.assertEqual(count_articles(database), 1)
            row = list_articles(database)[0]
            self.assertEqual(row["PMID"], "12345")
            self.assertEqual(row["PMCID"], "PMC67890")
            self.assertEqual(row["Text"], "Full text")

    def test_repeated_download_updates_one_catalog_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data" / "library.sqlite3"
            files = self.make_files(root, fulltext=False)
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            self.article.title = "Updated evidence"
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            self.assertEqual(count_articles(database), 1)
            row = list_articles(database)[0]
            self.assertEqual(row["Title"], "Updated evidence")
            self.assertEqual(row["Text"], "Title / abstract")

    def test_catalog_stores_relative_file_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data" / "library.sqlite3"
            save_article_record(database, self.article, paths=self.make_files(root, fulltext=True),
                                project_root=root, has_full_text=True)
            connection = sqlite3.connect(database)
            try:
                row = connection.execute("SELECT pubmed_xml_path, pmc_xml_path, json_path FROM articles").fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ("data\\raw\\PMID12345.xml", "data\\raw\\PMC67890.xml", "data\\raw\\PMID12345.json"))

    def test_multiple_articles_without_pmcid_can_be_imported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data/library.sqlite3"
            self.article.pmcid = ""
            files = self.make_files(root, fulltext=False)
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            self.article.pmid = "54321"
            other_files = [root / "data/raw/PMID54321.xml", root / "data/raw/PMID54321.json"]
            for path in other_files:
                path.write_bytes(b"content")
            save_article_record(database, self.article, paths=other_files, project_root=root, has_full_text=False)
            self.assertEqual(count_articles(database), 2)
            self.assertEqual({row["PMCID"] for row in list_articles(database)}, {"-"})

    def test_soft_delete_survives_upsert_and_restore_preserves_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data/library.sqlite3"
            files = self.make_files(root, fulltext=False)
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            set_articles_deleted(database, [self.article.pmid])
            self.assertEqual(count_articles(database), 0)
            self.assertEqual(count_articles(database, include_deleted=True), 1)
            self.assertEqual(list_articles(database), [])
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            self.assertEqual(count_articles(database), 0)
            self.assertTrue(all(path.read_bytes() == b"content" for path in files))
            # PMID and PMCID numbers are separate namespaces.
            other = PubMedArticle(pmid="67890", title="Different paper", abstract="Text", journal="", year="")
            self.assertEqual(active_documents([other.to_document()], database), [other.to_document()])
            set_articles_deleted(database, [self.article.pmid], deleted=False)
            self.assertEqual(count_articles(database), 1)

    def test_rebuild_does_not_resurrect_hidden_xml_or_old_index_rows(self):
        from src.index import build_index, save_docs, save_index, rebuild_local_index, load_docs, load_index
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "data/library.sqlite3"
            raw, processed = root / "data/raw", root / "data/processed"
            files = self.make_files(root, fulltext=False)
            files[0].write_text('<PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>Local evidence</ArticleTitle><Abstract><AbstractText>An abstract.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>')
            save_article_record(database, self.article, paths=files, project_root=root, has_full_text=False)
            docs = [self.article.to_document()]
            save_docs(docs, processed / "docs.jsonl")
            save_index(build_index(docs, remove_stopwords=False), processed / "index.pkl")
            set_articles_deleted(database, ["12345"])
            self.assertEqual(load_docs(processed / "docs.jsonl"), [])
            for _ in range(2):
                self.assertEqual(rebuild_local_index(raw, processed), 0)
                self.assertEqual(load_index(processed / "index.pkl").N, 0)
            self.assertTrue(files[0].exists())
            set_articles_deleted(database, ["12345"], deleted=False)
            self.assertEqual(rebuild_local_index(raw, processed), 1)


if __name__ == "__main__":
    unittest.main()
