"""Tests for the local SQLite article catalog."""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.library import count_articles, list_articles, save_article_record
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


if __name__ == "__main__":
    unittest.main()
