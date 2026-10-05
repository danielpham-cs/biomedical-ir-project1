"""Offline checks for corpus activation, repeat runs and restore."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_glp1_corpus as migration
from src.index import load_docs
from src.library import count_articles


def record(pmid):
    return (f'<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>'
            f'<ArticleTitle>GLP-1 study {pmid}</ArticleTitle><Language>eng</Language>'
            '<Abstract><AbstractText>GLP-1 receptor treatment improves glucose control.</AbstractText></Abstract>'
            '</Article></MedlineCitation></PubmedArticle>').encode()


class GLP1MigrationTests(unittest.TestCase):
    def setup_corpus(self, root):
        raw = root / 'data/raw'
        raw.mkdir(parents=True)
        (raw / 'PMID1.xml').write_bytes(record('1'))
        manifest = {'query': 'old topic', 'saved': [{'pmid': '1'}]}
        (raw / 'pubmed_abstract_meta.json').write_text(json.dumps(manifest))
        return manifest

    def test_activation_is_repeatable_and_restore_keeps_all_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = self.setup_corpus(root)
            with patch.object(migration, 'ROOT', root), \
                 patch.object(migration, 'search_pmids', return_value=[str(n) for n in range(1, 13)]) as search, \
                 patch.object(migration, 'fetch_records', side_effect=lambda ids: [record(pmid) for pmid in ids]) as fetch:
                migration.run(10)
                manifest = json.loads((root / 'data/raw/pubmed_abstract_meta.json').read_text())
                self.assertEqual(count_articles(root / 'data/library.sqlite3'), 10)
                self.assertEqual(count_articles(root / 'data/library.sqlite3', include_deleted=True), 11)
                self.assertNotIn('1', {d.pmid for d in load_docs(root / 'data/processed/docs.jsonl')})
                calls = (search.call_count, fetch.call_count)
                migration.run(10)
                self.assertEqual((search.call_count, fetch.call_count), calls)
                migration.restore(root / manifest['archive'])
                self.assertEqual(count_articles(root / 'data/library.sqlite3'), 1)
                self.assertEqual([d.pmid for d in load_docs(root / 'data/processed/docs.jsonl')], ['1'])
                self.assertEqual(json.loads((root / 'data/raw/pubmed_abstract_meta.json').read_text()), previous)
                self.assertTrue(all((root / 'data/raw' / f'PMID{n}.xml').exists() for n in range(1, 12)))

    def test_download_failure_does_not_replace_old_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = self.setup_corpus(root)
            with patch.object(migration, 'ROOT', root), \
                 patch.object(migration, 'search_pmids', return_value=[str(n) for n in range(2, 12)]), \
                 patch.object(migration, 'fetch_records', side_effect=RuntimeError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'offline'):
                    migration.run(10)
                self.assertEqual(json.loads((root / 'data/raw/pubmed_abstract_meta.json').read_text()), previous)
                self.assertEqual((root / 'data/raw/PMID1.xml').read_bytes(), record('1'))
                self.assertFalse((root / 'data/archives').exists())


if __name__ == '__main__':
    unittest.main()
