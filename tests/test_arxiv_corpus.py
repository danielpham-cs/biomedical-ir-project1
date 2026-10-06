"""arXiv computer-science corpus: feed parsing, balanced categories, snapshot round-trip, deletion (offline)."""
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_store import delete_corpus, fetch_arxiv_corpus, list_corpora, parse_arxiv_feed
from src.zipf_experiment import load_snapshot


def feed(category, count, start=0, extra=''):
    entries = ''.join(
        f'<entry><id>http://arxiv.org/abs/2510.{start + i:05d}v2</id><title>Paper {i}\n  in {category}</title>'
        f'<summary>We study\n  {category} problem number {i} with   new methods.</summary>'
        f'<published>2025-10-0{1 + i % 9}T00:00:00Z</published>'
        f'<arxiv:primary_category term="{category}"/></entry>' for i in range(count))
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">{entries}{extra}</feed>').encode()


class ArxivCorpusTests(unittest.TestCase):
    def test_parse_normalizes_whitespace_and_skips_old_ids(self):
        old = ('<entry><id>http://arxiv.org/abs/cs/0101001v1</id><title>Old</title><summary>Old.</summary>'
               '<published>2001-01-01T00:00:00Z</published><arxiv:primary_category term="cs.AI"/></entry>')
        entries = parse_arxiv_feed(feed('cs.AI', 2, extra=old))
        self.assertEqual([e['id'] for e in entries], ['2510.00000', '2510.00001'])  # version stripped, old id skipped
        self.assertEqual(entries[0]['abstract'], 'We study cs.AI problem number 0 with new methods.')
        self.assertEqual((entries[0]['title'], entries[0]['year'], entries[0]['primary']), ('Paper 0 in cs.AI', '2025', 'cs.AI'))

    def test_balanced_fetch_snapshot_and_delete(self):
        categories = ('cs.AI', 'cs.DB')

        def download(url):
            category = parse_qs(urlparse(url).query)['search_query'][0].split(':')[1]
            return feed(category, 8, start=100 * categories.index(category))

        with tempfile.TemporaryDirectory() as root:
            row = fetch_arxiv_corpus(root, categories=categories, per_category=3, download=download, pause=0)
            self.assertEqual(row['document_count'], 6)
            corpus = load_snapshot(Path(root) / row['id'] / 'corpus_snapshot.json')
            self.assertEqual(corpus.source, 'arxiv')
            self.assertEqual([d.journal for d in corpus.documents].count('arXiv cs.DB'), 3)  # equal share per subfield
            self.assertEqual(corpus.documents[0].pmid, '2510.00000')
            self.assertEqual(fetch_arxiv_corpus(root, categories=categories, per_category=3, download=None)['id'], row['id'])  # reused
            delete_corpus(root, row['id'])
            self.assertEqual(list_corpora(root, include_deleted=True), [])
            self.assertFalse((Path(root) / row['id']).exists())

    def test_too_few_abstracts_registers_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                fetch_arxiv_corpus(root, categories=('cs.AI',), per_category=5, download=lambda url: feed('cs.AI', 2), pause=0)
            self.assertEqual(list_corpora(root, include_deleted=True), [])


if __name__ == '__main__':
    unittest.main()
