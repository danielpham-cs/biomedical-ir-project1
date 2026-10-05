import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_store import fetch_corpus, list_corpora, set_archived, corpus_directory
from src.zipf_experiment import load_snapshot, load_corpus, analyze_corpus


def xml(pmid, language='eng'):
    return f'''<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
      <ArticleTitle>Trial {pmid}</ArticleTitle><Abstract><AbstractText>Patients receive GLP-1 treatments.</AbstractText></Abstract>
      <Language>{language}</Language></Article></MedlineCitation></PubmedArticle>'''.encode()


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fetch_snapshot_replay_idempotency_and_archive_restore(self):
        with patch('src.experiment_store.search_pmids', return_value=['1','1','2','3']) as search, patch('src.experiment_store.fetch_records', return_value=[xml('1'),xml('2'),xml('3','fre')]) as fetch:
            result = fetch_corpus(self.root, 'GLP-1', 3)
            self.assertEqual(result['document_count'], 2)
            self.assertEqual(result['requested'], 3)
            directory = corpus_directory(self.root, result['id'])
            frozen = load_snapshot(directory/'corpus_snapshot.json')
            raw = load_corpus(directory/'raw/pubmed_abstract_meta.json')
            self.assertEqual(analyze_corpus(frozen), analyze_corpus(raw))
            self.assertEqual(fetch_corpus(self.root, 'GLP-1', 3), result)
            self.assertEqual((search.call_count, fetch.call_count), (1,1))
            set_archived(self.root, result['id'])
            self.assertEqual(list_corpora(self.root), [])
            self.assertTrue((directory/'corpus_snapshot.json').exists())
            with self.assertRaisesRegex(ValueError, 'archived'):
                fetch_corpus(self.root, 'GLP-1', 3)
            set_archived(self.root, result['id'], False)
            self.assertEqual(len(list_corpora(self.root)), 1)

    def test_network_failure_not_registered_and_resume_skips_saved_batch(self):
        ids = [str(i) for i in range(1, 53)]
        with patch('src.experiment_store.search_pmids', return_value=ids), patch('src.experiment_store.fetch_records', side_effect=[[xml(i) for i in ids[:50]], RuntimeError('offline')]):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                fetch_corpus(self.root, 'GLP-1', 52)
        self.assertEqual(list_corpora(self.root), [])
        with patch('src.experiment_store.search_pmids') as search, patch('src.experiment_store.fetch_records', return_value=[xml('51'),xml('52')]) as fetch:
            result = fetch_corpus(self.root, 'GLP-1', 52)
            search.assert_not_called()
            fetch.assert_called_once_with(['51','52'])
            self.assertEqual(result['document_count'], 52)

    def test_tombstones_and_tampered_snapshot(self):
        with patch('src.experiment_store.search_pmids', return_value=['1','2']), patch('src.experiment_store.fetch_records', return_value=[xml('1'),xml('2')]):
            result = fetch_corpus(self.root, 'GLP-1', 2, excluded_pmids={'1'})
        path = corpus_directory(self.root, result['id'])/'corpus_snapshot.json'
        data = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual([d['pmid'] for d in data['documents']], ['2'])
        data['documents'][0]['abstract'] += ' tampered'
        path.write_text(json.dumps(data), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_snapshot(path)
        with self.assertRaises(ValueError):
            corpus_directory(self.root, '../outside')

    def test_invalid_input_does_not_call_network(self):
        with patch('src.experiment_store.search_pmids') as search:
            for query, count in [('',10), ('GLP-1',0), ('GLP-1',3001)]:
                with self.assertRaises(ValueError): fetch_corpus(self.root, query, count)
            search.assert_not_called()


if __name__ == '__main__': unittest.main()
