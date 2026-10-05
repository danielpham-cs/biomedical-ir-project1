import json
import math
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.parse_pmc import Document
from src.zipf_experiment import (AbstractCorpus, analyze_corpus, fit_zipf, load_corpus,
                                 region_fits, term_documents, term_spans, tokens, token_trace)


def corpus(texts):
    docs = [Document(str(i), str(i), '', 'Excluded title', text, '', '', '') for i, text in enumerate(texts, 1)]
    return AbstractCorpus(docs, 'test', 'test-fingerprint', {})


class DetailedZipfTests(unittest.TestCase):
    def test_conditions_differ_and_preserve_biomedical_tokens(self):
        text = 'The data, data. data; GLP‑1 24.9% (0.65) patients treatments !!!'
        stops = {'the'}
        self.assertIn('data,', tokens(text, 'A', stops))
        self.assertEqual(tokens(text, 'B', stops).count('data'), 3)
        self.assertNotIn('the', tokens(text, 'C', stops))
        for term in ('glp-1', '24.9%', '0.65'):
            self.assertIn(term, tokens(text, 'D', stops))
        self.assertIn('patient', tokens(text, 'D', stops))
        trace = token_trace(text, stops)
        for condition in 'ABCD':
            self.assertEqual([r[condition] for r in trace if r[condition]], tokens(text, condition, stops))

    def test_cf_df_idf_empty_document_denominator_and_term_locations(self):
        c = corpus(['alpha alpha beta', 'alpha gamma', 'the'])
        result = analyze_corpus(c, {'the'})
        rows = {r['term']: r for r in result['conditions']['C']['rows']}
        self.assertEqual((rows['alpha']['CF'], rows['alpha']['DF']), (3, 2))
        self.assertAlmostEqual(rows['alpha']['IDF'], math.log(3/2))
        self.assertEqual(result['conditions']['C']['stats']['documents'], 3)
        self.assertAlmostEqual(result['conditions']['C']['stats']['average_tokens_per_document'], 5/3)
        documents = term_documents(c, 'alpha', 'C', {'the'})
        self.assertEqual([r['TF'] for r in documents], [2, 1])
        self.assertAlmostEqual(documents[0]['TF_IDF'], 2*math.log(3/2))
        text = 'Patients, and patient.'
        spans = term_spans(text, 'patient', 'D', {'and'})
        self.assertEqual([text[a:b] for a,b in spans], ['Patients,', 'patient.'])
        common = analyze_corpus(corpus(['word', 'word word']), set())['conditions']['B']['rows'][0]
        self.assertEqual(common['IDF'], 0)

    def test_known_power_law_and_constant_tail(self):
        ranks = np.arange(1, 101)
        result = fit_zipf(ranks, 10000/ranks**1.3)
        self.assertAlmostEqual(result['slope'], -1.3)
        self.assertAlmostEqual(result['zipf_exponent'], 1.3)
        self.assertAlmostEqual(result['intercept'], 4)
        self.assertAlmostEqual(result['r_squared'], 1)
        self.assertLess(result['rmse_log10'], 1e-12)
        rows = [{'rank': int(r), 'CF': 1} for r in ranks]
        regions = region_fits(rows)
        self.assertEqual([regions[k]['points'] for k in list(regions)[1:]], [10,70,20])
        self.assertIsNone(regions['Tail (80–100%)']['r_squared'])
        self.assertIsNone(fit_zipf([], [])['slope'])
        with self.assertRaises(ValueError): fit_zipf([1,2],[1,0])

    def test_abstract_only_scope_and_source_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root/'pubmed_abstract_meta.json'
            manifest.write_text(json.dumps({'query':'GLP-1','saved':[{'pmid':'1'}]}))
            xml = '<PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle>Excluded</ArticleTitle><Language>eng</Language><Abstract><AbstractText Label="METHODS">Plain <i>abstract</i> text.</AbstractText></Abstract></Article><KeywordList><Keyword>Excludedkeyword</Keyword></KeywordList></MedlineCitation></PubmedArticle>'
            (root/'PMID1.xml').write_text(xml)
            first = load_corpus(manifest)
            self.assertEqual(first.documents[0].abstract,'Plain abstract text.')
            (root/'PMID1.xml').write_text(xml.replace('Excludedkeyword','Differentkeyword'))
            second = load_corpus(manifest)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertNotEqual(first.source_hashes,second.source_hashes)
            (root/'PMID1.xml').write_text(xml.replace('eng','fra'))
            with self.assertRaises(ValueError): load_corpus(manifest)


if __name__ == '__main__': unittest.main()
