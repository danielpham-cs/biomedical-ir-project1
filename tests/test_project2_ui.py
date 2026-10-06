"""Regression tests for the Homework 2 demo pages: rendering, state, correction and reader."""
import json
import re
import sys
import tempfile
import unittest
from html import unescape
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src.parse_pmc import Document
from src.word2vec import Word2VecConfig, train_skipgram
from src.zipf_experiment import AbstractCorpus, analyze_corpus, embedding_preprocessing, embedding_sentences
from ui.hw2 import common
from ui.hw2.registry import BY_KEY, PAGES


def fixture_corpus(prefix, texts, fingerprint):
    documents = [Document(f'PMID{prefix}{i}', f'{prefix}{i}', '', title, text, '', '', '2026') for i, (title, text) in enumerate(texts, 1)]
    return AbstractCorpus(documents, f'{prefix} test', fingerprint, {})


class Project2UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = AbstractCorpus([
            Document('PMID101', '101', '', 'GLP-1 semaglutide trial', 'The patients received semaglutide. GLP-1 receptor treatments helped patients.', '', '', '2026'),
            Document('PMID102', '102', '', 'Receptor study', 'The receptor and insulin treatments helped. Patients received semaglutide and insulin.', '', '', '2026'),
            Document('PMID103', '103', '', 'Other paper', 'Insulin helped patients. Semaglutide helped patients and receptor studies.', '', '', '2025'),
        ], 'GLP-1 test', 'ui-fixture-fingerprint', {})
        cls.other = fixture_corpus('9', [('Net', 'The neural network model learned features. The model used deep learning layers.'),
                                          ('Algo', 'An algorithm trained the network. Learning rates changed the model.'),
                                          ('Data', 'Deep models need data. The network architecture improved learning.')], 'other-fixture')
        cls.result = analyze_corpus(cls.corpus)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        config = Word2VecConfig(epochs=1, dimensions=4, min_count=1, sample=0)
        model = train_skipgram(embedding_sentences(cls.corpus), config, corpus_metadata=cls.corpus.metadata(), preprocessing=embedding_preprocessing())
        model.save(cls.output / 'skipgram.npz')
        (cls.output / 'run_metadata.json').write_text(json.dumps({'run_key': cls.result['run_key']}), encoding='utf8')
        (cls.output / 'research_answers.json').write_text(json.dumps({'RQ1': 'Fixture answer'}), encoding='utf8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # AppTest's standalone page has no multipage registry. Only replace global
        # page links; all experiment widgets, callbacks, rendering and reader run.
        for p in [patch('streamlit.page_link'), patch.object(common, 'experiment', return_value=(self.corpus, self.result)),
                  patch.object(common, 'OUTPUT', self.output), patch.object(common, 'EXPERIMENTS', self.output / 'extra_corpora')]:
            p.start()
            self.addCleanup(p.stop)

    def app(self, screen='data', /, **params):
        at = AppTest.from_file(str(common.ROOT / BY_KEY[screen].file), default_timeout=60)
        at.query_params.update(params)
        at.run()
        self.assert_clean(at)
        return at

    def assert_clean(self, at):
        self.assertEqual([e.message for e in at.exception], [])

    def captions(self, at):
        return ' '.join(c.value for c in at.caption)

    def test_every_page_renders_its_requirements(self):
        for page in PAGES:
            with self.subTest(page=page.key):
                at = self.app(page.key)
                header = next(m.value for m in at.markdown if 'class="p2-header"' in m.value)
                self.assertIn(page.title, unescape(header))
                for _, ref, _ in page.requirements:
                    self.assertIn(ref, unescape(header))

    def test_thin_ranks_keeps_shape_with_few_points(self):
        # 20,000 ranks: a falling head, then long plateaus of tied CF in the tail.
        rows = [{'rank': r, 'CF': max(1, 50_000 // r)} for r in range(1, 20_001)]
        thin = common.thin_ranks(rows)
        self.assertLess(len(thin), 1_500)
        self.assertEqual({r['CF'] for r in thin}, {r['CF'] for r in rows})
        self.assertEqual((thin[0], thin[-1]), (rows[0], rows[-1]))
        for cf in (2, 3, 4):  # both ends of each tail plateau survive (minimum CF here is 2)
            plateau = [r['rank'] for r in rows if r['CF'] == cf]
            self.assertTrue({plateau[0], plateau[-1]} <= {r['rank'] for r in thin})
        small = rows[:100]
        self.assertEqual(common.thin_ranks(small), small)

    def test_overview_is_minimal(self):
        at = self.app('overview')
        self.assertEqual([e.label for e in at.expander][0], 'System pipeline and modules')
        self.assertNotIn('Run provenance', [e.label for e in at.expander])
        self.assertFalse(any('Demonstrated on' in m.value for m in at.markdown))
        self.assertEqual(len(at.get('download_button')), 0)

    def test_corpus_filter_and_empty_state(self):
        at = self.app('data')
        at.text_input(key='_p2_filter').set_value('101').run()
        self.assertIn('1 matches of 3 papers', self.captions(at))
        at.text_input(key='_p2_filter').set_value('missing title').run()
        self.assertIn('No papers match', at.info[0].value)

    def test_porter_trace_and_reference_agreement(self):
        at = self.app('data')
        chain = next(m.value for m in at.markdown if 'class="p2-steps"' in m.value)
        self.assertIn('gener', chain)
        self.assertIn('Result: generalizations → gener', self.captions(at))
        agreement = next(m for m in at.metric if m.label.startswith('Agreement'))
        self.assertIn('0 differences', agreement.delta)

    def test_edit_distance_table_and_candidates(self):
        at = self.app('search', query='semaglutde')
        # Short default example, one per edit operation.
        expected = {'insert': 'insert ε → e', 'substitute': 'substitute z → s', 'transpose': 'transpose th → ht'}
        for name, text in expected.items():
            at.radio(key='p2_ed_example').set_value(name).run()
            self.assert_clean(at)
            self.assertEqual(next(m.value for m in at.metric if m.label == 'Edit distance'), '1')
            self.assertIn(text, self.captions(at))
        at.text_input(key='p2_spell_word').set_value('semaglutde').run()
        self.assertTrue(any('semaglutide' in m.value for m in at.markdown))

    def test_keyword_interpretation_and_location_chips(self):
        at = self.app('search', query='semaglutde receptor', mode='FUZZY', method='AND')
        at.toggle(key='p2_autocorrect').set_value(False).run()  # fuzzy matching of the typed word itself
        keywords = unescape(next(m.value for m in at.markdown if 'class="p2-keywords"' in m.value))
        self.assertIn('<s>semaglutde</s> → <b class="p2-fix">semaglutide</b> <small>(1 edit)</small>', keywords)
        self.assertIn('<b>receptor</b> <span class="p2-ok">✓</span>', keywords)
        card = unescape(next(m.value for m in at.markdown if 'class="p2-hit"' in m.value))
        # One muted meta line, a highlighted snippet and positions grouped by matched word.
        self.assertRegex(card, r'PMID \d+ · BM25 [\d.]+ · <span class="p2-hit-count">\d+ match')
        self.assertIn('class="p2-hit-snippet"', card)
        self.assertRegex(card, r'<b>semaglutide \(d=1\)</b> \d+')
        self.assertIn('p2_ed_left_livr', [t.key for t in at.text_input])  # DP table uses a short example, not the query

    def test_zipf_top50_switches_condition(self):
        at = self.app('zipf')
        choice = at.radio(key='p2_top50_condition')
        self.assertEqual(choice.value, 'B')  # starts at the page's condition
        for c in 'ABCD':
            choice.set_value(c).run()
            self.assert_clean(at)
            table = next(d.value for d in at.dataframe if list(d.value.columns) == ['rank', 'term', 'CF', 'DF', 'relative_frequency'])
            expected = [r['term'] for r in self.result['conditions'][c]['rows'][:len(table)]]
            self.assertEqual(list(table['term']), expected)
            self.assertIn(f'Condition {c}. CF(t)', self.captions(at))
            choice = at.radio(key='p2_top50_condition')

    def test_zipf_effect_of_preprocessing_table(self):
        at = self.app('zipf')
        self.assertIn('Zipf exponent k', [m.label for m in at.metric])
        table = next(m.value for m in at.markdown if 'Step added' in m.value)
        for header in ('Vocabulary', 'Tokens', 'Top 3 terms', 'Zipf k', 'Slope', 'Intercept', 'R²', 'RMSE'):
            self.assertIn(header, table)
        conditions = self.result['conditions']
        for c in 'ABCD':  # one row per condition, with its own exponent
            self.assertIn(f'<b>{c}</b>', table)
            self.assertIn(f"<b>{conditions[c]['fits']['All']['zipf_exponent']:.3f}</b>", table)
        c, d = (conditions[x]['stats']['vocabulary'] for x in 'CD')
        self.assertIn(f'{d:,}<br><small class="p2-change p2-nowrap">{d - c:+,}', table)  # change against the row above
        summary = next(m.value for m in at.markdown if '**A → B**' in m.value)
        self.assertIn('**B → C**', summary)
        self.assertIn('**C → D**', summary)
        self.assertFalse(any('p2-chip' in m.value for m in at.markdown))

    def test_autocorrect_searches_corrected_query_and_can_be_turned_off(self):
        at = self.app('search', query='semaglutde')
        self.assertIn('3 matching documents', self.captions(at))
        next(b for b in at.button if b.key and b.key.startswith('p2_open_')).click().run()
        self.assertIn('Query: semaglutide · STEM', self.captions(at))  # the popup highlights the corrected query
        at.session_state['p2_popup'] = None
        at.toggle(key='p2_autocorrect').set_value(False).run()
        self.assertIn('0 matching documents', self.captions(at))

    def test_correction_apply_keeps_query(self):
        at = self.app('search', query='semaglutde')
        self.assertEqual(at.text_input(key='_p2_query').value, 'semaglutde')
        next(b for b in at.button if b.label == 'Apply spelling suggestion').click().run()
        self.assert_clean(at)
        self.assertEqual(at.text_input(key='_p2_query').value, 'semaglutide')
        self.assertIn('3 matching documents', self.captions(at))
        for mode, query in [('PARTIAL', 'semaglut'), ('FUZZY', 'semaglutde')]:
            at.selectbox(key='_p2_mode').set_value(mode).run()
            at.text_input(key='_p2_query').set_value(query).run()
            self.assert_clean(at)
            self.assertIn('3 matching documents', self.captions(at))

    def test_internal_link_roundtrip_preserves_context_and_theme(self):
        at = self.app('data', filter='semaglutide', theme='dark')
        html = next(m.value for m in at.markdown if 'class="p2-table"' in m.value and 'PMID' in m.value)
        href = unescape(re.search(r'href="([^"]+)"', html).group(1))
        self.assertTrue(href.startswith('/HW2_Data?'))
        params = {k: v[0] for k, v in parse_qs(urlparse(href).query).items()}
        reader = self.app('data', **params)
        self.assertTrue(reader.session_state.ui_dark_mode)
        self.assertEqual(reader.subheader[0].value, self.corpus.documents[0].title)
        back = next(m.value for m in reader.markdown if 'Back to Data' in m.value)
        back_url = unescape(re.search(r'href="([^"]+)"', back).group(1))
        restored = self.app('data', **{k: v[0] for k, v in parse_qs(urlparse(back_url).query).items()})
        self.assertEqual(restored.text_input(key='_p2_filter').value, 'semaglutide')
        self.assertTrue(restored.session_state.ui_dark_mode)

    def test_term_reader_uses_condition_not_old_search(self):
        at = self.app('terms', paper='101', source='term', condition='D', term='patient', query='insulin')
        self.assertIn('TF = 2', self.captions(at))
        abstract = next(m.value for m in at.markdown if 'class="p2-reading p2-abstract"' in m.value)
        self.assertEqual(len(re.findall(r'<mark>patients\b[^<]*</mark>', abstract)), 2)  # every match highlighted
        self.assertIn('2 matches highlighted.', self.captions(at))
        self.assertFalse(any(b.label in {'Next match', 'Previous match'} for b in at.button))
        missing = self.app('data', paper='999')
        self.assertIn('archived or outside', missing.warning[0].value)

    def test_constant_tail_and_missing_report(self):
        at = self.app('zipf')
        region = next(s for s in at.selectbox if s.label == 'Rank region')
        region.set_value(region.options[-1]).run()
        self.assert_clean(at)
        self.assertEqual([m.value for m in at.metric if m.label == 'R²'][-1], 'N/A')  # Details tab: selected region
        with patch.object(common, 'OUTPUT', self.output / 'missing'):
            missing = self.app('overview')  # downloads are no longer on the page; answers still render
            self.assertEqual(missing.warning, [])
            self.assertTrue(any(e.label.startswith('RQ1') for e in missing.expander))

    def test_word2vec_shows_training_pairs(self):
        at = self.app('word2vec')
        self.assertIn('Skip-gram: the green word predicts each orange word', self.captions(at))
        self.assertEqual(next(m.value for m in at.metric if m.label == 'Dimensions'), '4')
        at.radio(key='p2_w2v_view').set_value('cbow').run()
        self.assert_clean(at)
        self.assertIn('CBOW: the', self.captions(at))
        self.assertIn('are averaged into one vector h that predicts the green word', self.captions(at))
        self.assertNotIn('v_center', self.captions(at))  # underscores in captions render as italics
        self.assertEqual(at.radio(key='p2_w2v_view').options, ['Skip-gram', 'CBOW'])

    def test_word2vec_map_shows_only_the_query_and_its_neighbours(self):
        at = self.app('word2vec')
        at.text_input(key='p2_embedding_query').set_value('semaglutide').run()
        self.assert_clean(at)
        self.assertFalse(any(s.label == 'Words on the map' for s in at.selectbox))  # no 'Frequent words' mode
        self.assertTrue(any('nearest words by cosine' in c.value for c in at.caption))
        at.text_input(key='p2_embedding_query').set_value('unknownword').run()
        self.assert_clean(at)
        self.assertTrue(any('to see it and its nearest words on the map' in i.value for i in at.info))

    def test_word2vec_sentence_dropdown_shows_the_source_sentence(self):
        at = self.app('word2vec')
        sentence = next(s for s in at.selectbox if s.label.startswith('2 · Sentence'))
        self.assertEqual(sentence.options, ['1/2 · The patients received semaglutide.', '2/2 · GLP-1 receptor treatments helped patients.'])
        original = next(m.value for m in at.markdown if 'Original sentence' in m.value)
        self.assertIn('<s>The</s> patients received semaglutide.', original)  # the stopword removed by condition C
        sentence.set_value(1).run()
        self.assert_clean(at)
        original = next(m.value for m in at.markdown if 'Original sentence' in m.value)
        self.assertIn('GLP-1 receptor treatments helped patients.', original)
        tokens_row = next(m.value for m in at.markdown if 'class="p2-steps"' in m.value)
        self.assertIn('treatments', tokens_row)

    def test_search_result_opens_in_popup(self):
        at = self.app('search', query='patients')
        title = next(b for b in at.button if b.key and b.key.startswith('p2_open_'))
        title.click().run()
        self.assert_clean(at)
        self.assertTrue(any(s.value in {d.title for d in self.corpus.documents} for s in at.subheader))  # paper shown
        self.assertIn('2 matches highlighted.', self.captions(at))
        abstract = next(m.value for m in at.markdown if 'class="p2-reading p2-abstract"' in m.value)
        self.assertEqual(abstract.count('<mark>'), 2)
        # Straight to the full abstract: no excerpt and no previous/next navigation.
        self.assertEqual(sum('class="p2-reading' in m.value for m in at.markdown), 1)
        self.assertFalse(any(b.label in {'Next match', 'Previous match'} for b in at.button))
        self.assertTrue(any('class="p2-hit"' in m.value for m in at.markdown))  # result list is still on the page
        self.assertFalse(at.query_params.get('paper'))  # no navigation to a reader URL

    def test_formulas_render_as_latex_and_corpus_text_is_escaped(self):
        at = self.app('search', query='semaglutde')
        formula = next(l.value for l in at.latex if 'transpose' in l.value)
        self.assertIn(r'a_i\neq b_j', formula)
        self.assertFalse(any('a_i' in c.value for c in at.caption))
        self.assertEqual(common.md_escape('glp_1*x'), r'glp\_1\*x')

    def test_background_training_saves_model_once(self):
        import time
        from ui.hw2 import training
        from src.word2vec import Word2VecModel
        config = Word2VecConfig(epochs=2, dimensions=4, min_count=1, sample=0, architecture='cbow')
        target = self.output / 'background_cbow.npz'
        sentences = embedding_sentences(self.corpus)
        job = training.start('test-key', sentences, config, self.corpus.metadata(), embedding_preprocessing(), target)
        self.assertIs(training.start('test-key', sentences, config, self.corpus.metadata(), embedding_preprocessing(), target), job)
        for _ in range(200):
            if job['status'] != 'running':
                break
            time.sleep(0.05)
        self.assertEqual(job['status'], 'done', job['error'])
        self.assertEqual(job['epoch'], 2)
        self.assertEqual(Word2VecModel.load(target).metadata['architecture'], 'CBOW with negative sampling')
        self.assertNotIn('test-key', training.running())

    def test_search_has_no_year_filter(self):
        at = self.app('search', query='semaglutide')  # fixture papers span 2025 and 2026
        self.assertFalse(any(s.label == 'Publication year' for s in at.selectbox))
        self.assertIn('3 matching documents', self.captions(at))

    def test_domains_compare_two_corpora(self):
        cid = 'b' * 20
        row = dict(id=cid, query='arXiv computer science: cs.AI (3 newest per primary category)', document_count=3,
                   created_at='2026-09-30', deleted_at=None)
        corpora = {common.MAIN: self.corpus, cid: self.other}
        with patch.object(common, 'list_corpora', return_value=[row]), \
                patch.object(common, 'corpus_by_id', side_effect=lambda signature, corpus_id=common.MAIN: corpora[corpus_id]):
            at = self.app('zipf')
        table = next(d.value for d in at.dataframe if 'Measure' in d.value.columns)
        self.assertEqual(list(table['Measure'])[:3], ['Documents', 'Tokens', 'Vocabulary'])
        self.assertIn('Top 10 terms', list(table['Measure']))
        self.assertTrue(any('Does domain specificity' in e.label for e in at.expander))
        self.assertIn('Computer science (arXiv)', table.columns)  # domains are fixed and named, no pickers
        self.assertFalse(any(b.label.startswith('Domain') for b in at.selectbox))
        self.assertFalse(any(t.label.startswith('Short name') for t in at.text_input))

    def test_shared_corpus_selection_and_archive_are_reversible(self):
        cid = 'a' * 20
        row = dict(id=cid, query='Extra topic', document_count=3, created_at='2026-09-30', deleted_at=None)
        state = {'archived': False}

        def listing(root, include_deleted=False):
            return [dict(row, deleted_at='2026-09-30' if state['archived'] else None)] if include_deleted or not state['archived'] else []

        def archive(root, chosen, value):
            self.assertEqual(chosen, cid)
            state['archived'] = value
        with patch.object(common, 'list_corpora', side_effect=listing), patch.object(common, 'set_archived', side_effect=archive):
            at = self.app('data')
            at.selectbox(key='_p2_corpus').set_value(cid).run()
            self.assertEqual(at.selectbox(key='_p2_corpus').value, cid)
            next(b for b in at.button if b.label == 'Archive corpus').click().run()
            self.assert_clean(at)
            self.assertTrue(state['archived'])
            self.assertEqual(at.selectbox(key='_p2_corpus').value, common.MAIN)
            next(b for b in at.button if b.label == 'Restore corpus').click().run()
            self.assert_clean(at)
            self.assertFalse(state['archived'])

    def test_fetch_error_keeps_current_corpus(self):
        with patch.object(common, 'fetch_corpus', side_effect=RuntimeError('network unavailable')):
            at = self.app('data')
            next(b for b in at.button if b.label == 'Fetch corpus').click().run()
            self.assert_clean(at)
            self.assertIn('not activated', at.error[0].value)
            self.assertEqual(at.selectbox(key='_p2_corpus').value, common.MAIN)


if __name__ == '__main__':
    unittest.main()
