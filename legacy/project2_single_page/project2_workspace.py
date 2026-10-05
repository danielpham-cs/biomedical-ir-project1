"""Seven lazy screens for the frozen, abstract-only Project 2 experiment."""
from __future__ import annotations
import json
import hashlib
import math
from html import escape
from pathlib import Path
from urllib.parse import urlencode
import streamlit as st
from src.homework2 import rows_csv
from src.index import build_index
from src.matching import highlight_spans, make_match_plan, document_matches, searchable_text
from src.search import search, build_spelling_vocabulary, suggest_query_correction
from src.word2vec import Word2VecConfig, Word2VecModel, train_skipgram, model_key
from src.zipf_experiment import (CONDITIONS, VERSION, load_corpus, load_snapshot, analyze_corpus, token_trace,
    tokens, term_documents, term_spans, embedding_sentences, embedding_preprocessing)
from ui.brand import sidebar_nav, inject_css, masthead, intro, footer
from ui.i18n import dark_mode
from ui.project2_embeddings import embedding_map
from src.experiment_store import list_corpora, fetch_corpus, set_archived, corpus_directory
from src.library import hidden_article_ids

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/raw/pubmed_abstract_meta.json'
OUTPUT = ROOT / 'output/project2_glp1'
EXPERIMENTS = ROOT / 'data/experiments'
VIEWS = ['Corpus', 'Preprocessing', 'Zipf analysis', 'CF / DF / IDF', 'Word2Vec', 'Search & spelling', 'Report & demo']
PARAMS = {'filter': '', 'page': '1', 'size': '25', 'condition': 'B', 'term': 'glp-1',
          'query': 'glp-1 receptor', 'mode': 'STEM', 'method': 'AND', 'edits': '1', 'corpus': 'glp1-main'}


def file_stamp(path):
    return (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (str(path), 0, 0)


@st.cache_data(show_spinner=False)
def experiment(signature, corpus_id='glp1-main'):
    if corpus_id == 'glp1-main':
        corpus = load_corpus(MANIFEST)
        directory = OUTPUT
    else:
        directory = corpus_directory(EXPERIMENTS, corpus_id)
        corpus = load_snapshot(directory/'corpus_snapshot.json')
        hidden = hidden_article_ids(ROOT/'data/library.sqlite3')
        if any(f'PMID{d.pmid}' in hidden for d in corpus.documents):
            raise ValueError('This snapshot includes an archived library paper. Restore it before reopening the complete experiment.')
    path = directory / 'analysis.json'
    result = json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    from src.preprocess import _stop_words
    if (not result or result['corpus']['fingerprint'] != corpus.fingerprint
            or result['settings']['version'] != VERSION
            or result['settings']['stopwords'] != sorted(_stop_words())):
        result = analyze_corpus(corpus)
        if corpus_id != 'glp1-main':
            from src.pubmed import _write_atomic
            _write_atomic(path, json.dumps(result, ensure_ascii=False).encode('utf8'))
    return corpus, result


def selected_output():
    corpus_id = st.session_state.get('p2_corpus', 'glp1-main')
    return OUTPUT if corpus_id == 'glp1-main' else corpus_directory(EXPERIMENTS, corpus_id)


def corpus_manager():
    st.subheader('Build and manage PubMed corpora')
    st.caption('The 1,000-paper GLP-1 course corpus stays available as the pinned reference. Additional corpora have their own snapshots, analysis and models.')
    with st.expander('Fetch a new corpus from PubMed'):
        with st.form('p2_fetch_corpus'):
            query = st.text_input('Query term', 'GLP-1', help='Disease, drug, or a full PubMed query. English and has-abstract filters are added.')
            count = st.number_input('Maximum documents', 1, 3000, 1000, step=1)
            fetch = st.form_submit_button('Fetch corpus')
        if fetch:
            bar = st.progress(0, text='Searching PubMed…')
            try:
                hidden = {p[4:] for p in hidden_article_ids(ROOT/'data/library.sqlite3') if p.startswith('PMID')}
                row = fetch_corpus(EXPERIMENTS, query, int(count), excluded_pmids=hidden,
                                   progress=lambda n, total: bar.progress(n/total, text=f'{n}/{total} validated English abstracts'))
                st.session_state.p2_corpus = row['id']
                st.session_state.p2_fetch_message = f"Saved {row['document_count']} abstracts (requested maximum {row['requested']})."
                st.session_state.pop('_p2_corpus', None)
                st.session_state.pop('p2_model_path', None)
                st.rerun()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f'Corpus not activated: {exc}. Downloaded batches are kept; retry the same query to resume.')
            finally: bar.empty()
    if st.session_state.get('p2_fetch_message'):
        st.success(st.session_state.p2_fetch_message)
    rows = list_corpora(EXPERIMENTS, include_deleted=True)
    if rows:
        st.dataframe([{'ID': r['id'], 'Query': r['query'], 'Documents': r['document_count'], 'Created': r['created_at'], 'Status': 'Archived' if r['deleted_at'] else 'Active'} for r in rows], hide_index=True, width='stretch')
        with st.expander('Archive / restore additional corpora'):
            by_id = {r['id']: r for r in rows}
            cid = st.selectbox('Corpus to manage', list(by_id), format_func=lambda x: f"{by_id[x]['query']} · {by_id[x]['document_count']} abstracts · {x[:8]}")
            archived = bool(by_id[cid]['deleted_at'])
            st.caption('Soft-delete only: XML, snapshots, statistics and model files are retained. Other corpora are unaffected.')
            if st.button('Restore corpus' if archived else 'Archive corpus'):
                set_archived(EXPERIMENTS, cid, not archived)
                if st.session_state.p2_corpus == cid and not archived:
                    st.session_state.p2_corpus = 'glp1-main'
                    st.session_state.pop('_p2_corpus', None)
                st.rerun()
    st.divider()


@st.cache_resource(show_spinner=False)
def retrieval_index(fingerprint, _documents):
    return build_index(_documents), build_spelling_vocabulary(_documents)


@st.cache_data(show_spinner=False, max_entries=32)
def documents_for_term(fingerprint, term, condition, _corpus):
    return term_documents(_corpus, term, condition)


@st.cache_resource(show_spinner=False, max_entries=4)
def saved_model(path, stamp):
    return Word2VecModel.load(Path(path))


@st.cache_data(show_spinner=False, max_entries=16)
def model_catalog(signature, fingerprint):
    import numpy as np
    catalog, seen = {}, set()
    for path, _, _ in signature:
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['metadata'].item()))
            config = Word2VecConfig(**meta['config'])
            expected = model_key(config, fingerprint, embedding_preprocessing('C'))
            if meta.get('model_key') != expected or expected in seen: continue
            seen.add(expected)
            catalog[path] = f'Skip-gram · dim {config.dimensions} · window {config.window} · epochs {config.epochs} · min_count {config.min_count} · {expected[:8]}'
        except (OSError, ValueError, KeyError):
            continue
    return catalog


@st.cache_data(show_spinner=False, max_entries=8)
def checked_report(directory, signature, run_key):
    """Only expose published evidence when the complete recorded run still matches."""
    directory = Path(directory).resolve()
    try:
        meta = json.loads((directory/'run_metadata.json').read_text(encoding='utf8'))
        files = meta.get('files', {})
        required = {'report.pdf', 'executive_summary.pdf', 'research_answers.json', 'skipgram.npz'}
        if meta.get('run_key') != run_key or not required.issubset(files):
            return False, meta
        for name, digest in files.items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False, meta
        return True, meta
    except (OSError, ValueError, KeyError):
        return False, {}


def init_state():
    if not st.session_state.get('p2_initialized'):
        for key, default in PARAMS.items():
            st.session_state['p2_' + key] = st.query_params.get(key, default)
        view = st.query_params.get('view', 'Corpus')
        st.session_state.p2_view = view if view in VIEWS else 'Corpus'
        if 'theme' in st.query_params:
            st.session_state.ui_dark_mode = st.query_params['theme'] == 'dark'
        st.session_state.p2_initialized = True
    if 'p2_next_view' in st.session_state:
        st.session_state.p2_view = st.session_state.pop('p2_next_view')
        st.session_state.pop('_p2_navigation', None)
    st.session_state.setdefault('_p2_navigation', st.session_state.p2_view)


def remember(key):
    st.session_state['p2_' + key] = st.session_state['_p2_' + key]
    if key in {'filter', 'size'}:
        st.session_state.p2_page = '1'
        st.session_state.pop('_p2_page', None)
    if key == 'corpus':
        st.session_state.pop('p2_model_path', None)
        st.session_state.p2_filter = ''
        st.session_state.p2_page = '1'
        st.session_state.pop('_p2_filter', None)
        st.session_state.pop('_p2_page', None)
        st.query_params.clear()


def text_field(label, key, **kwargs):
    st.session_state.setdefault('_p2_' + key, st.session_state['p2_' + key])
    return st.text_input(label, key='_p2_' + key, on_change=remember, args=(key,), **kwargs)


def select_field(label, key, options, **kwargs):
    value = st.session_state['p2_' + key]
    if value not in options:
        value = options[0]
        st.session_state['p2_' + key] = value
    st.session_state.setdefault('_p2_' + key, value)
    if st.session_state['_p2_' + key] not in options:
        st.session_state['_p2_' + key] = value
    return st.selectbox(label, options, key='_p2_' + key, on_change=remember, args=(key,), **kwargs)


def url(**overrides):
    params = {key: st.session_state.get('p2_' + key, default) for key, default in PARAMS.items()}
    params.update(view=st.session_state.p2_view, theme='dark' if dark_mode() else 'light')
    params.update(overrides)
    return '/Homework_2?' + urlencode(params)


def link(title, **params):
    return f'<a href="{escape(url(**params), quote=True)}" target="_self">{escape(title)}</a>'


def navigate(view):
    st.session_state.p2_next_view = view
    st.query_params.clear()
    st.rerun()


def csv_download(label, rows, name):
    st.download_button(label, rows_csv(rows).encode('utf-8-sig'), name, 'text/csv')


def html_table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows)
    st.markdown(f'<div class="p2-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def condition_picker():
    return select_field('Preprocessing condition', 'condition', list(CONDITIONS), format_func=lambda c: f'{c} - {CONDITIONS[c]}')


def chart(spec):
    color = '#c9d8dc' if dark_mode() else '#304651'
    spec['background'] = 'transparent'
    spec['config'] = {'axis': {'labelColor': color, 'titleColor': color, 'gridColor': '#566b7630'},
                      'legend': {'labelColor': color, 'titleColor': color}, 'view': {'stroke': None}}
    st.vega_lite_chart(spec=spec, width='stretch')


def corpus_panel(corpus, result):
    st.write('Inspect the experiment corpus')
    a, b, c = st.columns(3)
    a.metric('English abstracts', len(corpus.documents))
    b.metric('Unique PMIDs', len({d.pmid for d in corpus.documents}))
    c.metric('Pipeline', 'A–D')
    st.caption('Frozen selection in PubMed returned order. Analysis uses abstract text only; titles, keywords and full text are excluded.')
    with st.expander('Source query and snapshot'):
        st.code(corpus.query, language=None)
        st.code(corpus.fingerprint, language=None)
        st.write('Source XML and SHA-256 hashes are retained. Archived papers are excluded. This is a single-topic, non-random sample.')
    query = text_field('Search papers by title or PMID', 'filter', placeholder='Enter part of a paper title or PMID').strip().casefold()
    matches = [d for d in corpus.documents if not query or query in d.title.casefold() or query in d.pmid]
    left, right = st.columns(2)
    with left:
        size = int(select_field('Rows per page', 'size', ['25', '50']))
    with right:
        pages = max(1, math.ceil(len(matches) / size))
        page = int(select_field('Page', 'page', [str(n) for n in range(1, pages + 1)]))
    start = (page - 1) * size
    st.caption(f'{len(matches):,} matches of {len(corpus.documents):,} papers. Showing {start+1 if matches else 0}-{min(start+size, len(matches))}. Titles open the internal reader.')
    if not matches:
        st.info('No papers match this title or PMID. Clear the search to show the whole corpus.')
        return
    html_table(['PMID', 'Title', 'Year'], [[escape(d.pmid), link(d.title, paper=d.pmid, source='corpus'), escape(str(d.year or ''))] for d in matches[start:start+size]])
    csv_download('Download corpus list', [{'PMID': d.pmid, 'Title': d.title, 'Year': d.year} for d in matches], 'corpus.csv')


def preprocessing_panel(corpus, result):
    html_table(['Condition', 'Cumulative steps'], [[c, escape(v)] for c, v in CONDITIONS.items()])
    st.dataframe([dict(condition=c, **v['stats']) for c, v in result['conditions'].items()], hide_index=True, width='stretch')
    sandbox = st.toggle('Try your own text (sandbox)', value=False)
    if sandbox:
        st.caption('Sandbox only. This text does not change corpus counts, models or reports.')
        raw = st.text_area('Text to inspect', 'The GLP-1 receptors, patients and treatments were studied.', height=100)
    else:
        doc = st.selectbox('Choose an abstract', corpus.documents, format_func=lambda d: f'{d.pmid} | {d.title}', key='p2_preprocess_doc')
        raw = doc.abstract
    condition = condition_picker()
    left, right = st.columns(2)
    with left:
        st.write('Original abstract' if not sandbox else 'Original sandbox text')
        st.text(raw)
    with right:
        words = tokens(raw, condition)
        st.write(f'Condition {condition}: {len(words)} tokens / {len(set(words))} unique')
        st.code(' | '.join(words), language=None, wrap_lines=True)
    trace = token_trace(raw)
    st.dataframe(trace, hide_index=True, width='stretch')
    csv_download('Download token trace', trace, 'token_trace.csv')
    st.subheader('Porter: compare C to D')
    c, d = (result['conditions'][k]['stats'] for k in ['C', 'D'])
    st.write(f"Vocabulary: {c['vocabulary']:,} → {d['vocabulary']:,} ({1-d['vocabulary']/c['vocabulary']:.1%} reduction). Same corpus and stopword list.")
    group_query = st.text_input('Find a Porter group', placeholder='patient, receptor, treat...').casefold()
    groups = [r for r in result['merged_groups'] if group_query in r['forms'] or group_query in r['stem']]
    st.dataframe(groups[:100], hide_index=True, width='stretch')
    csv_download('Download all matching Porter groups', groups, 'porter_groups.csv')
    with st.expander('Exact stopword list and preprocessing choices'):
        st.json(result['settings'])


def zipf_panel(corpus, result):
    condition = condition_picker()
    data = result['conditions'][condition]
    region = st.selectbox('Rank region', list(data['fits']))
    fit = data['fits'][region]
    st.caption(f"Condition {condition} · ranks {fit['rank_start']}-{fit['rank_end']} · {fit['points']:,} points. Regions: first 10%, next 70%, last 20%.")
    for col, key, label in zip(st.columns(5), ['slope', 'intercept', 'zipf_exponent', 'r_squared', 'rmse_log10'], ['Slope', 'Intercept', 'k = -slope', 'R²', 'RMSE (log10 CF)']):
        col.metric(label, 'N/A' if fit[key] is None else f'{fit[key]:.4f}')
    st.caption(fit['note'])
    rows = data['rows'][(fit['rank_start'] or 1)-1:fit['rank_end']]
    values = []
    for r in rows:
        values.extend([dict(rank=r['rank'], frequency=r['CF'], series='Observed CF', term=r['term']),
                       dict(rank=r['rank'], frequency=data['rows'][0]['CF']/r['rank'], series='1/r reference', term='')])
        if fit['slope'] is not None:
            values.append(dict(rank=r['rank'], frequency=10**fit['intercept']*r['rank']**fit['slope'], series='OLS fitted', term=''))
    for col, scale in zip(st.columns(2), ['linear', 'log']):
        with col:
            chart({'data': {'values': values}, 'mark': 'line', 'height': 290,
                   'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': scale}},
                                'y': {'field': 'frequency', 'type': 'quantitative', 'scale': {'type': scale}},
                                'color': {'field': 'series'}, 'tooltip': [{'field': x} for x in ['series', 'rank', 'term', 'frequency']]}})
    st.caption('OLS: log10(CF) = intercept + slope × log10(rank). The head-anchored 1/r curve is a reference, not the regression.')
    st.dataframe([dict(region=name, **f) for name, f in data['fits'].items()], hide_index=True, width='stretch')
    if st.toggle('Compare conditions A–D'):
        chart({'data': {'values': [dict(rank=r['rank'], CF=r['CF'], condition=c) for c, v in result['conditions'].items() for r in v['rows']]}, 'mark': 'line', 'height': 330,
               'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': 'log'}},
                            'y': {'field': 'CF', 'type': 'quantitative', 'scale': {'type': 'log'}}, 'color': {'field': 'condition'}}})
    if st.toggle('Residuals and frequency spectrum'):
        if fit['slope'] is not None:
            residuals = [dict(rank=r['rank'], residual=math.log10(r['CF'])-fit['intercept']-fit['slope']*math.log10(r['rank'])) for r in rows]
            chart({'data': {'values': residuals}, 'mark': 'point', 'height': 220, 'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': 'log'}}, 'y': {'field': 'residual', 'type': 'quantitative'}}})
        chart({'data': {'values': data['spectrum']}, 'mark': 'point', 'height': 220, 'encoding': {'x': {'field': 'frequency', 'type': 'quantitative', 'scale': {'type': 'log'}}, 'y': {'field': 'terms', 'type': 'quantitative', 'scale': {'type': 'log'}}}})
    st.subheader('Top 50 and full vocabulary')
    q = st.text_input('Filter vocabulary').casefold()
    filtered = [r for r in data['rows'] if q in r['term']]
    st.caption(f'{len(filtered):,} matching terms; first 50 displayed.')
    st.dataframe(filtered[:50], hide_index=True, width='stretch')
    csv_download('Download full frequency table', data['rows'], f'{condition}_frequency.csv')
    csv_download('Download regional fits', [dict(region=r, **f) for r, f in data['fits'].items()], f'{condition}_fits.csv')
    output = selected_output()
    current, _ = checked_report(str(output), tuple(file_stamp(p) for p in sorted(output.rglob('*')) if p.is_file()), result['run_key'])
    if current:
        with st.expander('Export published figures (all four conditions)'):
            for name in ['rank_frequency', 'loglog_fit', 'residuals']:
                path = output/'figures'/f'{name}.pdf'
                if path.exists(): st.download_button('Download ' + name.replace('_', ' '), path.read_bytes(), path.name, 'application/pdf')
    st.info('A high R² describes approximate linearity in log space. It does not prove a power law; tied low frequencies and region choices affect the result.')


def term_panel(corpus, result):
    condition = condition_picker()
    data = result['conditions'][condition]
    st.latex(r'\mathrm{IDF}(t)=\ln\!\left(\frac{N}{\mathrm{DF}(t)}\right),\quad \mathrm{TFIDF}(t,d)=\mathrm{TF}(t,d)\,\mathrm{IDF}(t)')
    st.caption(f'N = {len(corpus.documents):,}; natural logarithm. CF counts occurrences; DF counts documents. This textbook IDF is separate from BM25 scoring.')
    st.dataframe(data['selected_terms'], hide_index=True, width='stretch')
    chart({'data': {'values': data['selected_terms']}, 'mark': {'type': 'circle', 'size': 95}, 'height': 280,
           'encoding': {'x': {'field': 'DF', 'type': 'quantitative', 'scale': {'type': 'log'}},
                        'y': {'field': 'CF', 'type': 'quantitative', 'scale': {'type': 'log'}}, 'tooltip': [{'field': x} for x in ['term', 'CF', 'DF', 'IDF']]}})
    csv_download('Download selected CF / DF / IDF terms', data['selected_terms'], f'{condition}_selected_terms.csv')
    term = select_field('Inspect a term (type to search vocabulary)', 'term', [r['term'] for r in data['rows']])
    row = next(r for r in data['rows'] if r['term'] == term)
    for col, field in zip(st.columns(4), ['CF', 'DF', 'DF_fraction', 'IDF']):
        col.metric(field, f'{row[field]:.4f}' if isinstance(row[field], float) else row[field])
    docs = documents_for_term(corpus.fingerprint, term, condition, corpus)
    st.caption(f"{len(docs)} documents contain '{term}'. First 50 shown by raw TF. Open a title to verify the condition-aware matches.")
    html_table(['PMID', 'Title', 'TF', 'TF × IDF'], [[escape(r['PMID']), link(r['Title'], paper=r['PMID'], source='term'), str(r['TF']), f"{r['TF_IDF']:.4f}"] for r in docs[:50]])
    csv_download('Download all documents for term', docs, 'term_documents.csv')


def embeddings_panel(corpus, result):
    prep = embedding_preprocessing('C')
    default = selected_output() / 'skipgram.npz'
    path = Path(st.session_state.get('p2_model_path', str(default)))
    st.caption('Skip-gram with negative sampling. Condition C; windows stay within sentences. The published model uses the complete corpus.')
    with st.expander('Training settings / train another model'):
        with st.form('p2_train'):
            a, b, c = st.columns(3)
            dims = a.selectbox('Dimensions', [32, 64, 100, 128], index=1)
            window = b.number_input('Context window', 1, 15, 5)
            epochs = c.number_input('Epochs', 1, 100, 15)
            a, b, c = st.columns(3)
            minimum = a.number_input('Minimum count', 1, 100, 2)
            negatives = b.number_input('Negative samples', 1, 20, 5)
            seed = c.number_input('Seed', 0, 2147483647, 42)
            submitted = st.form_submit_button('Train Skip-gram on current corpus')
        if submitted:
            config = Word2VecConfig(dimensions=dims, window=int(window), epochs=int(epochs), min_count=int(minimum), negatives=int(negatives), seed=int(seed))
            key = model_key(config, corpus.fingerprint, prep)
            path = ROOT / 'data/homework2' / f'skipgram_{key}.npz'
            bar = st.progress(0)
            try:
                model = train_skipgram(embedding_sentences(corpus), config, corpus_metadata=corpus.metadata(), preprocessing=prep,
                                       progress=lambda epoch, total, loss: bar.progress(epoch/total, text=f'Epoch {epoch}/{total} · probe loss {loss:.4f}'))
                model.save(path)
                st.session_state.p2_model_path = str(path)
                st.session_state.p2_select_trained_model = True
                st.success('Model saved. The published report still describes its recorded model.')
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            finally:
                bar.empty()
        if path != default and st.button('Use published model'):
            st.session_state.pop('p2_model_path', None)
            st.session_state.p2_select_trained_model = True
            st.rerun()
    candidates = list((ROOT/'data/homework2').glob('skipgram_*.npz'))
    if default.exists(): candidates.insert(0, default)
    catalog = model_catalog(tuple(file_stamp(p) for p in candidates), corpus.fingerprint)
    if catalog:
        options = list(catalog)
        preferred = str(path) if str(path) in options else options[0]
        if st.session_state.pop('p2_select_trained_model', False):
            st.session_state.pop('p2_model_selector', None)
        if st.session_state.get('p2_model_selector') not in options:
            st.session_state.p2_model_selector = preferred
        chosen = st.selectbox('Saved models for this corpus', options, key='p2_model_selector', format_func=lambda p: catalog[p])
        path = Path(chosen)
    if not path.exists():
        st.info('No trained model is available. Train explicitly above, or run scripts/run_project2.py.')
        return
    try:
        model = saved_model(str(path), file_stamp(path))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load this model: {exc}. Train a new model to recover.')
        return
    config = Word2VecConfig(**model.metadata['config'])
    key = model_key(config, corpus.fingerprint, prep)
    if model.metadata.get('model_key') != key:
        st.warning('Saved model belongs to a different corpus or pipeline. Train a compatible model above.')
        return
    a, b, c = st.columns(3)
    a.metric('Trained words', len(model.words)); b.metric('Dimensions', model.vectors.shape[1]); c.metric('Epochs', config.epochs)
    chart({'data': {'values': model.metadata['history']}, 'mark': {'type': 'line', 'point': True}, 'height': 200,
           'encoding': {'x': {'field': 'epoch', 'type': 'quantitative'}, 'y': {'field': 'probe_loss', 'type': 'quantitative', 'scale': {'zero': False}}}})
    st.caption(model.metadata['loss_note'])
    query = st.text_input('Explore a word', 'glp-1', key='p2_embedding_query')
    words = tokens(query, 'C')
    valid = len(words) == 1 and words[0] in model.words
    if valid:
        neighbors = model.most_similar(words[0], 10)
        st.dataframe(neighbors, hide_index=True, width='stretch')
        csv_download('Download nearest words', neighbors, 'nearest_words.csv')
    else:
        st.info('Enter one word in the trained vocabulary. Try glp-1, semaglutide, receptor or insulin. OOV terms have no learned vector.')
    embedding_map(model, words[0] if valid else None, key)
    with st.expander('Model configuration and corpus evidence'):
        st.json(model.metadata)
    st.download_button('Download trained model', model.to_bytes(), 'skipgram.npz', 'application/octet-stream')


def search_panel(corpus, result):
    index, vocabulary = retrieval_index(corpus.fingerprint, corpus.documents)
    query = text_field('Search abstracts', 'query', placeholder='glp-1 receptor, semaglutde...')
    a, b = st.columns(2)
    with a:
        mode = select_field('Matching', 'mode', ['STEM', 'PARTIAL', 'FUZZY'])
    with b:
        method = select_field('Query logic', 'method', ['AND', 'OR', 'PHRASE'] if mode == 'STEM' else ['AND', 'OR'])
    edits = int(select_field('Maximum edit distance for fuzzy matching', 'edits', ['1', '2'])) if mode == 'FUZZY' else 1
    st.caption('BM25 ranking; abstract-only index. Standard matching uses the existing retrieval tokenizer and NLTK Porter, separate from A–D. Fuzzy uses the selected edit-distance limit; spelling correction proposes a replacement query.')
    if not query.strip():
        st.info('Enter a query to search the frozen GLP-1 corpus.')
        return
    suggestion = suggest_query_correction(query, index, vocabulary)
    if suggestion:
        st.write(f'Did you mean: **{suggestion}**')
        if st.button('Apply spelling suggestion'):
            st.session_state.p2_query = suggestion
            st.session_state.pop('_p2_query', None)
            st.rerun()
    hits = search(query, index, corpus.documents, method=method, match_mode=mode, max_edits=edits, top_k=len(corpus.documents))
    st.caption(f'{len(hits)} matching documents; first 20 displayed.')
    if not hits:
        st.info('No matches. Try OR, Partial or a corrected query.')
    for hit in hits[:20]:
        st.markdown(f'<div class="p2-result">{link(hit.title, paper=hit.pmid, source="search")}</div>', unsafe_allow_html=True)
        st.caption(f'PMID {hit.pmid} · BM25 {hit.score:.4f} · {len(hit.occurrences)} matched locations')
        st.markdown(f'<div class="p2-reading">{hit.snippet_html}</div>', unsafe_allow_html=True)
    if hits:
        csv_download('Download search results', [{'PMID': h.pmid, 'Title': h.title, 'BM25': h.score, 'Matches': len(h.occurrences)} for h in hits], 'search_results.csv')


def reader(corpus):
    pmid, source = st.query_params.get('paper', ''), st.query_params.get('source', 'corpus')
    st.markdown(f'<div class="p2-result">{link("← Back to " + st.session_state.p2_view)}</div>', unsafe_allow_html=True)
    doc = next((d for d in corpus.documents if d.pmid == pmid), None)
    if doc is None:
        st.warning('This paper is archived or outside the current experiment corpus.')
        return
    st.subheader(doc.title)
    st.caption(f'PMID {doc.pmid} · {doc.year or ""} · Abstract only')
    spans, locations = [], []
    text = searchable_text(doc) if source == 'search' else doc.abstract
    if source == 'term':
        condition, term = st.session_state.p2_condition, st.session_state.p2_term
        if condition not in CONDITIONS:
            st.error('Invalid preprocessing condition. Return to the term table.')
            return
        spans = term_spans(doc.abstract, term, condition)
        locations = [{'Matched text': doc.abstract[a:b], 'Start': a, 'End': b} for a, b in spans]
        st.caption(f'Condition {condition}: {term} · TF = {len(spans)}. Highlights use this condition, not an earlier query.')
    elif source == 'search':
        try:
            plan = make_match_plan(st.session_state.p2_query, method=st.session_state.p2_method, match_mode=st.session_state.p2_mode, max_edits=int(st.session_state.p2_edits))
        except ValueError as exc:
            st.error(str(exc)); return
        matches = document_matches(doc, plan)
        spans = [(m.start, m.end) for m in matches]
        locations = [m.row(text) for m in matches]
        st.caption(f'Query: {st.session_state.p2_query} · {st.session_state.p2_mode} / {st.session_state.p2_method}')
    if spans:
        position_key = f'p2_occurrence_{pmid}_{source}_{st.session_state.p2_condition}_{st.session_state.p2_term}_{st.session_state.p2_query}'
        st.session_state.setdefault(position_key, 0)
        pos = min(st.session_state[position_key], len(spans)-1)
        left, center, right = st.columns([1, 3, 1])
        if left.button('Previous match', disabled=pos == 0):
            st.session_state[position_key] = pos-1; st.rerun()
        if right.button('Next match', disabled=pos == len(spans)-1):
            st.session_state[position_key] = pos+1; st.rerun()
        center.write(f'Match {pos+1} of {len(spans)}')
        a, b = spans[pos]; start = max(0, a-120); end = min(len(text), b+160)
        st.markdown('<div class="p2-reading">'+highlight_spans(text[start:end], [(a-start, b-start)])+'</div>', unsafe_allow_html=True)
    st.markdown('<div class="p2-reading p2-abstract">'+highlight_spans(text, spans).replace('\n\n', '<br/><br/>')+'</div>', unsafe_allow_html=True)
    if locations:
        with st.expander(f'All {len(locations)} matched locations'):
            st.dataframe(locations, hide_index=True, width='stretch')
        csv_download('Download match locations', locations, f'{pmid}_locations.csv')
    st.download_button('Download abstract', doc.abstract, f'{pmid}_abstract.txt', 'text/plain')


def report_panel(corpus, result):
    output = selected_output()
    current, meta = checked_report(str(output), tuple(file_stamp(p) for p in sorted(output.rglob('*')) if p.is_file()), result['run_key'])
    st.dataframe([{'Deliverable': name, 'Status': status} for name, status in [
        ('Corpus and A–D analysis', f'Computed on {len(corpus.documents)} abstracts'),
        ('Published report and model', 'Same experiment run' if current else 'Missing or stale: run scripts/run_project2.py'),
        ('Individual presentation', 'Use the walkthrough below; present and explain the results yourself')]], hide_index=True, width='stretch')
    if current:
        answers = json.loads((output/'research_answers.json').read_text(encoding='utf8'))
        for title, answer in answers.items():
            with st.expander(title): st.write(answer)
        for name, label, mime in [('report.pdf', 'Download full report', 'application/pdf'), ('executive_summary.pdf', 'Download one-page executive summary', 'application/pdf'), ('submission.zip', 'Download submission bundle', 'application/zip'), ('report.md', 'Download editable report (Markdown)', 'text/markdown')]:
            path = output/name
            if path.exists(): st.download_button(label, path.read_bytes(), name, mime)
        with st.expander('Run provenance'): st.json(meta)
    else:
        st.warning('Published downloads are hidden because they do not match the current experiment. Regenerate the complete run first.')
        if st.session_state.p2_corpus != 'glp1-main':
            st.code(f'python scripts/run_project2.py --snapshot "{output / "corpus_snapshot.json"}" --output "{output}"', language='powershell')
    st.subheader('Individual demo · 10–12 minutes')
    steps = [
        ('Corpus', '0:00–1:00 · Show the query and 1,000 unique PMIDs. Filter by title/PMID and open an internal paper.'),
        ('Preprocessing', '1:00–2:30 · Trace one abstract through A–D. Explain stopwords and the vocabulary change after Porter.'),
        ('Zipf analysis', '2:30–5:00 · Compare linear/log plots, k, R² and RMSE. Inspect head/middle/tail and explain undefined tail R².'),
        ('CF / DF / IDF', '5:00–6:30 · Compare and, glp-1 and a concentrated term such as cdr in condition B. Open its documents and verify TF.'),
        ('Word2Vec', '6:30–8:00 · Explain center → context, negative samples and training loss. Inspect glp-1 neighbors and PCA variance.'),
        ('Search & spelling', '8:00–10:00 · Try semaglutde; apply its suggestion. Show Partial/Fuzzy, open a result, navigate matches and return.'),
        ('Report & demo', '10:00–12:00 · Explain RQ1–RQ5, the final challenge and limitations. Download the summary and reproducible bundle.')]
    for view, text in steps:
        st.write(text)
        if view != 'Report & demo' and st.button('Open ' + view, key='demo_' + view): navigate(view)
    st.code('.\\.venv\\python.exe scripts\\run_project2.py', language='powershell')


def main():
    st.set_page_config(page_title='Project 2 | Helix', page_icon='🧬', layout='wide')
    init_state(); sidebar_nav(); inject_css()
    st.markdown('''<style>
      .p2-table {overflow-x:auto;border:1px solid var(--hx-line);border-radius:10px;}
      .p2-table table {width:100%;border-collapse:collapse;background:var(--hx-surface);color:var(--hx-ink);}
      .p2-table th {background:var(--hx-surface-alt);text-align:left;}
      .p2-table th,.p2-table td {padding:12px;border-bottom:1px solid var(--hx-line);}
      .p2-table a,.p2-result a {color:var(--hx-ink)!important;text-decoration:none;}
      .p2-table a:hover,.p2-table a:focus-visible,.p2-result a:hover,.p2-result a:focus-visible {color:var(--hx-teal)!important;text-decoration:underline;}
      .p2-result {margin-top:18px;font-weight:600;font-size:1.12rem;}
      .p2-reading {color:var(--hx-ink);line-height:1.8;padding:12px 0;overflow-wrap:anywhere;}
      .p2-abstract {border-top:1px solid var(--hx-line);margin-top:18px;}
      .p2-reading mark {background:#cfecd3;color:#163c29;border-radius:3px;padding:1px 2px;}
    </style>''', unsafe_allow_html=True)
    def change_view():
        st.session_state.p2_view = st.session_state._p2_navigation
        st.query_params.clear()
    st.sidebar.radio('Project 2 walkthrough', VIEWS, key='_p2_navigation', on_change=change_view)
    masthead('Project 2 / PubMed experiments')
    intro('EXPERIMENT WORKSPACE', st.session_state.p2_view, 'From source abstracts to measured text distributions, word vectors and document retrieval.')
    if st.session_state.p2_view == 'Corpus' and not st.query_params.get('paper'):
        corpus_manager()
    corpora = list_corpora(EXPERIMENTS)
    names = {'glp1-main': 'GLP-1 · 1,000 abstracts · course reference'}
    names.update({r['id']: f"{r['query']} · {r['document_count']} abstracts · {r['id'][:8]}" for r in corpora})
    if st.session_state.p2_corpus not in names:
        st.warning('Requested corpus is archived or unavailable. Return to the Corpus screen to restore it.')
        if st.button('Open reference corpus'):
            st.session_state.p2_corpus = 'glp1-main'
            st.session_state.pop('_p2_corpus', None)
            st.query_params.clear()
            st.rerun()
        return
    corpus_id = select_field('Active experiment corpus', 'corpus', list(names), format_func=lambda cid: names[cid])
    paths = [MANIFEST, ROOT/'data/library.sqlite3', OUTPUT/'analysis.json'] + sorted((ROOT/'data/raw').glob('PMID*.xml'))
    if corpus_id != 'glp1-main':
        paths += [selected_output()/'corpus_snapshot.json']
    try:
        with st.spinner('Loading the current experiment…'):
            corpus, result = experiment(tuple(file_stamp(p) for p in paths), corpus_id)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load a valid experiment: {exc}')
        st.code('.\\.venv\\python.exe scripts\\download_glp1_corpus.py --help')
        return
    st.caption(f'{len(corpus.documents):,} abstracts · corpus {corpus.fingerprint[:12]} · abstract-only · conditions A–D')
    if st.query_params.get('paper'):
        reader(corpus)
    else:
        panels = dict(zip(VIEWS, [corpus_panel, preprocessing_panel, zipf_panel, term_panel, embeddings_panel, search_panel, report_panel]))
        panels[st.session_state.p2_view](corpus, result)
        st.divider()
        left, right = st.columns(2)
        pos = VIEWS.index(st.session_state.p2_view)
        if pos and left.button('← ' + VIEWS[pos-1]): navigate(VIEWS[pos-1])
        if pos < len(VIEWS)-1 and right.button(VIEWS[pos+1] + ' →'): navigate(VIEWS[pos+1])
    footer()
