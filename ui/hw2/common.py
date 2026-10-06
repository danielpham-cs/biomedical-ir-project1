"""Shared state, loading, widgets and the internal reader for the Homework 2 pages.

Page modules call these through the module (``common.experiment(...)``), so tests
can patch one place. Widget values live in permanent ``p2_*`` keys and are copied
into the URL for reader links, so a full page reload restores the same context.
"""
from __future__ import annotations

import hashlib
import json
import math
from html import escape
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from src.experiment_store import corpus_directory, fetch_arxiv_corpus, fetch_corpus, list_corpora, set_archived  # noqa: F401 (patched in tests)
from src.index import build_index
from src.library import hidden_article_ids
from src.matching import document_matches, highlight_spans, make_match_plan, searchable_text
from src.search import build_spelling_vocabulary
from src.tables import rows_csv
from src.word2vec import Word2VecConfig, Word2VecModel, model_key
from src.zipf_experiment import (CONDITIONS, VERSION, analyze_corpus, embedding_preprocessing, load_corpus,
                                 load_snapshot, term_documents, term_spans)
from ui.hw2.registry import BY_KEY, PAGES
from ui.i18n import dark_mode

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/raw/pubmed_abstract_meta.json'
OUTPUT = ROOT / 'output/project2_glp1'
EXPERIMENTS = ROOT / 'data/experiments'
MODELS = ROOT / 'data/homework2'
MAIN = 'glp1-main'
PARAMS = {'filter': '', 'page': '1', 'size': '25', 'condition': 'B', 'term': 'glp-1',
          'query': 'glp-1 receptor', 'mode': 'STEM', 'method': 'AND', 'edits': '1', 'corpus': MAIN}


# ---------------------------------------------------------------- loading

def file_stamp(path):
    return (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (str(path), 0, 0)


@st.cache_data(show_spinner=False)
def experiment(signature, corpus_id=MAIN):
    corpus = corpus_by_id(signature, corpus_id)
    directory = OUTPUT if corpus_id == MAIN else corpus_directory(EXPERIMENTS, corpus_id)
    path = directory / 'analysis.json'
    result = json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    from src.preprocess import _stop_words
    if (not result or result['corpus']['fingerprint'] != corpus.fingerprint
            or result['settings']['version'] != VERSION
            or result['settings']['stopwords'] != sorted(_stop_words())):
        result = analyze_corpus(corpus)
        if corpus_id != MAIN:
            from src.pubmed import _write_atomic
            _write_atomic(path, json.dumps(result, ensure_ascii=False).encode('utf8'))
    return corpus, result


@st.cache_resource(show_spinner=False, max_entries=6)
def corpus_by_id(signature, corpus_id=MAIN):
    """The course corpus from its manifest, or an extra corpus from its frozen snapshot."""
    if corpus_id == MAIN:
        return load_corpus(MANIFEST)
    corpus = load_snapshot(corpus_directory(EXPERIMENTS, corpus_id) / 'corpus_snapshot.json')
    hidden = hidden_article_ids(ROOT / 'data/library.sqlite3')
    if any(f'PMID{d.pmid}' in hidden for d in corpus.documents):
        raise ValueError('This snapshot includes an archived library paper. Restore it before reopening the complete experiment.')
    return corpus


def corpus_names():
    names = {MAIN: 'GLP-1 · 1,000 abstracts · course reference'}
    names.update({r['id']: f"{r['query'][:70]} · {r['document_count']} abstracts · {r['id'][:8]}" for r in list_corpora(EXPERIMENTS)})
    return names


def signature(corpus_id):
    paths = [MANIFEST, ROOT / 'data/library.sqlite3', OUTPUT / 'analysis.json'] + sorted((ROOT / 'data/raw').glob('PMID*.xml'))
    if corpus_id != MAIN:
        paths += [corpus_directory(EXPERIMENTS, corpus_id) / 'corpus_snapshot.json']
    return tuple(file_stamp(p) for p in paths)


def selected_output():
    corpus_id = st.session_state.get('p2_corpus', MAIN)
    return OUTPUT if corpus_id == MAIN else corpus_directory(EXPERIMENTS, corpus_id)


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
            config = Word2VecConfig.from_dict(meta['config'])
            expected = model_key(config, fingerprint, embedding_preprocessing('C'))
            if meta.get('model_key') != expected or expected in seen:
                continue
            seen.add(expected)
            name = 'CBOW' if config.architecture == 'cbow' else 'Skip-gram'
            catalog[path] = f'{name} · dim {config.dimensions} · window {config.window} · epochs {config.epochs} · min_count {config.min_count} · {expected[:8]}'
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return catalog


@st.cache_data(show_spinner=False, max_entries=8)
def checked_report(directory, signature, run_key):
    """Only expose published evidence when the complete recorded run still matches."""
    directory = Path(directory).resolve()
    try:
        meta = json.loads((directory / 'run_metadata.json').read_text(encoding='utf8'))
        files = meta.get('files', {})
        required = {'report.pdf', 'executive_summary.pdf', 'research_answers.json', 'skipgram.npz'}
        if meta.get('run_key') != run_key or not required.issubset(files):
            return False, meta
        for name, digest in files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False, meta
        return True, meta
    except (OSError, ValueError, KeyError):
        return False, {}


def report_status(result):
    output = selected_output()
    return checked_report(str(output), tuple(file_stamp(p) for p in sorted(output.rglob('*')) if p.is_file()), result['run_key'])


# ---------------------------------------------------------------- state

def init_state():
    if not st.session_state.get('p2_initialized'):
        for key, default in PARAMS.items():
            st.session_state['p2_' + key] = st.query_params.get(key, default)
        if 'theme' in st.query_params:
            st.session_state.ui_dark_mode = st.query_params['theme'] == 'dark'
        st.session_state.p2_initialized = True


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


def current_page():
    return BY_KEY[st.session_state.get('p2_page_key', 'overview')]


def url(page=None, **overrides):
    params = {key: st.session_state.get('p2_' + key, default) for key, default in PARAMS.items()}
    params.update(theme='dark' if dark_mode() else 'light')
    params.update(overrides)
    target = BY_KEY[page] if page else current_page()
    return f'/{target.url_path}?' + urlencode(params)


def link(title, page=None, **params):
    return f'<a href="{escape(url(page, **params), quote=True)}" target="_self">{escape(title)}</a>'


def go(page_key):
    st.switch_page(BY_KEY[page_key].file)


def md_escape(text):
    """Escape corpus text before it goes into markdown, so '_' or '*' never turn into italics or bold."""
    return ''.join('\\' + ch if ch in '\\`*_{}[]<>()#+!|~$' else ch for ch in str(text))


# ---------------------------------------------------------------- widgets

def csv_download(label, rows, name, key=None):
    st.download_button(label, rows_csv(rows).encode('utf-8-sig'), name, 'text/csv', key=key)


def html_table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows)
    st.markdown(f'<div class="p2-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def condition_picker(label='Preprocessing condition'):
    return select_field(label, 'condition', list(CONDITIONS), format_func=lambda c: f'{c} - {CONDITIONS[c]}')


def thin_ranks(rows, per_decade=120):
    """Rows to *draw* on a log-rank axis: log-spaced ranks plus both ends of every equal-CF plateau.

    Regression numbers always use all rows; this only stops the browser from drawing
    tens of thousands of overlapping tail points (the tail is a staircase of ties).
    """
    if len(rows) <= 2 * per_decade:
        return rows
    targets = {round(10 ** (i / per_decade)) for i in range(int(math.log10(rows[-1]['rank']) * per_decade) + 1)}
    last = len(rows) - 1
    return [r for i, r in enumerate(rows)
            if r['rank'] in targets or i in (0, last) or rows[i - 1]['CF'] != r['CF'] or rows[i + 1]['CF'] != r['CF']]


def chart(spec, key=None):
    color = '#c9d8dc' if dark_mode() else '#304651'
    spec['background'] = 'transparent'
    spec['config'] = {'axis': {'labelColor': color, 'titleColor': color, 'gridColor': '#566b7630'},
                      'legend': {'labelColor': color, 'titleColor': color}, 'view': {'stroke': None}}
    st.vega_lite_chart(spec=spec, width='stretch', key=key)


def section(number, title, note=''):
    """Numbered section heading so the presenter can say 'step 2 of this page'."""
    # Plain spans, not <h3>: Streamlit injects anchor spans into headings.
    st.markdown(f'<div class="p2-section"><span class="p2-num">{escape(str(number))}</span>'
                f'<span class="p2-title">{escape(title)}</span></div>', unsafe_allow_html=True)
    if note:
        st.caption(note)


def requirement_header(page):
    """Compact header: title, question, requirement badges and the presenter's lines for this minute."""
    badges = ''.join(f'<span class="p2-doc p2-doc-{doc.lower()}" title="{escape(text, quote=True)}">{escape(doc)} · {escape(ref)}</span>'
                     for doc, ref, text in page.requirements)
    position = [p.key for p in PAGES].index(page.key)
    script = ''.join(f'<li>{escape(line)}</li>' for line in page.script)
    st.markdown(
        f'<section class="p2-header"><p class="hx-eyebrow">Homework 2 · page {position} of {len(PAGES) - 1}</p>'
        f'<h1>{escape(page.title)}</h1><p class="p2-question">{escape(page.question)}</p>'
        f'<div class="p2-reqs">{badges}</div>'
        + (f'<details class="p2-script"><summary>Demo script</summary><ul>{script}</ul></details>' if script else '')
        + '</section>', unsafe_allow_html=True)


def card(title, note=''):
    """One bordered card per question, stacked on a single scrolling page."""
    box = st.container(border=True)
    box.markdown(f'<div class="p2-card-title">{escape(title)}</div>', unsafe_allow_html=True)
    if note:
        box.caption(note)
    return box


def more(title='More details', note='For questions after the demo.'):
    """Heading of the collapsed detail / answer blocks at the bottom of a page."""
    st.markdown(f'<div class="p2-more">{escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)


def answers(title, items, expanded=False):
    """The brief's questions with answers computed from the current run."""
    st.markdown(f'**{escape(title)}**')
    for question, answer in items.items():
        with st.expander(question, expanded=expanded):
            st.write(answer)


def inject_css():
    st.markdown('''<style>
      .p2-table {overflow-x:auto;border:1px solid var(--hx-line);border-radius:10px;}
      .p2-table table {width:100%;border-collapse:collapse;background:var(--hx-surface);color:var(--hx-ink);}
      .p2-table th {background:var(--hx-surface-alt);text-align:left;white-space:nowrap;}
      .p2-table th,.p2-table td {padding:10px 12px;border-bottom:1px solid var(--hx-line);}
      .p2-table a,.p2-result a {color:var(--hx-ink)!important;text-decoration:none;}
      .p2-table a:hover,.p2-table a:focus-visible,.p2-result a:hover,.p2-result a:focus-visible {color:var(--hx-teal)!important;text-decoration:underline;}
      .p2-result {margin-top:18px;font-weight:600;font-size:1.12rem;}
      .p2-hit {padding:0 0 14px;border-bottom:1px solid var(--hx-line);}
      [class*="st-key-p2_hit_"] {gap:0.1rem;margin-top:10px;}
      [class*="st-key-p2_open_"] button {padding:0;min-height:0;justify-content:flex-start;text-align:left;}
      [class*="st-key-p2_open_"] button p {font-weight:600;font-size:1.05rem;line-height:1.4;color:var(--hx-ink);}
      [class*="st-key-p2_open_"] button:hover p {color:var(--hx-teal);text-decoration:underline;}
      .p2-hit .p2-result {margin-top:14px;font-size:1.05rem;line-height:1.4;}
      .p2-hit-meta {color:var(--hx-muted);font-size:0.85rem;margin-top:4px;}
      .p2-hit-count {color:var(--hx-teal);font-weight:600;}
      .p2-hit-snippet {color:var(--hx-ink);font-size:0.92rem;line-height:1.6;margin-top:6px;}
      .p2-hit-snippet mark {background:#cfecd3;color:#163c29;border-radius:3px;padding:0 2px;}
      .p2-hit-pos {color:var(--hx-muted);font-size:0.78rem;margin-top:6px;font-variant-numeric:tabular-nums;}
      .p2-hit-pos b {color:var(--hx-ink);font-weight:600;}
      .p2-reading {color:var(--hx-ink);line-height:1.8;padding:12px 0;overflow-wrap:anywhere;}
      .p2-abstract {border-top:1px solid var(--hx-line);margin-top:18px;}
      .p2-reading mark {background:#cfecd3;color:#163c29;border-radius:3px;padding:1px 2px;}      .p2-header {border-bottom:1px solid var(--hx-line);padding-bottom:0.8rem;margin-bottom:0.8rem;}
      .p2-header h1 {margin:0.2rem 0 0.4rem;}
      .p2-question {font-size:1.08rem;color:var(--hx-muted);margin:0 0 0.8rem;}
      .p2-reqs {display:flex;flex-wrap:wrap;gap:6px;}
      .p2-script {margin-top:0.7rem;color:var(--hx-muted);font-size:0.9rem;}
      .p2-script summary {cursor:pointer;color:var(--hx-teal);}
      .p2-script ul {margin:0.3rem 0 0 1rem;padding:0;}
      .p2-doc {display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.78rem;
               border:1px solid var(--hx-line);background:var(--hx-surface-alt);color:var(--hx-ink);}
      .p2-doc-brief {border-color:#e3a36b;}
      .p2-doc-spec {border-color:var(--hx-teal);}
      .p2-section {display:flex;align-items:center;gap:10px;margin:1.4rem 0 0.5rem;padding-bottom:6px;border-bottom:1px solid var(--hx-line);}
      .p2-num {flex:none;display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;
               border:1.5px solid var(--hx-teal);color:var(--hx-teal);font-size:0.75rem;font-weight:700;line-height:1;}
      .p2-title {font-size:1.05rem;font-weight:650;color:var(--hx-ink);letter-spacing:-0.01em;}
      .p2-dp {overflow-x:auto;}
      .p2-dp table {border-collapse:collapse;font-family:ui-monospace,Consolas,monospace;font-size:0.9rem;}
      .p2-dp th,.p2-dp td {border:1px solid var(--hx-line);min-width:2.1rem;height:2.1rem;text-align:center;color:var(--hx-ink);}
      .p2-dp th {background:var(--hx-surface-alt);font-weight:400;color:var(--hx-muted);padding:0 6px;}
      .p2-dp th b {font-weight:700;color:var(--hx-teal);}
      .p2-dp td.path {background:#cfecd3;color:#163c29;font-weight:700;}
      .p2-dp td.edit {background:#f6d2b8;color:#5a2a0c;font-weight:700;}
      .p2-original {color:var(--hx-ink);font-size:0.95rem;line-height:1.6;margin:0.2rem 0 0.4rem;}
      .p2-original span {display:block;color:var(--hx-muted);font-size:0.8rem;margin-bottom:2px;}
      .p2-original s {color:var(--hx-muted);opacity:0.7;}
      .p2-steps {display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-family:ui-monospace,Consolas,monospace;}
      .p2-steps span {padding:4px 10px;border-radius:6px;border:1px solid var(--hx-line);background:var(--hx-surface);color:var(--hx-ink);}
      .p2-steps span.on {border-color:var(--hx-teal);background:#cfecd3;color:#163c29;}
      .p2-steps span.ctx {border-color:#e3a36b;background:#fbe8d9;color:#5a2a0c;}
      .p2-steps i {color:var(--hx-muted);font-style:normal;}
      .p2-card-title {font-size:1.08rem;font-weight:700;color:var(--hx-ink);margin-bottom:0.2rem;}
      .p2-keywords {color:var(--hx-ink);font-size:0.95rem;margin:0.3rem 0 0.6rem;}
      .p2-keywords s {color:var(--hx-muted);}
      .p2-keywords small {color:var(--hx-muted);}
      .p2-fix {color:#b4561c;}
      .p2-ok {color:var(--hx-teal);font-weight:700;}
      .p2-more {margin:1.6rem 0 0.2rem;font-weight:700;color:var(--hx-muted);text-transform:uppercase;font-size:0.78rem;letter-spacing:0.06em;}
      .p2-barcell {display:flex;align-items:center;gap:10px;min-width:14rem;}
      .p2-track {flex:0 0 11rem;height:12px;border-radius:4px;background:var(--hx-surface-alt);overflow:hidden;}
      .p2-bar {display:block;height:100%;border-radius:4px;background:var(--hx-teal);}
      .p2-barcell > span:last-child {white-space:nowrap;font-variant-numeric:tabular-nums;}
      .p2-change {color:var(--hx-muted);margin-left:4px;}
      .p2-nowrap {white-space:nowrap;margin-left:0;}
    </style>''', unsafe_allow_html=True)


# ---------------------------------------------------------------- reader

def paper_id(doc):
    """'PMID 12345' for PubMed abstracts, 'arXiv 2510.01234' for arXiv ones."""
    return f'arXiv {doc.pmid}' if doc.doc_id.startswith('arXiv:') else f'PMID {doc.pmid}'


def reader(corpus):
    pmid, source = st.query_params.get('paper', ''), st.query_params.get('source', 'corpus')
    st.markdown(f'<div class="p2-result">{link("← Back to " + current_page().title)}</div>', unsafe_allow_html=True)
    doc = next((d for d in corpus.documents if d.pmid == pmid), None)
    if doc is None:
        st.warning('This paper is archived or outside the current experiment corpus.')
        return
    st.subheader(doc.title)
    st.caption(f'{paper_id(doc)} · {doc.year or ""} · Abstract only')
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
        query = st.query_params.get('query', st.session_state.p2_query)
        try:
            text, spans, locations = search_matches(doc, query, st.session_state.p2_method, st.session_state.p2_mode, st.session_state.p2_edits)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.caption(f'Query: {query} · {st.session_state.p2_mode} / {st.session_state.p2_method}')
    paper_text(doc, text, spans, locations)


def search_matches(doc, query, method, mode, edits):
    """The searchable text of a paper and every location the query matches in it."""
    plan = make_match_plan(query, method=method, match_mode=mode, max_edits=int(edits))
    text = searchable_text(doc)
    matches = document_matches(doc, plan)
    return text, [(m.start, m.end) for m in matches], [m.row(text) for m in matches]


def paper_text(doc, text, spans, locations):
    """The full abstract with every match highlighted, then the table of match locations."""
    if spans:
        st.caption(f'{len(spans)} match{"es" if len(spans) != 1 else ""} highlighted.')
    st.markdown('<div class="p2-reading p2-abstract">' + highlight_spans(text, spans).replace('\n\n', '<br/><br/>') + '</div>',
                unsafe_allow_html=True)
    if locations:
        with st.expander(f'All {len(locations)} matched locations'):
            st.dataframe(locations, hide_index=True, width='stretch')
        st.download_button('Download match locations', rows_csv(locations).encode('utf-8-sig'), f'{doc.pmid}_locations.csv', 'text/csv',
                           key=f'p2_loc_csv_{doc.pmid}', on_click='ignore')
    # on_click='ignore': a download must not rerun the page, which would close an open popup.
    st.download_button('Download abstract', doc.abstract, f'{doc.pmid}_abstract.txt', 'text/plain', key=f'p2_abs_{doc.pmid}', on_click='ignore')


def open_popup(pmid):
    st.session_state.p2_popup = pmid


def close_popup():
    st.session_state.pop('p2_popup', None)


@st.dialog('Paper', width='large', on_dismiss=close_popup)
def paper_dialog(doc, query, method, mode, edits):
    """Search result opened in a popup: the result list stays in place behind it.

    The open paper is kept in session state (open_popup / close_popup), so the popup
    survives any full rerun until the viewer closes it.
    """
    st.subheader(doc.title)
    st.caption(f'{paper_id(doc)} · {doc.year or ""} · Abstract only · Query: {query} · {mode} / {method}')
    try:
        text, spans, locations = search_matches(doc, query, method, mode, edits)
    except ValueError as exc:
        st.error(str(exc))
        return
    paper_text(doc, text, spans, locations)
