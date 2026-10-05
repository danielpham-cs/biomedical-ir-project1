"""Frame shared by every Homework 2 page: header, corpus context, reader and step navigation."""
from __future__ import annotations

import streamlit as st

from ui.brand import footer, inject_css, masthead, sidebar_nav
from ui.hw2 import common
from ui.hw2.registry import BY_KEY, PAGES


def corpus_context():
    """Select the experiment corpus shared by all pages; return (corpus, result) or None."""
    names = common.corpus_names()
    if st.session_state.p2_corpus not in names:
        st.warning('Requested corpus is archived or unavailable. Restore it on the Data page (Details → Manage corpora).')
        if st.button('Open reference corpus'):
            st.session_state.p2_corpus = common.MAIN
            st.session_state.pop('_p2_corpus', None)
            st.query_params.clear()
            st.rerun()
        return None
    with st.sidebar:
        corpus_id = common.select_field('Active experiment corpus', 'corpus', list(names), format_func=lambda cid: names[cid])
    try:
        with st.spinner('Loading the current experiment…'):
            corpus, result = common.experiment(common.signature(corpus_id), corpus_id)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load a valid experiment: {exc}')
        st.code('.\\.venv\\python.exe scripts\\download_glp1_corpus.py --help')
        return None
    st.sidebar.caption(f'{len(corpus.documents):,} abstracts · corpus {corpus.fingerprint[:12]} · abstract-only · conditions A–D')
    return corpus, result


def step_navigation(key):
    keys = [p.key for p in PAGES]
    position = keys.index(key)
    st.divider()
    left, right = st.columns(2)
    if position and left.button('← ' + PAGES[position - 1].title, key='p2_prev'):
        common.go(keys[position - 1])
    if position < len(keys) - 1 and right.button(PAGES[position + 1].title + ' →', key='p2_next'):
        common.go(keys[position + 1])


def run_page(key, panel, *, before=None, needs_corpus=True):
    """Render one page. `before` runs ahead of corpus loading (e.g. corpus management)."""
    page = BY_KEY[key]
    st.set_page_config(page_title=f'HW2 · {page.title} | Helix', page_icon='🧬', layout='wide')
    common.init_state()
    st.session_state.p2_page_key = key
    sidebar_nav()
    inject_css()
    common.inject_css()
    masthead('Homework 2 / Zipf · Porter · Word2Vec · Edit distance')
    reading = bool(st.query_params.get('paper'))
    if not reading:
        common.requirement_header(page)
        if before:
            before()
    if not needs_corpus:
        panel()
    else:
        loaded = corpus_context()
        if loaded is None:
            footer()
            return
        if reading:
            common.reader(loaded[0])
        else:
            panel(*loaded)
    if not reading:
        step_navigation(key)
    footer()
