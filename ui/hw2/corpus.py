"""Corpus sections: dataset checks (Spec · Dataset), vocabulary (Spec · Part II), browsing and corpus management."""
import math
from html import escape

import streamlit as st

from src.library import hidden_article_ids
from ui.hw2 import common


def manager():
    """Fetch another PubMed corpus, or archive/restore one."""
    st.caption('The 1,000-paper GLP-1 course corpus stays available as the pinned reference. Additional corpora have their own snapshots, analysis and models.')
    with st.form('p2_fetch_corpus'):
        query = st.text_input('Query term', 'GLP-1', help='Disease, drug, or a full PubMed query. English and has-abstract filters are added.')
        count = st.number_input('Maximum documents', 1, 3000, 1000, step=1)
        fetch = st.form_submit_button('Fetch corpus')
    if fetch:
        bar = st.progress(0, text='Searching PubMed…')
        try:
            hidden = {p[4:] for p in hidden_article_ids(common.ROOT / 'data/library.sqlite3') if p.startswith('PMID')}
            row = common.fetch_corpus(common.EXPERIMENTS, query, int(count), excluded_pmids=hidden,
                                      progress=lambda n, total: bar.progress(n / total, text=f'{n}/{total} validated English abstracts'))
            st.session_state.p2_corpus = row['id']
            st.session_state.p2_fetch_message = f"Saved {row['document_count']} abstracts (requested maximum {row['requested']})."
            st.session_state.pop('_p2_corpus', None)
            st.session_state.pop('p2_model_path', None)
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f'Corpus not activated: {exc}. Downloaded batches are kept; retry the same query to resume.')
        finally:
            bar.empty()
    if st.session_state.get('p2_fetch_message'):
        st.success(st.session_state.p2_fetch_message)
    rows = common.list_corpora(common.EXPERIMENTS, include_deleted=True)
    if rows:
        st.dataframe([{'ID': r['id'], 'Query': r['query'], 'Documents': r['document_count'], 'Created': r['created_at'],
                       'Status': 'Archived' if r['deleted_at'] else 'Active'} for r in rows], hide_index=True, width='stretch')
        by_id = {r['id']: r for r in rows}
        cid = st.selectbox('Corpus to manage', list(by_id), format_func=lambda x: f"{by_id[x]['query'][:60]} · {by_id[x]['document_count']} abstracts · {x[:8]}")
        archived = bool(by_id[cid]['deleted_at'])
        st.caption('Soft-delete only: XML, snapshots, statistics and model files are retained. Other corpora are unaffected.')
        if st.button('Restore corpus' if archived else 'Archive corpus'):
            common.set_archived(common.EXPERIMENTS, cid, not archived)
            if st.session_state.p2_corpus == cid and not archived:
                st.session_state.p2_corpus = common.MAIN
                st.session_state.pop('_p2_corpus', None)
            st.rerun()


STEP_LABELS = {'A': 'lowercase', 'B': '+ strip punctuation', 'C': '+ remove stopwords', 'D': '+ Porter stem'}


def dataset_metrics(corpus, result):
    """Dataset requirements in four short numbers."""
    docs = corpus.documents
    cols = st.columns(4)
    cols[0].metric('Abstracts', f'{len(docs):,}')
    cols[1].metric('Unique PMIDs', f'{len({d.pmid for d in docs}):,}')
    cols[2].metric('English, with abstract', f"{sum(bool(d.abstract.strip()) for d in docs):,}")
    cols[3].metric('Tokens / abstract (A)', f"{result['conditions']['A']['stats']['average_tokens_per_document']:.0f}")


def vocabulary_overview(result):
    """Part II numbers per condition, with a bar for the vocabulary size (Part VI at a glance)."""
    stats = {c: v['stats'] for c, v in result['conditions'].items()}
    largest = max(s['vocabulary'] for s in stats.values())
    rows, previous = [], None
    for condition, s in stats.items():
        change = '' if previous is None else f'<small class="p2-change">{s["vocabulary"] / previous - 1:+.0%}</small>'
        bar = f'<span class="p2-track"><span class="p2-bar" style="width:{100 * s["vocabulary"] / largest:.0f}%"></span></span>'
        rows.append([f'<b>{condition}</b>', escape(STEP_LABELS[condition]), f'{s["tokens"]:,}',
                     f'<div class="p2-barcell">{bar}<span>{s["vocabulary"]:,} {change}</span></div>',
                     f'{s["average_tokens_per_document"]:.0f}'])
        previous = s['vocabulary']
    common.html_table(['', 'Condition', 'Tokens', 'Unique terms', 'Tokens / doc'], rows)
    st.caption(f"Each condition adds one step to the previous one. Documents: {stats['A']['documents']:,} in every condition.")


def vocabulary_table(result):
    rows = [{'Condition': c, 'Documents': v['stats']['documents'], 'Total tokens': v['stats']['tokens'],
             'Unique terms': v['stats']['vocabulary'], 'Avg tokens / doc': round(v['stats']['average_tokens_per_document'], 1),
             'Hapax': v['stats']['hapax'], 'Zipf k': round(v['fits']['All']['zipf_exponent'], 3)} for c, v in result['conditions'].items()]
    st.dataframe(rows, hide_index=True, width='stretch')
    common.csv_download('Download vocabulary table', rows, 'vocabulary_by_condition.csv')


def source(corpus):
    st.markdown('**PubMed query (same subject)**')
    st.code(corpus.query, language=None)
    st.caption(f'Fingerprint {corpus.fingerprint}. AbstractText only (no titles, keywords or full text); PubMed returned order; '
               'single-topic, non-random sample. Every record is checked for a unique PMID, English language and a nonempty abstract.')


def browse(corpus):
    docs = corpus.documents
    query = common.text_field('Search papers by title or PMID', 'filter', placeholder='Enter part of a paper title or PMID').strip().casefold()
    matches = [d for d in docs if not query or query in d.title.casefold() or query in d.pmid]
    left, right = st.columns(2)
    with left:
        size = int(common.select_field('Rows per page', 'size', ['25', '50']))
    with right:
        pages = max(1, math.ceil(len(matches) / size))
        page = int(common.select_field('Page', 'page', [str(n) for n in range(1, pages + 1)]))
    start = (page - 1) * size
    st.caption(f'{len(matches):,} matches of {len(docs):,} papers. Showing {start + 1 if matches else 0}-{min(start + size, len(matches))}. Titles open the internal reader.')
    if not matches:
        st.info('No papers match this title or PMID. Clear the search to show the whole corpus.')
        return
    common.html_table(['PMID', 'Title', 'Year'], [[escape(d.pmid), common.link(d.title, paper=d.pmid, source='corpus'), escape(str(d.year or ''))]
                                                  for d in matches[start:start + size]])
    common.csv_download('Download corpus list', [{'PMID': d.pmid, 'Title': d.title, 'Year': d.year} for d in matches], 'corpus.csv')
