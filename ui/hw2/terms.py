"""Page 3 (≈45 s): DF for every term (Part IV), CF vs DF (Part VII), IDF and TF-IDF (Part VIII)."""
import math
from html import escape

import streamlit as st

from src.hw2_answers import part7_answers, part8_answer
from ui.hw2 import common


def render(corpus, result):
    condition = common.condition_picker()
    data = result['conditions'][condition]
    n = len(corpus.documents)
    selected = data['selected_terms']
    demo = common.card(f'CF vs DF vs IDF for {len(selected)} terms (Parts VII, VIII)',
                       f'Condition {condition}. CF counts occurrences, DF counts abstracts, IDF = ln(N / DF) with N = {n:,}, '
                       f'so IDF runs from 0 (in every abstract) to ln N = {math.log(n):.2f} (in one abstract).')
    with demo:
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(sorted(({'term': r['term'], 'CF': r['CF'], 'DF': r['DF'], 'CF / DF': round(r['CF_per_document'], 2),
                                  'IDF': round(r['IDF'], 3)} for r in selected), key=lambda r: r['IDF']),
                         hide_index=True, width='stretch', height=420)
        with right:
            common.chart({'data': {'values': selected}, 'height': 400, 'title': f'CF vs DF of {len(selected)} terms (log-log)',
                          'layer': [{'mark': {'type': 'circle', 'size': 90},
                                     'encoding': {'x': {'field': 'DF', 'type': 'quantitative', 'scale': {'type': 'log'}},
                                                  'y': {'field': 'CF', 'type': 'quantitative', 'scale': {'type': 'log'}},
                                                  'color': {'field': 'IDF', 'type': 'quantitative', 'scale': {'scheme': 'tealblues'}},
                                                  'tooltip': [{'field': x} for x in ['term', 'CF', 'DF', 'IDF']]}},
                                    {'mark': {'type': 'text', 'dx': 8, 'align': 'left', 'fontSize': 10},
                                     'encoding': {'x': {'field': 'DF', 'type': 'quantitative'}, 'y': {'field': 'CF', 'type': 'quantitative'},
                                                  'text': {'field': 'term'}}}]}, key='p2_cf_df')
        st.caption(f'Rows sorted by IDF. Terms in almost every abstract get IDF ≈ 0; a term in one abstract gets ln N = {math.log(n):.2f}. '
                   'Distance above the diagonal = repetition inside the same documents.')
        common.csv_download('Download CF / DF / IDF table', selected, f'{condition}_selected_terms.csv')

    common.more()
    with st.expander('Document frequency of every term (Part IV)'):
        st.caption('DF(t) = abstracts containing t; DF ≤ N and DF ≤ CF.')
        q = st.text_input('Filter terms', key='p2_df_filter').casefold()
        rows = [{'term': r['term'], 'CF': r['CF'], 'DF': r['DF'], 'DF / N': round(r['DF_fraction'], 4), 'CF / DF': round(r['CF_per_document'], 2),
                 'IDF': round(r['IDF'], 4)} for r in data['rows'] if q in r['term']]
        st.caption(f'{len(rows):,} terms · N = {n:,} documents')
        st.dataframe(rows, hide_index=True, width='stretch', height=280)
        common.csv_download('Download CF / DF / IDF for all terms', rows, f'{condition}_cf_df.csv')
    with st.expander('One term across documents (TF, TF × IDF, highlighted reader)', expanded=bool(st.query_params.get('term'))):
        term = common.select_field('Term (type to search the vocabulary)', 'term', [r['term'] for r in data['rows']])
        row = next(r for r in data['rows'] if r['term'] == term)
        for col, field in zip(st.columns(4), ['CF', 'DF', 'DF_fraction', 'IDF']):
            col.metric(field, f'{row[field]:.4f}' if isinstance(row[field], float) else row[field])
        docs = common.documents_for_term(corpus.fingerprint, term, condition, corpus)
        st.caption(f"{len(docs)} documents contain '{term}'. First 50 shown by raw TF. Open a title to verify the condition-aware matches.")
        common.html_table(['PMID', 'Title', 'TF', 'TF × IDF'], [[escape(r['PMID']), common.link(r['Title'], paper=r['PMID'], source='term'), str(r['TF']), f"{r['TF_IDF']:.4f}"]
                                                              for r in docs[:50]])
        common.csv_download('Download all documents for term', docs, 'term_documents.csv')

    common.more('Answers', '')
    common.answers('Part VII · CF vs DF', part7_answers(result, condition))
    common.answers('Part VIII · IDF and Zipf', part8_answer(result, condition))
