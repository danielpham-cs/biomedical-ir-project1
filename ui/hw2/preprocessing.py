"""Preprocessing sections: steps 1–5 (Spec · Part I) and the effect of conditions A–D (Spec · Part VI)."""
from html import escape

import streamlit as st

from src.zipf_experiment import CONDITIONS, token_trace, tokens
from ui.hw2 import common

STEPS = [('1', 'Tokenization', 'Split on whitespace; every non-space run is one token.'),
         ('2', 'Case folding', 'Data, data, DATA → data.'),
         ('3', 'Punctuation', 'Strip boundary punctuation ("data," "data." → data); keep internal marks such as GLP-1, 0.5, 24%.'),
         ('4', 'Stopwords', 'Two versions: kept (A, B) and removed (C, D), NLTK English list.'),
         ('5', 'Stemming', 'Own Porter 1980 implementation on ASCII alphabetic words.')]
TRACE_COLUMNS = {'original': '1 · token', 'A': '2 · case folded', 'B': '3 · punctuation', 'C': '4 · stopwords removed',
                 'D': '5 · Porter stem', 'action': 'what changed'}
SAMPLE = 'The GLP-1 receptors were activated; DATA, data. Data! Treatments helped patients.'


def trace_table(raw, changed_only=False):
    trace = token_trace(raw)
    rows = [{TRACE_COLUMNS[k]: r[k] for k in TRACE_COLUMNS} for r in trace if not changed_only or r['action'] != 'unchanged']
    st.dataframe(rows, hide_index=True, width='stretch')
    return trace


def sandbox():
    """Demo: one short sentence through steps 1–5; A–D are prefixes of this pipeline."""
    raw = st.text_input('Sentence', SAMPLE, key='p2_sandbox_text')
    trace_table(raw)
    st.caption('A = steps 1–2 · B = 1–3 · C = 1–4 · D = 1–5 (cumulative). Sandbox text never changes corpus counts.')


def definitions():
    common.html_table(['Step', 'Operation', 'Rule in this system'], [[s, escape(t), escape(d)] for s, t, d in STEPS])
    common.html_table(['Condition', 'Steps applied', 'Definition'],
                      [[c, s, escape(CONDITIONS[c])] for c, s in zip(CONDITIONS, ['1–2', '1–3', '1–4', '1–5'])])


def abstract_trace(corpus):
    doc = st.selectbox('Choose an abstract', corpus.documents, format_func=lambda d: f'{d.pmid} | {d.title}', key='p2_preprocess_doc')
    changed_only = st.checkbox('Show only tokens that change', value=True)
    trace = trace_table(doc.abstract, changed_only)
    common.csv_download('Download token trace', trace, 'token_trace.csv')
    condition = common.condition_picker('Resulting token stream for condition')
    words = tokens(doc.abstract, condition)
    st.write(f'Condition {condition}: {len(words)} tokens / {len(set(words))} unique')
    st.code(' | '.join(words), language=None, wrap_lines=True)


def condition_effect(result):
    data = result['conditions']
    st.markdown('**Top 10 terms per condition**')
    st.dataframe([{'Rank': i + 1} | {f'{c} (CF)': f"{v['rows'][i]['term']} ({v['rows'][i]['CF']:,})" for c, v in data.items()}
                  for i in range(min(10, *(len(v['rows']) for v in data.values())))], hide_index=True, width='stretch')
    common.chart({'data': {'values': [dict(rank=r['rank'], CF=r['CF'], condition=c) for c, v in data.items() for r in common.thin_ranks(v['rows'])]},
                  'mark': 'line', 'height': 300, 'title': 'Rank–frequency, log-log, conditions A–D',
                  'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': 'log'}},
                               'y': {'field': 'CF', 'type': 'quantitative', 'scale': {'type': 'log'}},
                               'color': {'field': 'condition'}}}, key='p2_conditions_curve')
