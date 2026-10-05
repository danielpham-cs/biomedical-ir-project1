"""Porter sections: own stemmer step by step and the before/after comparison (Brief · Porter)."""
from html import escape

import streamlit as st

from src.porter import porter_stem, porter_steps
from src.zipf_experiment import token_trace
from ui.hw2 import common

EXAMPLES = ['generalizations', 'receptors', 'treatments', 'relational', 'hopping', 'glp-1']


@st.cache_data(show_spinner=False, max_entries=4)
def reference_agreement(fingerprint, words):
    """Compare our stems with NLTK's ORIGINAL_ALGORITHM mode over the whole vocabulary."""
    from nltk.stem import PorterStemmer
    reference = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
    checked = [w for w in words if w.isascii() and w.isalpha()]
    different = [{'word': w, 'ours': porter_stem(w), 'nltk_original': reference.stem(w)} for w in checked if porter_stem(w) != reference.stem(w)]
    return len(checked), different


def stem_chain(rows):
    parts = [f'<span>{escape(rows[0]["before"])}</span>'] if rows else []
    for row in rows:
        if row['changed']:
            parts.append(f'<i>—{escape(row["step"])}→</i><span class="on">{escape(row["after"])}</span>')
    return '<div class="p2-steps">' + ''.join(parts) + '</div>'


def word_steps(full=False):
    """Demo: the rules that fire for one word (all eight steps in the details view)."""
    left, right = st.columns([2, 3])
    picked = left.selectbox('Example word', EXAMPLES, key=f'p2_porter_example_{full}')
    word = right.text_input('Or type a word', picked, key=f'p2_porter_word_{picked}_{full}').strip()
    rows = porter_steps(word)
    if not rows:
        st.info(f"'{word}' is not an ASCII alphabetic word, so it is kept unchanged (identifiers such as GLP-1 or 24.9% keep their form).")
        return
    st.markdown(stem_chain(rows), unsafe_allow_html=True)
    shown = rows if full else [r for r in rows if r['changed']]
    st.dataframe([{'Step': r['step'], 'Purpose': r['title'], 'Input': r['before'], 'Output': r['after'], 'Rule': r['rule'],
                   'm(output)': r['m'], '[C](VC)^m[V]': r['form']} for r in shown], hide_index=True, width='stretch')
    st.caption(f'Result: {word.lower()} → {porter_stem(word)}')


def agreement(corpus, result):
    checked, different = reference_agreement(corpus.fingerprint, tuple(r['term'] for r in result['conditions']['C']['rows']))
    st.metric('Agreement with NLTK ORIGINAL Porter', f'{checked - len(different):,} / {checked:,} words', f'{len(different)} differences')
    if different:
        st.dataframe(different[:50], hide_index=True, width='stretch')


def before_after(result):
    c, d = (result['conditions'][k] for k in 'CD')
    cols = st.columns(3)
    cols[0].metric('Vocabulary C → D', f"{d['stats']['vocabulary']:,}", f"{d['stats']['vocabulary'] - c['stats']['vocabulary']:,}")
    cols[1].metric('Hapax C → D', f"{d['stats']['hapax']:,}", f"{d['stats']['hapax'] - c['stats']['hapax']:,}")
    cols[2].metric('Zipf k C → D', f"{d['fits']['All']['zipf_exponent']:.3f}", f"{d['fits']['All']['zipf_exponent'] - c['fits']['All']['zipf_exponent']:+.3f}")
    st.dataframe([{'Rank': i + 1, 'Before (C)': f"{c['rows'][i]['term']} ({c['rows'][i]['CF']:,})",
                   'After (D)': f"{d['rows'][i]['term']} ({d['rows'][i]['CF']:,})"} for i in range(min(15, len(c['rows']), len(d['rows'])))],
                 hide_index=True, width='stretch')


def merged_groups(result):
    group_query = st.text_input('Find a Porter group', placeholder='patient, receptor, treat...').casefold()
    groups = [r for r in result['merged_groups'] if group_query in r['forms'] or group_query in r['stem']]
    st.dataframe(groups[:100], hide_index=True, width='stretch')
    common.csv_download('Download all matching Porter groups', groups, 'porter_groups.csv')


def abstract_stems(corpus):
    doc = st.selectbox('Abstract', corpus.documents, format_func=lambda d: f'{d.pmid} | {d.title}', key='p2_porter_doc')
    trace = [r for r in token_trace(doc.abstract) if r['C']]
    st.dataframe([{'Word (C)': r['C'], 'Stem (D)': r['D'], 'Changed': r['C'] != r['D']} for r in trace], hide_index=True, width='stretch', height=240)
    st.caption(f"{sum(r['C'] != r['D'] for r in trace)} of {len(trace)} tokens change in this abstract.")
