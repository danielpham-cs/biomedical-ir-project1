"""Optional challenge (tab on the Zipf page): does domain specificity affect the Zipf distribution?"""
import streamlit as st

from src.domain_compare import compare_domains, domain_answer
from src.library import hidden_article_ids
from src.zipf_experiment import CONDITIONS
from ui.hw2 import common

DEFAULT_B = ('("deep learning"[Title] OR "machine learning"[Title] OR "neural network"[Title]) '
             'AND (algorithm[Title/Abstract] OR architecture[Title/Abstract])')


@st.cache_data(show_spinner=False, max_entries=8)
def comparison(signature_a, signature_b, id_a, id_b, condition, size):
    corpus_a = common.corpus_by_id(signature_a, id_a)
    corpus_b = common.corpus_by_id(signature_b, id_b)
    return compare_domains(corpus_a, corpus_b, condition=condition, size=size)


def short_name(corpus_id, names):
    if corpus_id == common.MAIN:
        return 'GLP-1 (medical)'
    rows = {r['id']: r['query'] for r in common.list_corpora(common.EXPERIMENTS)}
    return 'Machine learning (computing)' if rows.get(corpus_id) == DEFAULT_B else names[corpus_id].split(' · ')[0][:30]


def fetch_default():
    st.info('A second domain is needed. Fetch 500 computing-method abstracts from PubMed (GLP-1 papers are excluded), '
            'or fetch any other query on the Data page (Details → Manage corpora).')
    st.code(DEFAULT_B, language=None)
    if st.button('Fetch domain B (500 abstracts)'):
        bar = st.progress(0, text='Searching PubMed…')
        try:
            main = common.corpus_by_id(common.signature(common.MAIN), common.MAIN)
            hidden = {p[4:] for p in hidden_article_ids(common.ROOT / 'data/library.sqlite3') if p.startswith('PMID')}
            common.fetch_corpus(common.EXPERIMENTS, DEFAULT_B, 500, excluded_pmids=hidden | {d.pmid for d in main.documents},
                                progress=lambda n, total: bar.progress(n / total, text=f'{n}/{total} abstracts'))
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f'Fetch failed: {exc}. Downloaded batches are kept; retry to resume.')
        finally:
            bar.empty()


def render():
    """Compare two corpora; return the data-grounded answer (None until a second domain exists)."""
    names = common.corpus_names()
    if len(names) < 2:
        fetch_default()
        return None
    ids = list(names)
    common.section(1, 'Choose two domains', 'Both domains are cut to the same number of abstracts so that vocabulary and token counts are comparable.')
    a, b = st.columns(2)
    id_a = a.selectbox('Domain A', ids, index=0, format_func=names.get, key='p2_domain_a')
    others = [i for i in ids if i != id_a]
    id_b = b.selectbox('Domain B', others, index=0, format_func=names.get, key='p2_domain_b')
    a, b = st.columns(2)
    label_a = a.text_input('Short name A', short_name(id_a, names), key=f'p2_domain_label_{id_a}_a') or 'A'
    label_b = b.text_input('Short name B', short_name(id_b, names), key=f'p2_domain_label_{id_b}_b') or 'B'
    if label_a == label_b:
        label_b += ' (B)'
    try:
        sig_a, sig_b = common.signature(id_a), common.signature(id_b)
        sizes = [len(common.corpus_by_id(sig_a, id_a).documents), len(common.corpus_by_id(sig_b, id_b).documents)]
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load a domain corpus: {exc}')
        return None
    a, b = st.columns(2)
    size = int(a.number_input('Abstracts per domain', min(10, min(sizes)), min(sizes), min(500, min(sizes)), step=10, key='p2_domain_size'))
    condition = b.selectbox('Preprocessing condition', list(CONDITIONS), index=2, format_func=lambda c: f'{c} - {CONDITIONS[c]}', key='p2_domain_condition')
    with st.spinner('Comparing domains…'):
        result = comparison(sig_a, sig_b, id_a, id_b, condition, size)
    pa, pb = result['A'], result['B']

    common.section(2, 'Measures required by the spec')
    measures = [('Documents', 'documents', '{:,}'), ('Tokens', 'tokens', '{:,}'), ('Vocabulary', 'vocabulary', '{:,}'),
                ('Zipf exponent k', 'zipf_exponent', '{:.4f}'), ('R²', 'r_squared', '{:.4f}'), ('RMSE (log10 CF)', 'rmse_log10', '{:.4f}'),
                ('Average tokens / document', 'average_tokens_per_document', '{:.1f}'), ('Hapax share', 'hapax_share', '{:.1%}')]
    rows = [{'Measure': label, label_a: fmt.format(pa[key]), label_b: fmt.format(pb[key])} for label, key, fmt in measures]
    rows.append({'Measure': 'Top 10 terms', label_a: ', '.join(pa['top_terms']), label_b: ', '.join(pb['top_terms'])})
    st.dataframe(rows, hide_index=True, width='stretch')
    common.csv_download('Download domain comparison', rows, 'domain_comparison.csv')

    common.section(3, 'Rank–frequency curves', 'Relative frequency CF / tokens, so the two curves share one scale.')
    values = [dict(rank=r['rank'], relative=r['CF'] / p['tokens'], domain=label, term=r['term'])
              for label, p in ((label_a, pa), (label_b, pb)) for r in common.thin_ranks(p['rows'])]
    common.chart({'data': {'values': values}, 'mark': 'line', 'height': 320,
                  'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': 'log'}},
                               'y': {'field': 'relative', 'type': 'quantitative', 'scale': {'type': 'log'}, 'title': 'CF / tokens'},
                               'color': {'field': 'domain', 'title': None},
                               'tooltip': [{'field': x} for x in ['domain', 'rank', 'term', 'relative']]}}, key='p2_domains_curve')
    st.dataframe([{'Region': region, f'k {label_a}': round(pa['regions'][region]['zipf_exponent'] or 0, 4),
                   f'k {label_b}': round(pb['regions'][region]['zipf_exponent'] or 0, 4)} for region in pa['regions']],
                 hide_index=True, width='stretch')

    common.section(4, 'Vocabulary overlap')
    cols = st.columns(3)
    cols[0].metric('Shared terms', f"{result['shared_vocabulary']:,}")
    cols[1].metric('Jaccard overlap', f"{result['jaccard']:.1%}")
    cols[2].metric('Shared top-10 terms', len(result['shared_top10']))
    a, b = st.columns(2)
    a.markdown(f'**Frequent in {label_a}, absent from {label_b}**')
    a.write(', '.join(result['distinctive_A']) or '—')
    b.markdown(f'**Frequent in {label_b}, absent from {label_a}**')
    b.write(', '.join(result['distinctive_B']) or '—')

    answer = domain_answer(result, (label_a, label_b))
    st.info('The answer to "Does domain specificity affect the Zipf distribution?" is in the Answers tab.')
    return answer
