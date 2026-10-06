"""Optional challenge (tab on the Zipf page): does domain specificity affect the Zipf distribution?"""
import streamlit as st

from src.domain_compare import compare_domains, domain_answer
from src.experiment_store import ARXIV_CATEGORIES
from src.zipf_experiment import CONDITIONS
from ui.hw2 import common

ARXIV_NAME = 'Computer science (arXiv)'


@st.cache_data(show_spinner=False, max_entries=8)
def comparison(signature_a, signature_b, id_a, id_b, condition, size):
    corpus_a = common.corpus_by_id(signature_a, id_a)
    corpus_b = common.corpus_by_id(signature_b, id_b)
    return compare_domains(corpus_a, corpus_b, condition=condition, size=size)


def domain_b():
    """The second domain: the arXiv computer-science corpus, else the newest other experiment corpus (None if none)."""
    rows = common.list_corpora(common.EXPERIMENTS)
    row = next((r for r in rows if r['query'].startswith('arXiv computer science')), rows[0] if rows else None)
    if row is None:
        return None
    if row['query'].startswith('arXiv computer science'):
        return row['id'], ARXIV_NAME, f"arXiv · {len(ARXIV_CATEGORIES)} CS subfields ({', '.join(ARXIV_CATEGORIES)})"
    return row['id'], row['query'][:30], 'PubMed'


def fetch_default():
    st.info('A second domain is needed. Fetch 500 computer-science abstracts from arXiv: the 50 newest papers whose primary '
            f"category is each of {', '.join(ARXIV_CATEGORIES)}.")
    if st.button('Fetch domain B (500 arXiv abstracts)'):
        bar = st.progress(0, text='Querying arXiv…')
        try:
            common.fetch_arxiv_corpus(common.EXPERIMENTS, progress=lambda n, total: bar.progress(n / total, text=f'{n}/{total} abstracts'))
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f'Fetch failed: {exc}. Retry in a minute (arXiv limits request rate).')
        finally:
            bar.empty()


def render():
    """Compare two corpora; return the data-grounded answer (None until a second domain exists)."""
    second = domain_b()
    if second is None:
        fetch_default()
        return None
    (id_a, label_a, source_a), (id_b, label_b, source_b) = (common.MAIN, 'GLP-1 (medical)', 'PubMed · GLP-1 query'), second
    try:
        sig_a, sig_b = common.signature(id_a), common.signature(id_b)
        size = min(500, len(common.corpus_by_id(sig_a, id_a).documents), len(common.corpus_by_id(sig_b, id_b).documents))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load a domain corpus: {exc}')
        return None
    common.section(1, 'Two domains', f'Same number of abstracts ({size}) and the same preprocessing, so only the domain differs.')
    a, b = st.columns(2)
    a.markdown(f'**A · {label_a}**  \n{source_a} · {size} abstracts')
    b.markdown(f'**B · {label_b}**  \n{source_b} · {size} abstracts')
    condition = a.selectbox('Preprocessing condition', list(CONDITIONS), index=2, format_func=lambda c: f'{c} - {CONDITIONS[c]}', key='p2_domain_condition')
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
    return answer
