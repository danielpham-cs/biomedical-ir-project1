"""Page 5 (≈45 s): search with spelling correction, match locations, and the DP edit distance behind it (Brief)."""
import re
from collections import Counter
from html import escape

import streamlit as st

from src.matching import edit_distance_table
from src.search import search, spelling_candidates, suggest_query_correction
from ui.hw2 import common

LOCATION_CHIPS = 12
# Short real typos, one per edit operation; every correct word is in the GLP-1 vocabulary.
DP_EXAMPLES = {'insert': ('livr', 'liver'), 'substitute': ('doze', 'dose'), 'transpose': ('weigth', 'weight')}


def prefix_label(word, n):
    """Header of row/column n: the whole prefix word[:n], its newest letter in bold (ε for the empty prefix)."""
    if n == 0:
        return 'ε'
    return f'{escape(word[:n - 1])}<b>{escape(word[n - 1])}</b>'


def dp_table(left, right, table, steps):
    """D[i][j] compares the prefixes left[:i] and right[:j], so the headers show those prefixes."""
    path = {s['cell']: s['operation'] for s in steps}
    head = '<tr><th></th>' + ''.join(f'<th>{prefix_label(right, j)}</th>' for j in range(len(right) + 1)) + '</tr>'
    body = ''
    for i, row in enumerate(table):
        cells = ''
        for j, value in enumerate(row):
            op = path.get((i, j))
            css = '' if op is None else (' class="path"' if op == 'match' else ' class="edit"')
            tip = f'{left[:i] or "ε"} → {right[:j] or "ε"}: {value} edit{"s" if value != 1 else ""}'
            cells += f'<td{css} title="{escape(tip, quote=True)}">{value}</td>'
        body += f'<tr><th>{prefix_label(left, i)}</th>{cells}</tr>'
    return f'<div class="p2-dp"><table>{head}{body}</table></div>'


def interpret(query, vocabulary):
    """Per keyword: found in the vocabulary, corrected to the closest word, or unknown."""
    rows = []
    for word in re.findall(r'[A-Za-z0-9][A-Za-z0-9-]*', query):
        folded = word.casefold()
        if folded in vocabulary or len(word) < 4 or word.isupper() or not word.isalpha():
            rows.append({'original': word, 'corrected': None, 'distance': 0, 'known': folded in vocabulary})
            continue
        candidates = spelling_candidates(word, vocabulary, 1)
        best = candidates[0] if candidates else None
        rows.append({'original': word, 'corrected': best and best['candidate'], 'distance': best['edit_distance'] if best else None, 'known': False})
    return rows


def keyword_line(rows):
    """One sentence: corrected words struck through with their replacement, known words ticked."""
    parts = []
    for r in rows:
        if r['corrected']:
            edits = f"{r['distance']} edit{'s' if r['distance'] != 1 else ''}"
            parts.append(f'<s>{escape(r["original"])}</s> → <b class="p2-fix">{escape(r["corrected"])}</b> <small>({edits})</small>')
        elif r['known']:
            parts.append(f'<b>{escape(r["original"])}</b> <span class="p2-ok">✓</span>')
        else:
            parts.append(f'{escape(r["original"])} <small>(kept as typed)</small>')
    st.markdown(f'<div class="p2-keywords">Keywords: {" &nbsp;·&nbsp; ".join(parts)}</div>', unsafe_allow_html=True)


def positions_line(hit):
    """Character offsets grouped by matched word: 'semaglutide 181 · 445 · … | receptor 24 · 672'."""
    groups = {}
    for o in hit.occurrences:
        label = o.word.casefold() + (f' (d={o.distance})' if o.distance else '')
        groups.setdefault(label, []).append(o.start)
    parts = []
    for label, starts in groups.items():
        shown = ' · '.join(str(s) for s in starts[:LOCATION_CHIPS])
        more = f' +{len(starts) - LOCATION_CHIPS}' if len(starts) > LOCATION_CHIPS else ''
        parts.append(f'<b>{escape(label)}</b> {shown}{more}')
    return ' &nbsp;|&nbsp; '.join(parts)


def result_card(hit, doc, query, method, mode, edits):
    """Title (opens the paper in a popup), one muted metadata line, a highlighted snippet, grouped positions."""
    count = len(hit.occurrences)
    summary = ', '.join(f'{w} ×{n}' for w, n in Counter(o.word.casefold() for o in hit.occurrences).most_common())
    with st.container(key=f'p2_hit_{hit.pmid}'):
        st.button(common.md_escape(hit.title), key=f'p2_open_{hit.pmid}', type='tertiary', help='Open this paper in a popup',
                  on_click=common.open_popup, args=(hit.pmid,))
        st.markdown(
            f'<div class="p2-hit">'
            f'<div class="p2-hit-meta">PMID {escape(hit.pmid)} · BM25 {hit.score:.3f} · '
            f'<span class="p2-hit-count">{count} match{"es" if count != 1 else ""}</span> ({escape(summary)})</div>'
            + (f'<div class="p2-hit-snippet">{hit.snippet_html}</div>' if hit.snippet_html else '')
            + f'<div class="p2-hit-pos">Character positions: {positions_line(hit)}</div></div>',
            unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, max_entries=32)
def cached_search(fingerprint, query, method, mode, edits, _index, _documents):
    """Ranking all 1,000 abstracts takes ~0.6–0.8 s; reruns (opening a popup, paging) reuse the result."""
    return search(query, _index, _documents, method=method, match_mode=mode, max_edits=edits, top_k=len(_documents))


@st.cache_data(show_spinner=False, max_entries=64)
def cached_spelling(fingerprint, query, _index, _vocabulary):
    return interpret(query, _vocabulary), suggest_query_correction(query, _index, _vocabulary)


def render(corpus, result):
    index, vocabulary = common.retrieval_index(corpus.fingerprint, corpus.documents)

    with common.card('Query matching with spelling correction',
                     'Unknown keywords are matched to the closest corpus word by DP edit distance. Every match is located (word @ character offset) in each abstract.'):
        query = common.text_field('Query keywords', 'query', placeholder='e.g. semaglutde receptor')
        a, b, c = st.columns(3)
        with a:
            mode = common.select_field('Matching', 'mode', ['STEM', 'PARTIAL', 'FUZZY'],
                                       help='STEM = exact after Porter stemming · PARTIAL = substring inside a token · FUZZY = whole token within edit distance')
        with b:
            method = common.select_field('Query logic', 'method', ['AND', 'OR', 'PHRASE'] if mode == 'STEM' else ['AND', 'OR'])
        with c:
            edits = int(common.select_field('Maximum edit distance (fuzzy)', 'edits', ['1', '2'])) if mode == 'FUZZY' else 1
        if not query.strip():
            st.info('Enter a query to search the frozen corpus.')
        else:
            keywords, suggestion = cached_spelling(corpus.fingerprint, query, index, vocabulary)
            keyword_line(keywords)
            effective = query
            if suggestion:
                left, right = st.columns([3, 1])
                auto = left.toggle('Auto-correct keywords', value=True, key='p2_autocorrect')
                if auto:
                    effective = suggestion
                    left.write(f'Showing results for **{suggestion}** (typed: {query})')
                else:
                    left.write(f'Did you mean: **{suggestion}**')
                if right.button('Apply spelling suggestion'):
                    st.session_state.p2_query = suggestion
                    st.session_state.pop('_p2_query', None)
                    st.rerun()
            hits = cached_search(corpus.fingerprint, effective, method, mode, edits, index, corpus.documents)
            st.caption(f'{len(hits)} matching documents; first 5 displayed; all in the CSV. Click a title to open the paper in a popup '
                       'with every match highlighted.')
            if not hits:
                st.info('No matches. Apply the suggestion, or try OR, Partial or Fuzzy.')
            by_pmid = {d.pmid: d for d in corpus.documents}
            for hit in hits[:5]:
                result_card(hit, by_pmid[hit.pmid], effective, method, mode, edits)
            if st.session_state.get('p2_popup') in by_pmid:
                common.paper_dialog(by_pmid[st.session_state.p2_popup], effective, method, mode, edits)
            if hits:
                common.csv_download('Download search results', [{'PMID': h.pmid, 'Title': h.title, 'BM25': h.score, 'Matches': len(h.occurrences)} for h in hits], 'search_results.csv')

    with common.card('How the correction is computed · edit distance by dynamic programming',
                     'Each cell D[i][j] is the fewest edits that turn the first i letters of the typed word into the first j letters '
                     'of the vocabulary word; every edit costs 1.'):
        # LaTeX, not markdown: underscores in a_i / b_j would otherwise be read as italics.
        st.latex(r'D[i][j]=\min\begin{cases}D[i-1][j]+1 & \text{delete}\\ D[i][j-1]+1 & \text{insert}\\ '
                 r'D[i-1][j-1]+[a_i\neq b_j] & \text{substitute (0 if the letters match)}\\ D[i-2][j-2]+1 & \text{transpose two adjacent letters}\end{cases}')
        example = st.radio('Example', list(DP_EXAMPLES), horizontal=True, key='p2_ed_example',
                           format_func=lambda name: f'{DP_EXAMPLES[name][0]} → {DP_EXAMPLES[name][1]} ({name})')
        typed, target = DP_EXAMPLES[example]
        a, b = st.columns(2)
        typed = a.text_input('Typed word', typed, key=f'p2_ed_left_{typed}').strip().casefold()
        target = b.text_input('Vocabulary word', target, key=f'p2_ed_right_{target}').strip().casefold()
        if typed and target and len(typed) <= 30 and len(target) <= 30:
            table, steps = edit_distance_table(typed, target)
            st.markdown(dp_table(typed, target, table, steps), unsafe_allow_html=True)
            edits_done = [s for s in steps if s['operation'] != 'match']
            st.metric('Edit distance', table[-1][-1])
            i, j = len(typed), max(len(target) - 1, 0)
            st.caption(f'Read a cell as "row prefix → column prefix": row **{common.md_escape(typed[:i] or "ε")}**, column '
                       f'**{common.md_escape(target[:j] or "ε")}** holds {table[i][j]}, the edits that turn one into the other. '
                       'The bottom-right cell compares the whole words. Green cells: matched characters on one optimal path. '
                       'Orange cells: an edit on that path. '
                       + ('; '.join(f"{s['operation']} {s['from'] or 'ε'} → {s['to'] or 'ε'}" for s in edits_done) or 'identical words'))
        else:
            st.info('Enter two words of at most 30 characters.')

    common.more()
    with st.expander('Spelling candidates for one word (ranked by distance, then corpus frequency)'):
        # 'livr' is one edit from both 'liver' (233×) and 'live' (7×): the tie is broken by corpus frequency.
        word = st.text_input('Misspelled word', 'livr', key='p2_spell_word').strip()
        if word:
            candidates = spelling_candidates(word, vocabulary)
            if word.casefold() in vocabulary:
                st.success(f"'{word}' is already in the corpus vocabulary ({vocabulary[word.casefold()]:,} occurrences); no correction is suggested.")
            elif candidates:
                st.write(f"Best correction: **{candidates[0]['candidate']}** (distance {candidates[0]['edit_distance']})")
            else:
                st.warning('No vocabulary word within the allowed distance (1 for ≤ 4 letters, otherwise 2).')
            if candidates:
                st.dataframe(candidates, hide_index=True, width='stretch')
