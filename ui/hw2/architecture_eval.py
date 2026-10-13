"""Skip-gram vs CBOW: evaluation of the ten part 2 models (scripts/run_project2_part2.py)."""
import json
from statistics import mean

import numpy as np
import pandas as pd
import streamlit as st

from src import embedding_study as study
from ui.hw2 import common

PART2 = common.ROOT / 'output/project2_part2_embedding'
MODELS = ('Skip-gram', 'CBOW')
ARCH = {'Skip-gram': 'skipgram', 'CBOW': 'cbow'}
COLORS = {'Skip-gram': '#087f78', 'CBOW': '#e07b1f'}
# The last three pairs of study.PAIRS are unrelated controls (nausea–kidney, mice–semaglutide, cancer–weight).
CONTROLS = 3
BANDS = ['2-4', '5-9', '10-49', '50-199', '>=200']
EXAMPLE = 'glp-1'
EXAMPLE_NOTE = (
    'glp-1 is the most frequent word of the corpus (1,606 times), and the two models agree on none of its top-5 neighbours. '
    'Skip-gram returns the words glp-1 appears **with**: "ras"/"ra" from "GLP-1 RAs" (receptor agonists), '
    '"glucose-dependent", "polypeptide/glucagon-like". CBOW returns words that can **replace** glp-1 in the same sentence slot: '
    'other spellings ("glp1", "glp)-1") and combinations ("glp-1/gip", "gip)/glp-1"). '
    "Relative to random word pairs, Skip-gram's top 5 stand out clearly (z ≈ 2.2–2.7), while CBOW's own top 5 reach only "
    "z ≈ 0.6–1.1, because every CBOW cosine is high. Skip-gram also ranks CBOW's spelling variants above random (z ≈ 1.4–2.1). "
    "In CBOW, Skip-gram's \"ras\" and \"ra\" fall far below random (z ≈ −2). "
    'Both lists contain rare noise ("cornerstones", "croatia", seen 2 times). Changing only the seed keeps 8 of 10 '
    'Skip-gram neighbours and 7 of 10 CBOW neighbours: glp-1 is frequent, so both models are stable for it.')


def main_models():
    """The Skip-gram and CBOW models trained with identical settings (dim 100, window 5, 15 epochs, seed 42)."""
    return [PART2 / 'models' / f'{ARCH[m]}_main.npz' for m in MODELS]


@st.cache_data(show_spinner=False)
def experiments(stamp):
    return json.loads((PART2 / 'experiments.json').read_text(encoding='utf8'))


SETTINGS = {'main': 'Main (100 dimensions, window 5)', 'window2': 'Window 2', 'dim50': '50 dimensions'}
# Same group colours as the report figures (src/project2_part2_figures.py).
GROUP_COLORS = dict(zip(study.GROUPS, ['#087f78', '#c07828', '#7654a3', '#c64f4f', '#2f6db5', '#5b8c2a', '#b5487f', '#8a6d3b', '#4a5560']))
OTHER = 'Other frequent words'
MAP_NOTE = '''
**What the map shows.** Each dot is one word. t-SNE squeezes the 100-dimensional vectors into 2 dimensions so they can be drawn.
Coloured, labelled dots are the words of nine groups chosen from the GLP-1 domain before training; grey dots are the 300 most frequent words.
If the model learned meaning, words of the same colour should sit together.

**How t-SNE works.** It looks at which words are close to each word in the full vectors and tries to keep those words close in 2-D.
That is why it shows clear clusters, while the PCA map above (a straight, linear projection that keeps only the two directions with the most
spread) shows mostly one cloud.

**Be careful when reading it:**
- the distance between two clusters and the size of a cluster mean nothing;
- the picture changes with its settings (perplexity) and with the random start;
- it can only rearrange what the vectors already contain.

**Purity** puts a number on the picture: for each coloured word, the share of its 3 nearest coloured words that are in the same group.
Random placement would give about {chance:.0%}. t-SNE keeps about the same purity as the full vectors; PCA loses most of it.

**A cluster in a picture is not proof of meaning.** Words can also cluster because they come from the same abstracts or have similar frequency.
That is why every claim on this page is checked with cosine similarity in the full vectors, not on the map.

**Settings.** With *Window 2* the model only sees the next word or two, so it groups words that play the same grammar role; most topic groups
become weaker, especially for CBOW. *50 dimensions* gives a map very close to the main model: with this little data, 50 dimensions are enough.
'''


def projection_chart(points, x, y, title, legend, key):
    rows = [{'word': p['word'], 'count': p['count'], 'group': p['group'] or OTHER, 'x': p[x], 'y': p[y], 'labelled': p['group'] is not None}
            for p in points]
    rows.sort(key=lambda r: r['labelled'])  # grey points first, so the coloured ones are drawn on top
    color = {'field': 'group', 'type': 'nominal', 'title': None,
             'scale': {'domain': list(GROUP_COLORS) + [OTHER], 'range': list(GROUP_COLORS.values()) + ['#cfd8dc']},
             'legend': {'orient': 'right', 'symbolSize': 60} if legend else None}
    axis = {'title': None, 'labels': False, 'ticks': False, 'grid': False, 'domain': False}
    common.chart({'data': {'values': rows}, 'height': 520, 'title': title,
                  'encoding': {'x': {'field': 'x', 'type': 'quantitative', 'axis': axis, 'scale': {'zero': False}},
                               'y': {'field': 'y', 'type': 'quantitative', 'axis': axis, 'scale': {'zero': False}}},
                  'layer': [{'mark': {'type': 'circle', 'opacity': 0.9},
                             'params': [{'name': f'{key}_zoom', 'select': 'interval', 'bind': 'scales'}],
                             'encoding': {'color': color,
                                          'size': {'condition': {'test': 'datum.labelled', 'value': 70}, 'value': 18},
                                          'tooltip': [{'field': 'word', 'title': 'Word'}, {'field': 'group', 'title': 'Group'},
                                                      {'field': 'count', 'title': 'Corpus frequency'}]}},
                            {'transform': [{'filter': 'datum.labelled'}],
                             'mark': {'type': 'text', 'align': 'left', 'dx': 6, 'dy': -5, 'fontSize': 10},
                             'encoding': {'text': {'field': 'word'}, 'color': color}}]},
                 key=key)


def projection_maps(d):
    """Labelled t-SNE map of either model, read from experiments.json (no t-SNE is run in the page)."""
    a, b = st.columns(2)
    model = a.radio('Model', MODELS, horizontal=True, key='p2_map_model')
    setting = b.radio('Setting', list(SETTINGS), horizontal=True, format_func=SETTINGS.get, key='p2_map_setting')
    projection = d['projections'][f'{ARCH[model]}_{setting}']
    purity = projection['purity']
    projection_chart(projection['points'], 'tsne_x', 'tsne_y',
                     f"{model} · {SETTINGS[setting].lower()} · t-SNE (perplexity {projection['perplexity']:.0f})", True, 'p2_map_tsne')
    st.caption(f"Purity (share of a word's 3 nearest coloured words in its own group): t-SNE {purity['tsne_2d']:.2f}, full vectors "
               f"{purity['cosine']:.2f}, PCA {purity['pca_2d']:.2f}, random {d['chance_purity']:.2f}. "
               'Hover a dot for the word; scroll to zoom into a dense cluster, drag to move, double-click to reset.')
    with st.expander('How to read the t-SNE map'):
        st.markdown(MAP_NOTE.replace('{chance:.0%}', f"{d['chance_purity']:.0%}"))


def load(name):
    path = PART2 / 'models' / f'{name}.npz'
    return common.saved_model(str(path), common.file_stamp(path)) if path.exists() else None


def cosines(model, word):
    unit = model.vectors / np.maximum(np.linalg.norm(model.vectors, axis=1, keepdims=True), 1e-12)
    return unit @ unit[model.words.index(word)]


def worked_example(d, word):
    """Top-10 neighbours of one word in both models, their seed stability, and each model's score for the other's neighbours."""
    models = {m: load(f'{ARCH[m]}_main') for m in MODELS}
    seeds = {m: load(f'{ARCH[m]}_seed7') for m in MODELS}
    if any(models[m] is None or word not in models[m].words for m in MODELS):
        st.caption(f"'{word}' is not in both part 2 models; try {EXAMPLE}.")
        return
    count = int(models['Skip-gram'].counts[models['Skip-gram'].words.index(word)])
    st.markdown(f'**Worked example: {common.md_escape(word)}** (seen {count:,} times)')
    st.caption('The word explored above, in both models side by side. Same corpus and settings (100 dimensions, window 5, 15 epochs, seed 42); only the architecture differs, so every difference comes from Skip-gram vs CBOW.')
    top = {m: models[m].most_similar(word, 10) for m in MODELS}
    tops = {m: [r['word'] for r in top[m]] for m in MODELS}
    columns = st.columns(2)
    for column, m in zip(columns, MODELS):
        other = MODELS[1 - MODELS.index(m)]
        retrained = None
        if seeds[m] is not None and word in seeds[m].words:
            retrained = [r['word'] for r in seeds[m].most_similar(word, 10)]
        rows = []
        for r in top[m]:
            row = {'word': r['word'], 'cosine': round(r['cosine_similarity'], 3),
                   'corpus count': int(models[m].counts[models[m].words.index(r['word'])])}
            if retrained is not None:
                row['seed 7'] = '✓ kept' if r['word'] in retrained else '✗ dropped'
            row[f'also in {other} top 10'] = '✓' if r['word'] in tops[other] else ''
            rows.append(row)
        with column:
            if retrained is None:
                st.markdown(f'**{m}** · top 10')
                st.dataframe(rows, hide_index=True, width='stretch')
                continue
            kept = sum(row['seed 7'] == '✓ kept' for row in rows)
            st.markdown(f'**{m}** · top 10 · {kept}/10 kept when retrained with seed 7')
            # Highlight the neighbours that disappear when only the random seed changes.
            frame = pd.DataFrame(rows)
            st.dataframe(frame.style.apply(lambda row: ['background-color:#fbe8d9;color:#5a2a0c' if row['seed 7'] == '✗ dropped' else ''] * len(row),
                                           axis=1).format({'cosine': '{:.3f}'}),
                         hide_index=True, width='stretch')
            dropped = [w for w in tops[m] if w not in retrained]
            new = [w for w in retrained if w not in tops[m]]
            if dropped:
                st.caption(f"Highlighted = dropped with seed 7: {', '.join(dropped)}. Replaced by: {', '.join(new)}.")
    shared = len(set(tops['Skip-gram']) & set(tops['CBOW']))
    st.caption(f'Shared top-10 neighbours: {shared} of 10.')
    # Score each model's top-5 in both spaces; z = (cosine − mean random-pair cosine) / SD, so the two spaces are comparable.
    sims = {m: cosines(models[m], word) for m in MODELS}
    words = list(dict.fromkeys(tops['Skip-gram'][:5] + tops['CBOW'][:5]))
    rows = []
    for w in words:
        row = {'word': w, 'top 5 of': ' + '.join(m for m in MODELS if w in tops[m][:5])}
        for m in MODELS:
            if w in models[m].words:
                c = float(sims[m][models[m].words.index(w)])
                row[f'{m} cosine'] = round(c, 3)
                row[f'{m} z vs random'] = round((c - d['baseline'][m]['mean']) / d['baseline'][m]['std'], 1)
        rows.append(row)
    if word == EXAMPLE:
        with st.expander('What the glp-1 example shows: Skip-gram finds words used with glp-1, CBOW finds words that replace it'):
            st.markdown(EXAMPLE_NOTE)
    with st.expander("Extra: how each model scores the other model's neighbours"):
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('z = (cosine − mean cosine of random word pairs) ÷ their SD in that model; '
                   'CBOW cosines are high everywhere, so compare z, not raw cosine.')


def scorecard(d):
    """One row per criterion: value for each model, and which one is better (None when not comparable)."""
    training = {r['model']: r for r in d['training'] if r['experiment'] == 'main'}
    rows = []

    def row(criterion, values, fmt, better, note):
        winner = None
        if better:
            pick = max if better == 'higher' else min
            winner = pick(MODELS, key=lambda m: values[m])
        rows.append({'Criterion': criterion, **{m: fmt(values[m]) for m in MODELS},
                     'Better': winner or '—', 'How to read it': note})

    row('Training examples / epoch', {m: training[m]['examples_per_epoch'] for m in MODELS}, '{:,}'.format, None,
        'Skip-gram: one example per (center, context) pair; CBOW: one per position.')
    row('Training time (s)', {m: training[m]['seconds'] for m in MODELS}, '{:.0f}'.format, 'lower',
        'Same machine, all ten models trained concurrently.')
    row('Probe loss, epoch 0 → 15', {m: (training[m]['loss_start'], training[m]['loss_end']) for m in MODELS},
        lambda v: f'{v[0]:.3f} → {v[1]:.3f}', None,
        'Different prediction tasks, so the two losses are not comparable.')
    row('Mean cosine of random word pairs', {m: d['baseline'][m]['mean'] for m in MODELS}, '{:.2f}'.format, 'lower',
        'Baseline: a lower value means a high cosine is more informative.')
    margins = {m: mean(r['margin'] for r in d['group_cosines'][f'{m} / main']) for m in MODELS}
    row('Semantic groups: within − between cosine', margins, '{:.3f}'.format, None,
        'Raw margin; inflated by CBOW\'s higher overall cosines.')
    row('Semantic groups: margin ÷ random-pair SD', {m: margins[m] / d['baseline'][m]['std'] for m in MODELS},
        '{:.2f}'.format, 'higher', 'Margin rescaled by each space\'s own spread of cosines: the fair comparison.')
    pairs, controls = d['pairs'][:-CONTROLS], d['pairs'][-CONTROLS:]
    row(f'Related pairs: mean percentile ({len(pairs)} pairs)',
        {m: mean(p[f'{ARCH[m]}_percentile'] for p in pairs) for m in MODELS}, '{:.1f}'.format, 'higher',
        'Percentile of each pair\'s cosine among random pairs; 50 = chance.')
    row(f'Unrelated control pairs: mean percentile ({CONTROLS} pairs)',
        {m: mean(p[f'{ARCH[m]}_percentile'] for p in controls) for m in MODELS}, '{:.1f}'.format, 'lower',
        'Should stay near or below 50.')
    seed = {m: {r['frequency_band']: r['mean_overlap'] for r in d['overlap'][f'{m} | seed 42 vs seed 7 (same setting)']}
            for m in MODELS}
    row('Seed stability, rare words (seen 2–4×)', {m: seed[m]['2-4'] for m in MODELS}, '{:.0%}'.format, 'higher',
        'Share of top-10 neighbours kept when only the random seed changes.')
    row('Seed stability, frequent words (≥ 200×)', {m: seed[m]['>=200'] for m in MODELS}, '{:.0%}'.format, 'higher',
        'Same measure for the most frequent words.')
    purity = {m: d['projections'][f'{ARCH[m]}_main']['purity'] for m in MODELS}
    for key, label in (('cosine', 'full vectors'), ('pca_2d', 'PCA 2-D'), ('tsne_2d', 't-SNE 2-D')):
        row(f'Group purity of nearest neighbours, {label}', {m: purity[m][key] for m in MODELS}, '{:.0%}'.format, 'higher',
            f"Chance = {d['chance_purity']:.0%}.")
    return rows, seed


def loss_chart(d):
    values = [{'model': r['model'], 'epoch': epoch, 'probe_loss': round(loss, 4)}
              for r in d['training'] if r['experiment'] == 'main' for epoch, loss in enumerate(r['history'])]
    common.chart({'data': {'values': values}, 'height': 230, 'title': 'Probe loss per epoch (main settings)',
                  'mark': {'type': 'line', 'point': {'size': 50}, 'strokeWidth': 2},
                  'encoding': {'x': {'field': 'epoch', 'type': 'quantitative', 'title': 'Epoch'},
                               'y': {'field': 'probe_loss', 'type': 'quantitative', 'title': 'Probe loss', 'scale': {'zero': False}},
                               'color': {'field': 'model', 'type': 'nominal', 'title': None,
                                         'scale': {'domain': list(MODELS), 'range': list(COLORS.values())},
                                         'legend': {'orient': 'top'}},
                               'tooltip': [{'field': 'model'}, {'field': 'epoch'}, {'field': 'probe_loss', 'title': 'loss'}]}},
                 key='p2_arch_loss')


def stability_chart(seed):
    values = [{'model': m, 'band': band, 'overlap': round(seed[m][band], 3)} for m in MODELS for band in BANDS]
    common.chart({'data': {'values': values}, 'height': 230,
                  'title': 'Neighbours kept when only the seed changes',
                  'mark': {'type': 'bar', 'cornerRadiusEnd': 4, 'width': {'band': 0.85}},
                  'encoding': {'x': {'field': 'band', 'type': 'ordinal', 'sort': BANDS, 'title': 'Times the word occurs in the corpus',
                                     'axis': {'labelAngle': 0}},
                               'xOffset': {'field': 'model', 'sort': list(MODELS)},
                               'y': {'field': 'overlap', 'type': 'quantitative', 'title': 'Neighbours kept',
                                     'axis': {'format': '%'}, 'scale': {'domain': [0, 1]}},
                               'color': {'field': 'model', 'type': 'nominal', 'title': None,
                                         'scale': {'domain': list(MODELS), 'range': list(COLORS.values())},
                                         'legend': {'orient': 'top'}},
                               'tooltip': [{'field': 'model'}, {'field': 'band', 'title': 'frequency'},
                                           {'field': 'overlap', 'title': 'kept', 'format': '.0%'}]}},
                 key='p2_arch_stability')


def render_maps():
    path = PART2 / 'experiments.json'
    if not path.exists():
        st.info('Run scripts/run_project2_part2.py to compute the PCA and t-SNE maps.')
        return
    projection_maps(experiments(common.file_stamp(path)))


def render(word=EXAMPLE):
    path = PART2 / 'experiments.json'
    if not path.exists():
        st.info('Run scripts/run_project2_part2.py to train both architectures and measure them.')
        return
    d = experiments(common.file_stamp(path))
    rows, seed = scorecard(d)
    training = {r['model']: r for r in d['training'] if r['experiment'] == 'main'}
    sg, cb = training['Skip-gram'], training['CBOW']
    cols = st.columns(4)
    cols[0].metric('Examples / epoch', f"{sg['examples_per_epoch'] / cb['examples_per_epoch']:.1f}×",
                   help='Skip-gram ÷ CBOW, same window (±5).')
    cols[1].metric('Training time', f"{sg['seconds'] / cb['seconds']:.1f}×", help='Skip-gram ÷ CBOW.')
    cols[2].metric('Rare-word stability', f"{seed['Skip-gram']['2-4']:.0%} vs {seed['CBOW']['2-4']:.0%}",
                   help='Skip-gram vs CBOW: top-10 neighbours kept across seeds, words seen 2–4 times.')
    shared = {r['frequency_band']: r['mean_overlap'] for r in d['overlap']['Skip-gram vs CBOW | main']}['all']
    cols[3].metric('Neighbours shared', f'{shared:.0%}', help='Top-10 neighbours the two models have in common, whole vocabulary.')
    left, right = st.columns(2)
    with left:
        loss_chart(d)
        st.caption('Both losses fall, but they score different tasks (averaged window → word vs word → each context word); '
                   'a lower loss is not a better embedding.')
    with right:
        stability_chart(seed)
        st.caption('Skip-gram is equally stable at every frequency. CBOW is stable only for frequent words: '
                   'a rare word\'s vector is mostly its random start.')
    worked_example(d, word or EXAMPLE)
    wins = {m: sum(r['Better'] == m for r in rows) for m in MODELS}
    with st.expander(f"Scorecard: Skip-gram better on {wins['Skip-gram']} of {len(rows)} criteria, CBOW on {wins['CBOW']}"):
        st.caption('Same corpus, dimensions 100, window 5, 15 epochs, seed 42; only the architecture differs. '
                   '"—" means the two values cannot be ranked.')
        st.dataframe(rows, hide_index=True, width='stretch')
    with st.expander('Verdict: Skip-gram is the better choice for this corpus'):
        st.markdown(
            f"Skip-gram is better on {wins['Skip-gram']} of the measured criteria and CBOW on {wins['CBOW']}. "
            f"CBOW is about {sg['seconds'] / cb['seconds']:.1f}× faster and gives tight, high-cosine clusters of interchangeable "
            'frequent words (drug names, adverse events). But all its cosines are high (random pairs '
            f"{d['baseline']['CBOW']['mean']:.2f} vs {d['baseline']['Skip-gram']['mean']:.2f}), so once that is accounted for its groups "
            'separate less, and its rare-word neighbours change with the seed. Skip-gram costs '
            f"{sg['examples_per_epoch'] / cb['examples_per_epoch']:.0f}× more updates but gives topical neighbours that are as stable for "
            'rare terms as for frequent ones, which is what a small domain corpus with many rare scientific terms needs. '
            'For this corpus Skip-gram is the better choice; CBOW is a reasonable fast choice when only frequent words matter.')
    with st.expander('Nearest neighbours of the query words and rare words, side by side'):
        neighbours = {}
        for r in d['neighbors'] + d['rare_neighbors']:
            neighbours.setdefault((r['query'], r['query_count']), {}).setdefault(r['model'], []).append(
                f"{r['word']} {r['cosine_similarity']:.2f}")
        agree = {r['query']: r['shared_top5'] for r in d['model_agreement']}
        st.dataframe([{'Query (count)': f'{q} ({n})', **{m: ', '.join(v.get(m, [])) for m in MODELS}, 'Shared of 5': agree.get(q)}
                      for (q, n), v in neighbours.items()], hide_index=True, width='stretch')
    with st.expander('Word pairs and semantic groups'):
        st.dataframe([{'Pair': f'{a} – {b}', 'Kind': 'control' if i >= len(study.PAIRS) - CONTROLS else 'related',
                       'Counts': p['counts'],
                       'Skip-gram cosine': round(p['skipgram_cosine'], 3), 'Skip-gram pct': round(p['skipgram_percentile'], 1),
                       'CBOW cosine': round(p['cbow_cosine'], 3), 'CBOW pct': round(p['cbow_percentile'], 1)}
                      for i, ((a, b), p) in enumerate(zip(study.PAIRS, d['pairs']))], hide_index=True, width='stretch')
        groups = {m: {r['group']: r for r in d['group_cosines'][f'{m} / main']} for m in MODELS}
        st.dataframe([{'Group': g, 'Words': groups['Skip-gram'][g]['words'],
                       **{f'{m} within': round(groups[m][g]['within_mean_cosine'], 3) for m in MODELS},
                       **{f'{m} between': round(groups[m][g]['between_mean_cosine'], 3) for m in MODELS},
                       **{f'{m} margin ÷ SD': round(groups[m][g]['margin'] / d['baseline'][m]['std'], 2) for m in MODELS}}
                      for g in groups['Skip-gram']], hide_index=True, width='stretch')
    common.csv_download('Download scorecard', rows, 'cbow_vs_skipgram_scorecard.csv', key='p2_arch_csv')
    report = PART2 / 'report.pdf'
    if report.exists():
        st.download_button('Download the full part 2 report (PDF)', report.read_bytes(), 'project2_part2_report.pdf',
                           'application/pdf', key='p2_arch_pdf')
