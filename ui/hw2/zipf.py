"""Page 2 (≈1 min): rank–frequency vs log-log (Part V), pairwise effect of preprocessing (Part VI), CF Top 50 (Part III)."""
import math
from html import escape

import streamlit as st

from src.hw2_answers import part5_answers
from ui.hw2 import common, domains

METRICS = [('slope', 'Slope'), ('intercept', 'Intercept a'), ('zipf_exponent', 'k = −slope'), ('r_squared', 'R²'), ('rmse_log10', 'RMSE (log10 CF)')]
# Legend names in the brief's own terms (Part V): the data, the linear regression, Zipf's ideal law.
DATA = 'CF of each term (data)'
REFERENCE = "Ideal Zipf's law 1/r (k = 1)"
REGRESSION = 'Linear regression line'


def curve(rows, fit, first_cf, reference=True):
    values = []
    for r in rows:
        # Every series carries the term at that rank: the lines overlap (1/r equals the data at rank 1),
        # so a tooltip on any of them must still name the word.
        values.append(dict(rank=r['rank'], frequency=r['CF'], series=DATA, term=r['term']))
        if reference:
            values.append(dict(rank=r['rank'], frequency=first_cf / r['rank'], series=REFERENCE, term=r['term']))
        if fit['slope'] is not None:
            values.append(dict(rank=r['rank'], frequency=10 ** fit['intercept'] * r['rank'] ** fit['slope'], series=REGRESSION, term=r['term']))
    return values


SERIES_STYLE = {DATA: ('#087f78', [1, 0], 2.6), REGRESSION: ('#e07b1f', [7, 4], 2.0), REFERENCE: ('#8f9aa1', [2, 3], 1.8)}


def powers_of_ten(low, high):
    """Axis ticks 1, 10, 100, … covering [low, high]: one grid line per decade instead of every minor tick."""
    return [10 ** p for p in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)]


def axis(field, title, scale, data_values):
    spec = {'field': field, 'type': 'quantitative', 'title': title, 'scale': {'type': scale}}
    if scale == 'log':
        low, high = min(data_values), max(data_values)
        # Domain hugs the data (no empty decade after the last rank); grid lines only at 1, 10, 100, …
        spec['scale'].update(domain=[low, high], nice=False)
        spec['axis'] = {'values': [t for t in powers_of_ten(low, high) if low <= t <= high], 'format': '~s', 'labelOverlap': True}
    else:
        spec['axis'] = {'format': '~s', 'tickCount': 5}
    return spec


def plot(values, scale, title, key, height=280):
    """Data solid teal, regression dashed orange, ideal Zipf dotted grey; legend stacked so it never gets cut."""
    present = [s for s in SERIES_STYLE if any(v['series'] == s for v in values)]
    common.chart({'data': {'values': values}, 'mark': {'type': 'line', 'clip': True}, 'height': height, 'title': title,
                  'encoding': {'x': axis('rank', 'rank r', scale, [v['rank'] for v in values]),
                               'y': axis('frequency', 'CF', scale, [v['frequency'] for v in values if v['series'] == DATA]),
                               'color': {'field': 'series', 'title': None,
                                         'scale': {'domain': present, 'range': [SERIES_STYLE[s][0] for s in present]},
                                         'legend': {'orient': 'bottom', 'direction': 'vertical', 'labelLimit': 400, 'symbolType': 'stroke'}},
                               'strokeDash': {'field': 'series', 'legend': None,
                                              'scale': {'domain': present, 'range': [SERIES_STYLE[s][1] for s in present]}},
                               'strokeWidth': {'field': 'series', 'legend': None,
                                               'scale': {'domain': present, 'range': [SERIES_STYLE[s][2] for s in present]}},
                               'tooltip': [{'field': 'term', 'title': 'Word at this rank'}, {'field': 'rank', 'title': 'Rank', 'format': ','},
                                           {'field': 'series', 'title': 'Line'},
                                           {'field': 'frequency', 'title': 'CF (actual or predicted)', 'format': ',.1~f'}]}}, key=key)


STEP_ADDED = {'A': 'tokenize + lowercase', 'B': '+ remove punctuation', 'C': '+ remove stopwords', 'D': '+ Porter stemming'}
CONDITION_COLORS = {'A': '#8f9aa1', 'B': '#5688bf', 'C': '#e07b1f', 'D': '#087f78'}


def change(value, previous, digits=0):
    """'19,231' plus a small muted change against the row above (blank for the first row)."""
    shown = f'{value:,.{digits}f}'
    if previous is None:
        return shown
    diff = value - previous
    relative = f' ({diff / previous:+.0%})' if digits == 0 and previous else ''
    return f'{shown}<br><small class="p2-change p2-nowrap">{diff:+,.{digits}f}{relative}</small>'


def effect_summary(conditions):
    """One sentence per step, from the numbers: what each preprocessing step changed most."""
    s = {c: v['stats'] for c, v in conditions.items()}
    k = {c: v['fits']['All']['zipf_exponent'] for c, v in conditions.items()}
    head = {c: common.md_escape(', '.join(r['term'] for r in v['rows'][:3])) for c, v in conditions.items()}
    return (f"**A → B** (punctuation): vocabulary {s['B']['vocabulary'] / s['A']['vocabulary'] - 1:+.0%} — 'data,' and 'data.' merge; tail shortens.  \n"
            f"**B → C** (stopwords): tokens {s['C']['tokens'] / s['B']['tokens'] - 1:+.0%}, vocabulary almost unchanged — "
            f"the head changes from *{head['B']}* to *{head['C']}*.  \n"
            f"**C → D** (stemming): vocabulary {s['D']['vocabulary'] / s['C']['vocabulary'] - 1:+.0%}, tokens unchanged — "
            f"word forms merge, so the tail shortens.  \n"
            f"Zipf k moves {k['A']:.3f} → {k['B']:.3f} → {k['C']:.3f} → {k['D']:.3f}: the exponent depends on how a 'term' is defined.")


def effect_card(result):
    """Part VI in one place: every condition's numbers in one table, and all four curves on one chart."""
    conditions = result['conditions']
    with common.card('Effect of preprocessing (Conditions A–D)',
                     'Each condition adds one step to the previous one; changes are against the row above. '
                     'Spec Part VI: vocabulary size, high-frequency terms, Zipf exponent, shape of the distribution.'):
        rows, previous = [], None
        for c, v in conditions.items():
            s, f = v['stats'], v['fits']['All']
            rows.append([f'<b>{c}</b>', escape(STEP_ADDED[c]),
                         change(s['vocabulary'], previous and previous['vocabulary']),
                         change(s['tokens'], previous and previous['tokens']),
                         escape(', '.join(r['term'] for r in v['rows'][:3])),
                         f"<b>{f['zipf_exponent']:.3f}</b>", f"{f['slope']:.3f}", f"{f['intercept']:.3f}",
                         f"{f['r_squared']:.3f}", f"{f['rmse_log10']:.3f}"])
            previous = s
        common.html_table(['', 'Step added', 'Vocabulary', 'Tokens', 'Top 3 terms', 'Zipf k', 'Slope', 'Intercept', 'R²', 'RMSE'], rows)
        values = [dict(rank=r['rank'], frequency=r['CF'], condition=c, term=r['term'])
                  for c, v in conditions.items() for r in common.thin_ranks(v['rows'])]
        common.chart({'data': {'values': values}, 'mark': {'type': 'line', 'clip': True}, 'height': 300,
                      'title': 'Rank–frequency, log-log, all four conditions',
                      'encoding': {'x': axis('rank', 'rank r', 'log', [v['rank'] for v in values]),
                                   'y': axis('frequency', 'CF', 'log', [v['frequency'] for v in values]),
                                   'color': {'field': 'condition', 'title': 'Condition',
                                             'scale': {'domain': list(CONDITION_COLORS), 'range': list(CONDITION_COLORS.values())},
                                             'legend': {'orient': 'right'}},
                                   'strokeWidth': {'value': 2},
                                   'tooltip': [{'field': 'condition', 'title': 'Condition'}, {'field': 'term', 'title': 'Word at this rank'},
                                               {'field': 'rank', 'title': 'Rank', 'format': ','}, {'field': 'frequency', 'title': 'CF', 'format': ','}]}},
                     key='p2_zipf_conditions')
        st.markdown(effect_summary(conditions))


def render(corpus, result):
    condition = common.condition_picker()
    data = result['conditions'][condition]
    rows = data['rows']
    fit = data['fits']['All']

    with common.card('Experiment 1 vs 2 · does the curve become a straight line in log-log space?',
                     f'Condition {condition}. Same rank–frequency data plotted twice. Linear regression on the log values: '
                     'log10(CF) = a − b·log10(r); Zipf exponent k = b.'):
        for col, (key, label) in zip(st.columns(5), [('zipf_exponent', 'Zipf exponent k'), ('r_squared', 'R²'), ('rmse_log10', 'RMSE (log10)'),
                                                      ('slope', 'Slope'), ('intercept', 'Intercept a')]):
            col.metric(label, f'{fit[key]:.3f}')
        left, right = st.columns(2)
        with left:
            plot([v for v in curve(rows[:200], fit, rows[0]['CF']) if v['series'] != REGRESSION], 'linear',
                 'Experiment 1 · linear scale (curved), ranks 1–200', 'p2_zipf_linear')
        with right:
            plot(curve(common.thin_ranks(rows), fit, rows[0]['CF']), 'log', 'Experiment 2 · log-log scale (straight)', 'p2_zipf_full')
        top = sum(r['CF'] for r in rows[:max(1, len(rows) // 100)]) / data['stats']['tokens']
        st.caption(f"The top 1% of terms hold {top:.1%} of all tokens; {data['stats']['hapax']:,} terms ({data['stats']['hapax'] / len(rows):.1%}) occur once.")

    effect_card(result)

    common.more()
    with st.expander('Regression by rank region, with residuals'):
        region = st.selectbox('Rank region', list(data['fits']), help='Head = top 10% of ranks, middle = next 70%, tail = last 20%.')
        region_fit = data['fits'][region]
        if region_fit['slope'] is None:
            note = region_fit['note']
        elif region_fit['r_squared'] is None:
            note = 'All frequencies are equal here, so R² is undefined — not evidence of a perfect Zipf fit.'
        else:
            note = 'Linear regression on the log values: it describes the trend; it is not a statistical test of Zipf\'s law.'
        st.caption(f"Ranks {region_fit['rank_start']}–{region_fit['rank_end']} · {region_fit['points']:,} points. {note}")
        for col, (key, label) in zip(st.columns(5), METRICS):
            col.metric(label, 'N/A' if region_fit[key] is None else f'{region_fit[key]:.4f}')
        # Fit numbers above use every rank in the region; the charts draw a thinned subset.
        subset = common.thin_ranks(rows[(region_fit['rank_start'] or 1) - 1:region_fit['rank_end']])
        left, right = st.columns(2)
        with left:
            plot(curve(subset, region_fit, rows[0]['CF']), 'log', f'Log-log, {region}', 'p2_zipf_loglog', 240)
        with right:
            if region_fit['slope'] is not None:
                residuals = [dict(rank=r['rank'], residual=math.log10(r['CF']) - region_fit['intercept'] - region_fit['slope'] * math.log10(r['rank'])) for r in subset]
                common.chart({'data': {'values': residuals}, 'mark': {'type': 'point', 'size': 8}, 'height': 240, 'title': 'Residual (log10 CF)',
                              'encoding': {'x': {'field': 'rank', 'type': 'quantitative', 'scale': {'type': 'log'}},
                                           'y': {'field': 'residual', 'type': 'quantitative'}}}, key='p2_zipf_residuals')
        st.dataframe([dict(region=name, **f) for name, f in data['fits'].items()], hide_index=True, width='stretch')
        common.csv_download('Download regional fits', [dict(region=r, **f) for r, f in data['fits'].items()], f'{condition}_fits.csv')
    with st.expander('Collection frequency · Top 50 terms (Part III)'):
        names = list(result['conditions'])
        top_condition = st.radio('Condition', names, index=names.index(condition), horizontal=True, key='p2_top50_condition',
                                 format_func=lambda c: f'{c} · {STEP_ADDED[c]}')
        top_rows = result['conditions'][top_condition]['rows']
        q = st.text_input('Filter vocabulary').casefold()
        filtered = [r for r in top_rows if q in r['term']]
        st.caption(f'Condition {top_condition}. CF(t) = occurrences in the whole collection; ties alphabetical. '
                   f'{len(filtered):,} matching terms of {len(top_rows):,}; first 50 displayed.')
        st.dataframe([{k: r[k] for k in ('rank', 'term', 'CF', 'DF', 'relative_frequency')} for r in filtered[:50]],
                     hide_index=True, width='stretch', height=300)
        common.csv_download(f'Download full frequency table ({top_condition})', top_rows, f'{top_condition}_frequency.csv')
    with st.expander('Frequency spectrum and published figures'):
        common.chart({'data': {'values': data['spectrum']}, 'mark': 'point', 'height': 240, 'title': 'How many terms occur exactly k times',
                      'encoding': {'x': {'field': 'frequency', 'type': 'quantitative', 'scale': {'type': 'log'}, 'title': 'CF = k'},
                                   'y': {'field': 'terms', 'type': 'quantitative', 'scale': {'type': 'log'}, 'title': 'number of terms'}}}, key='p2_zipf_spectrum')
        common.csv_download('Download frequency spectrum', data['spectrum'], f'{condition}_spectrum.csv')
        current, _ = common.report_status(result)
        if current:
            for name in ['rank_frequency', 'loglog_fit', 'residuals']:
                path = common.selected_output() / 'figures' / f'{name}.pdf'
                if path.exists():
                    st.download_button('Download figure: ' + name.replace('_', ' '), path.read_bytes(), path.name, 'application/pdf')
    with st.expander('Optional challenge · two domains'):
        domain_answer = domains.render()

    common.more('Answers', '')
    items = part5_answers(result)
    if domain_answer:
        items['Optional · Does domain specificity affect the Zipf distribution?'] = domain_answer
    common.answers('Part V · Required analysis (condition B)', items)
