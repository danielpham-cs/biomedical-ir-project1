"""Figures for Project 2 part 2 (word embeddings), drawn from experiments.json."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .embedding_study import GROUPS

PALETTE = ['#087f78', '#c07828', '#7654a3', '#c64f4f', '#2f6db5', '#5b8c2a', '#b5487f', '#8a6d3b', '#4a5560']
GROUP_COLORS = dict(zip(GROUPS, PALETTE))
MODEL_COLORS = {'Skip-gram': '#087f78', 'CBOW': '#c07828'}


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
    return plt


def _scatter(ax, projection, axis, title):
    points = projection['points']
    x, y = f'{axis}_x', f'{axis}_y'
    others = [p for p in points if not p['group']]
    ax.scatter([p[x] for p in others], [p[y] for p in others], s=6, color='#cfd8dc', lw=0, label='Top-300 frequent words')
    for group, color in GROUP_COLORS.items():
        chosen = [p for p in points if p['group'] == group]
        ax.scatter([p[x] for p in chosen], [p[y] for p in chosen], s=22, color=color, label=group, edgecolor='white', lw=.4)
        for p in chosen:
            ax.annotate(p['word'], (p[x], p[y]), xytext=(3, 2), textcoords='offset points', fontsize=6.3, color=color)
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#cfd8dc')


def plot_all(results, directory: Path):
    plt = _setup()
    directory.mkdir(parents=True, exist_ok=True)
    training = {(r['experiment'], r['model']): r for r in results['training']}
    projections = results['projections']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, name in zip(axes, ('Skip-gram', 'CBOW')):
        for experiment, style in (('main', '-'), ('window2', '--'), ('dim50', ':')):
            history = training[(experiment, name)]['history']
            ax.plot(range(len(history)), history, style, marker='o', ms=2.5, color=MODEL_COLORS[name], label=experiment)
        ax.set(title=f'{name}: fixed training-probe loss', xlabel='Epoch (0 = initialization)', ylabel='Negative-sampling loss')
        ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.savefig(directory/'loss_curves.png', dpi=180); plt.close(fig)

    for axis, label in (('pca', 'PCA'), ('tsne', 't-SNE')):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
        for ax, arch, name in zip(axes, ('skipgram', 'cbow'), ('Skip-gram', 'CBOW')):
            p = projections[f'{arch}_main']
            extra = (f"{sum(p['pca_variance']):.1%} variance" if axis == 'pca' else f"perplexity {p['perplexity']:.0f}, KL {p['tsne_kl']:.2f}")
            _scatter(ax, p, axis, f"{name} · {label} ({extra}) · group purity {p['purity'][axis + '_2d']:.2f}")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center', ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(.5, -.07))
        fig.savefig(directory/f'{axis}_main.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
    for row, (arch, name) in enumerate((('skipgram', 'Skip-gram'), ('cbow', 'CBOW'))):
        for col, (experiment, caption) in enumerate((('window2', 'window 2'), ('main', 'window 5 (main)'), ('dim50', '50 dimensions'))):
            p = projections[f'{arch}_{experiment}']
            _scatter(axes[row, col], p, 'tsne', f"{name} · {caption} · purity {p['purity']['tsne_2d']:.2f}")
    fig.savefig(directory/'tsne_parameters.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for ax, key in zip(axes, ('skipgram_main_p5', 'skipgram_main', 'skipgram_main_p50')):
        p = projections[key]
        _scatter(ax, p, 'tsne', f"Skip-gram · perplexity {p['perplexity']:.0f} · purity {p['purity']['tsne_2d']:.2f}")
    fig.savefig(directory/'tsne_perplexity.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    bands = [r['frequency_band'] for r in results['overlap']['Skip-gram | seed 42 vs seed 7 (same setting)'] if r['frequency_band'] != 'all']
    width = .38
    for offset, name in ((-width / 2, 'Skip-gram'), (width / 2, 'CBOW')):
        rows = [r for r in results['overlap'][f'{name} | seed 42 vs seed 7 (same setting)'] if r['frequency_band'] != 'all']
        values = [r['mean_overlap'] for r in rows]
        bars = ax.bar(np.arange(len(bands)) + offset, values, width, color=MODEL_COLORS[name], label=name)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], fontsize=7, padding=2)
    ax.set_xticks(range(len(bands)), [f'{b}\n({r["words"]:,} words)' for b, r in zip(bands, rows)])
    ax.set(xlabel='Corpus frequency of the word', ylabel='Mean top-10 neighbor overlap', ylim=(0, 1),
           title='Neighbor stability when only the random seed changes')
    ax.legend(); ax.grid(axis='y', alpha=.15)
    fig.savefig(directory/'seed_stability.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    names = [r['group'] for r in results['group_cosines']['Skip-gram / main']]
    for offset, model in ((-width / 2, 'Skip-gram'), (width / 2, 'CBOW')):
        rows = {r['group']: r for r in results['group_cosines'][f'{model} / main']}
        ax.bar(np.arange(len(names)) + offset, [rows[n]['within_mean_cosine'] for n in names], width, color=MODEL_COLORS[model], label=f'{model}: within group')
        ax.scatter(np.arange(len(names)) + offset, [rows[n]['between_mean_cosine'] for n in names], color='#263238', marker='_', s=180, zorder=3,
                   label='between groups' if model == 'CBOW' else None)
    for model in ('Skip-gram', 'CBOW'):
        ax.axhline(results['baseline'][model]['mean'], color=MODEL_COLORS[model], ls=':', lw=1)
    ax.set_xticks(range(len(names)), names, rotation=20, ha='right')
    ax.set(ylabel='Mean cosine similarity', ylim=(0, 1.05), title='Within-group vs between-group cosine (dotted: random-pair mean)')
    ax.legend(fontsize=7, ncol=3, loc='upper center'); ax.grid(axis='y', alpha=.15)
    fig.savefig(directory/'group_cosines.png', dpi=180); plt.close(fig)
