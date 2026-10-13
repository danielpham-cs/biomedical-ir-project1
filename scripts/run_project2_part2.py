"""Project 2 part 2: train CBOW and Skip-gram variants, measure the embedding space, export the report."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')  # one BLAS thread per training process

import numpy as np
from src.tables import rows_csv
from src.pubmed import _write_atomic
from src.zipf_experiment import load_snapshot, embedding_sentences, embedding_preprocessing, tokens
from src.word2vec import Word2VecModel, train_skipgram, model_key
from src import embedding_study as study
from src.embedding_view import pca_2d
from src.tsne import tsne_2d
from src.eos import split_sentences

DEFAULT_SNAPSHOT = ROOT/'output/project2_glp1/corpus_snapshot.json'


def save_json(path, obj):
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def save_csv(path, rows):
    _write_atomic(path, rows_csv(rows).encode('utf-8-sig'))


def train_one(snapshot, output, experiment, architecture, epochs):
    """Train (or reuse) one model; runs in a worker process."""
    corpus = load_snapshot(snapshot)
    config = replace(study.EXPERIMENTS[experiment], architecture=architecture, epochs=epochs)
    prep = embedding_preprocessing('C')
    key = model_key(config, corpus.fingerprint, prep)
    path = Path(output)/'models'/f'{architecture}_{experiment}.npz'
    if path.exists():
        try:
            if Word2VecModel.load(path).metadata.get('model_key') == key:
                return str(path), 'reused'
        except (OSError, ValueError, KeyError):
            pass
    model = train_skipgram(embedding_sentences(corpus), config, corpus_metadata=corpus.metadata(), preprocessing=prep)
    model.save(path)
    return str(path), f"trained in {model.metadata['elapsed_seconds']:.0f}s"


def train_all(snapshot, output, epochs, workers):
    jobs = [(e, a) for e in study.EXPERIMENTS for a in study.ARCHITECTURES]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {job: pool.submit(train_one, str(snapshot), str(output), *job, epochs) for job in jobs}
        models = {}
        for job, future in futures.items():
            path, status = future.result()
            print(f'{study.LABELS[job[1]]} / {job[0]}: {status}', flush=True)
            models[job] = Word2VecModel.load(Path(path))
    return models


def neighbor_rows(models, words):
    rows = []
    for word in words:
        for arch in study.ARCHITECTURES:
            model = models[('main', arch)]
            if word not in model.words:
                rows.append({'query': word, 'query_count': 0, 'model': study.LABELS[arch], 'rank': None,
                             'word': '(not in vocabulary)', 'cosine_similarity': None, 'corpus_count': None})
                continue
            count = int(model.counts[model.words.index(word)])
            for rank, row in enumerate(model.most_similar(word, 5), 1):
                rows.append({'query': word, 'query_count': count, 'model': study.LABELS[arch], 'rank': rank, **row})
    return rows


def project(model, ids, labels, perplexity=30.0):
    """PCA (fit on the whole vocabulary) and t-SNE (fit on the subset) for one model."""
    pca, variance = pca_2d(model.vectors)
    tsne, kl = tsne_2d(model.vectors[ids], perplexity=perplexity)
    lookup = dict(labels)
    points = [{'word': model.words[i], 'count': int(model.counts[i]), 'group': lookup.get(model.words[i]),
               'pca_x': float(pca[i, 0]), 'pca_y': float(pca[i, 1]),
               'tsne_x': float(tsne[n, 0]), 'tsne_y': float(tsne[n, 1])} for n, i in enumerate(ids)]
    labeled = [n for n, i in enumerate(ids) if model.words[i] in lookup]
    names = [lookup[model.words[ids[n]]] for n in labeled]
    purity = {'cosine': study.group_purity(model.vectors[[ids[n] for n in labeled]], names, cosine_space=True),
              'pca_2d': study.group_purity(pca[[ids[n] for n in labeled]], names),
              'tsne_2d': study.group_purity(tsne[labeled], names)}
    return {'points': points, 'pca_variance': variance.tolist(), 'tsne_kl': kl, 'perplexity': perplexity, 'purity': purity}


def analyze(models, corpus):
    results = {'experiments': {name: asdict(cfg) for name, cfg in study.EXPERIMENTS.items()}, 'training': []}
    for (experiment, arch), model in models.items():
        meta = model.metadata
        results['training'].append({'experiment': experiment, 'model': study.LABELS[arch], 'architecture': arch,
            'dimensions': meta['config']['dimensions'], 'window': meta['config']['window'],
            'min_count': meta['config']['min_count'], 'seed': meta['config']['seed'],
            'vocabulary': meta['vocabulary_size'], 'training_examples': meta['training_pairs'],
            'examples_per_epoch': meta['training_pairs'] // meta['config']['epochs'],
            'seconds': meta['elapsed_seconds'], 'loss_start': meta['history'][0]['probe_loss'],
            'loss_end': meta['history'][-1]['probe_loss'], 'history': [h['probe_loss'] for h in meta['history']]})
    main = {arch: models[('main', arch)] for arch in study.ARCHITECTURES}
    results['corpus'] = {k: main['skipgram'].metadata[k] for k in ('vocabulary_size', 'raw_tokens', 'retained_tokens', 'sentence_count')}
    results['corpus'].update(document_count=main['skipgram'].metadata['corpus']['document_count'],
                             fingerprint=main['skipgram'].metadata['corpus']['fingerprint'])
    counts = dict(zip(main['skipgram'].words, main['skipgram'].counts.tolist()))
    results['top_words'] = [{'word': w, 'count': c} for w, c in list(counts.items())[:30]]

    results['neighbors'] = neighbor_rows(models, study.QUERIES)
    results['rare_neighbors'] = neighbor_rows(models, study.RARE_QUERIES)
    agreement = []
    for word in study.QUERIES + study.RARE_QUERIES:
        if all(word in m.words for m in main.values()):
            a, b = ({r['word'] for r in m.most_similar(word, 5)} for m in main.values())
            agreement.append({'query': word, 'count': counts[word], 'shared_top5': len(a & b)})
    results['model_agreement'] = agreement

    baselines = {arch: study.random_pair_baseline(m) for arch, m in main.items()}
    results['baseline'] = {study.LABELS[a]: {'mean': float(b.mean()), 'std': float(b.std()),
                           'p95': float(np.percentile(b, 95))} for a, b in baselines.items()}
    pairs = []
    for a, b in study.PAIRS:
        row = {'pair': f'{a} – {b}', 'counts': f"{counts.get(a, 0)} / {counts.get(b, 0)}"}
        for arch, model in main.items():
            value = study.cosine(model, a, b)
            row[f'{arch}_cosine'] = value
            row[f'{arch}_percentile'] = study.percentile(value, baselines[arch])
        pairs.append(row)
    results['pairs'] = pairs
    results['group_cosines'] = {f'{study.LABELS[arch]} / {e}': study.group_cosines(models[(e, arch)])
                                for e in ('main', 'window2', 'dim50', 'mincount5') for arch in study.ARCHITECTURES}

    print('Neighbor stability across seeds and settings...', flush=True)
    comparisons = {'seed7': 'seed 42 vs seed 7 (same setting)', 'window2': 'window 5 vs window 2',
                   'dim50': '100 vs 50 dimensions', 'mincount5': 'min_count 2 vs 5'}
    results['overlap'] = {f'{study.LABELS[arch]} | {label}': study.neighbor_overlap(models[('main', arch)], models[(e, arch)])
                          for e, label in comparisons.items() for arch in study.ARCHITECTURES}
    results['overlap']['Skip-gram vs CBOW | main'] = study.neighbor_overlap(main['skipgram'], main['cbow'])

    documents = [set(tokens(d.abstract, 'C')) for d in corpus.documents]
    results['corpus'].update(words_before_stopwords=sum(len(tokens(d.abstract, 'B')) for d in corpus.documents),
                             distinct_words=len(set().union(*documents)))
    sentences = [(d.pmid, s, words) for d in corpus.documents for paragraph in d.abstract.split('\n\n')
                 for s in split_sentences(paragraph) if (words := tokens(s, 'C'))]
    results['samples'] = study.training_samples(sentences)
    results['cooccurrence'] = {study.LABELS[a]: study.cooccurrence_share(m, documents) for a, m in main.items()}

    print('PCA and t-SNE projections...', flush=True)
    labels = study.labeled_words(main['skipgram'])
    results['chance_purity'] = study.chance_purity([g for _, g in labels])
    results['projections'] = {}
    for e in ('main', 'window2', 'dim50'):
        for arch in study.ARCHITECTURES:
            model = models[(e, arch)]
            results['projections'][f'{arch}_{e}'] = project(model, study.projection_subset(model), labels)
    for perplexity in (5.0, 50.0):
        model = main['skipgram']
        results['projections'][f'skipgram_main_p{int(perplexity)}'] = project(model, study.projection_subset(model), labels, perplexity)
    return results


def run(snapshot, output, epochs, workers, report=True):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    (output/'tables').mkdir(exist_ok=True)
    models = train_all(snapshot, output, epochs, workers)
    corpus = load_snapshot(snapshot)
    results = analyze(models, corpus)
    save_json(output/'experiments.json', results)
    save_csv(output/'tables'/'training_summary.csv', [{k: v for k, v in r.items() if k != 'history'} for r in results['training']])
    save_csv(output/'tables'/'neighbors_top5.csv', results['neighbors'])
    save_csv(output/'tables'/'rare_word_neighbors.csv', results['rare_neighbors'])
    save_csv(output/'tables'/'pair_cosines.csv', results['pairs'])
    save_csv(output/'tables'/'group_cosines.csv', [dict(model=k, **r) for k, rows in results['group_cosines'].items() for r in rows])
    save_csv(output/'tables'/'neighbor_cooccurrence.csv', [dict(model=k, **r) for k, rows in results['cooccurrence'].items() for r in rows])
    save_csv(output/'tables'/'neighbor_overlap.csv', [dict(comparison=k, **r) for k, rows in results['overlap'].items() for r in rows])
    save_csv(output/'tables'/'projection_purity.csv', [dict(projection=k, perplexity=p['perplexity'], pca_variance=sum(p['pca_variance']),
                                                            tsne_kl=p['tsne_kl'], **p['purity']) for k, p in results['projections'].items()])
    save_csv(output/'tables'/'projection_coordinates.csv', [dict(projection=k, **pt) for k, p in results['projections'].items() for pt in p['points']])
    print('Experiments exported.', flush=True)
    if report:
        from src.project2_part2_report import export_report
        export_report(results, output)
    metadata = {'generated_at': datetime.now(timezone.utc).isoformat(), 'corpus': corpus.metadata(),
                'experiments': results['experiments'], 'epochs': epochs, 'python': platform.python_version(),
                'dependencies': {n: version(n) for n in ('numpy', 'nltk', 'matplotlib', 'reportlab')},
                'elapsed_seconds': perf_counter() - started,
                'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), ROOT/'src/word2vec.py', ROOT/'src/embedding_study.py', ROOT/'src/tsne.py',
                     ROOT/'src/embedding_view.py', ROOT/'src/project2_part2_report.py', ROOT/'src/project2_part2_figures.py']},
                'reproduce': f'python scripts/run_project2_part2.py --snapshot "{snapshot}" --epochs {epochs}',
                'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.rglob('*')) if p.is_file() and p.name not in {'submission.zip', 'run_metadata.json'}}}
    save_json(output/'run_metadata.json', metadata)
    with zipfile.ZipFile(output/'submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            # Models are large and rebuilt by this script; the zip carries the evidence, not the weights.
            if path.is_file() and path.name != 'submission.zip' and 'models' not in path.relative_to(output).parts:
                archive.write(path, path.relative_to(output))
    print(f'DONE: {output}; {perf_counter() - started:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT, help='Frozen 1,000-abstract corpus from part 1.')
    parser.add_argument('--output', type=Path, default=ROOT/'output/project2_part2_embedding')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--no-report', action='store_true', help='Only train and export experiment tables.')
    args = parser.parse_args()
    run(args.snapshot, args.output, args.epochs, args.workers, report=not args.no_report)


if __name__ == '__main__': main()
