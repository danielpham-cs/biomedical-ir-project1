"""Run all detailed Project 2 experiments, train/reuse Skip-gram, and export reports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from importlib.metadata import version
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))

import numpy as np
from src.tables import rows_csv
from src.pubmed import _write_atomic
from src.zipf_experiment import load_corpus, load_snapshot, analyze_corpus, embedding_sentences, embedding_preprocessing
from src.word2vec import Word2VecConfig, Word2VecModel, train_skipgram, model_key


def save_json(path, obj):
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def export_domains(corpus, domain_b, output, size=500):
    """Optional challenge: same-size comparison with a second domain snapshot."""
    from src.domain_compare import compare_domains, domain_answer
    other = load_snapshot(domain_b)
    size = min(size, len(corpus.documents), len(other.documents))
    comparison = compare_domains(corpus, other, condition='C', size=size)
    names = ('GLP-1 (medical)', 'Machine learning (computing)')
    measures = [('Documents', 'documents'), ('Tokens', 'tokens'), ('Vocabulary', 'vocabulary'), ('Zipf exponent', 'zipf_exponent'),
                ('R2', 'r_squared'), ('RMSE log10', 'rmse_log10'), ('Hapax share', 'hapax_share')]
    rows = [{'measure': label, names[0]: comparison['A'][key], names[1]: comparison['B'][key]} for label, key in measures]
    rows.append({'measure': 'Top 10 terms', names[0]: ', '.join(comparison['A']['top_terms']), names[1]: ', '.join(comparison['B']['top_terms'])})
    _write_atomic(output/'tables'/'domain_comparison.csv', rows_csv(rows).encode('utf-8-sig'))
    answer = domain_answer(comparison, names)
    summary = {key: value for key, value in comparison.items() if key not in {'A', 'B'}}
    summary.update({side: {k: v for k, v in comparison[side].items() if k != 'rows'} for side in 'AB'}, answer=answer)
    save_json(output/'domain_comparison.json', summary)
    lines = ['', '## Optional challenge — two domains', '',
             f'Domain B query: {other.query}. {size} abstracts per domain, condition C.', '',
             f'| Measure | {names[0]} | {names[1]} |', '|---|---|---|']
    lines += [f"| {r['measure']} | {r[names[0]]:.4f} | {r[names[1]]:.4f} |" if isinstance(r[names[0]], float) else f"| {r['measure']} | {r[names[0]]} | {r[names[1]]} |" for r in rows]
    lines += ['', answer, '']
    with open(output/'report.md', 'a', encoding='utf-8') as report:
        report.write('\n'.join(lines))
    print(f'Domain comparison exported ({size} abstracts per domain).', flush=True)


def run(manifest, output, limit=None, epochs=15, analysis_only=False, snapshot=None, domain_b=None):
    started = perf_counter()
    if snapshot and limit is not None:
        raise ValueError('A frozen snapshot must be replayed in full; do not combine --snapshot and --n.')
    corpus = load_snapshot(snapshot) if snapshot else load_corpus(manifest, limit)
    print(f'Corpus: {len(corpus.documents)} abstracts, fingerprint {corpus.fingerprint}', flush=True)
    result = analyze_corpus(corpus)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'analysis.json', result)
    save_json(output/'corpus_snapshot.json', {'metadata': corpus.metadata(), 'documents': [d.to_dict() for d in corpus.documents]})
    tables = output/'tables'
    tables.mkdir(exist_ok=True)
    summary, fits = [], []
    for condition, data in result['conditions'].items():
        summary.append(dict(condition=condition, **data['stats'], **data['fits']['All']))
        for name, rows in [('frequency',data['rows']), ('selected_terms',data['selected_terms']), ('spectrum',data['spectrum'])]:
            _write_atomic(tables/f'{condition}_{name}.csv', rows_csv(rows).encode('utf-8-sig'))
        fits += [dict(condition=condition, region=region, **fit) for region, fit in data['fits'].items()]
    for name, rows in [('condition_summary',summary), ('regression',fits), ('porter_groups',result['merged_groups'])]:
        _write_atomic(tables/f'{name}.csv', rows_csv(rows).encode('utf-8-sig'))
    print('A-D frequency tables and regression exported.', flush=True)
    if analysis_only:
        return result
    config = Word2VecConfig(epochs=epochs)
    prep = embedding_preprocessing('C')
    key = model_key(config, corpus.fingerprint, prep)
    model_path = output/'skipgram.npz'
    model = None
    if model_path.exists():
        try:
            candidate = Word2VecModel.load(model_path)
            if candidate.metadata.get('model_key') == key:
                model = candidate
        except (OSError, ValueError, KeyError):
            pass
    if model is None:
        model = train_skipgram(embedding_sentences(corpus), config, corpus_metadata=corpus.metadata(), preprocessing=prep,
            progress=lambda epoch,total,loss: print(f'Skip-gram {epoch}/{total}: probe loss {loss:.4f}', flush=True))
        model.save(model_path)
    model.save(ROOT/'data/homework2'/f'skipgram_{key}.npz')
    save_json(output/'training.json', model.metadata)
    from src.project2_report import export_report
    export_report(result, model, output)
    if domain_b:
        export_domains(corpus, domain_b, output)
    metadata = {'generated_at': datetime.now(timezone.utc).isoformat(), 'corpus': corpus.metadata(),
                'run_key': result['run_key'], 'model_key': key, 'word2vec_config': asdict(config),
                'python': platform.python_version(), 'numpy': np.__version__, 'elapsed_seconds': perf_counter()-started,
                'dependencies': {name: version(name) for name in ('numpy', 'nltk', 'matplotlib', 'reportlab', 'streamlit')},
                'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT/'src/zipf_experiment.py', ROOT/'src/porter.py', ROOT/'src/word2vec.py', ROOT/'src/project2_report.py', ROOT/'src/domain_compare.py']},
                'reproduce': f'python scripts/run_project2.py --snapshot "{output / "corpus_snapshot.json"}" --epochs {epochs}',
                'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.rglob('*')) if p.is_file() and p.name not in {'submission.zip','run_metadata.json'}}}
    save_json(output/'run_metadata.json', metadata)
    with zipfile.ZipFile(output/'submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'submission.zip':
                archive.write(path, path.relative_to(output))
    print(f'DONE: {output}; {perf_counter()-started:.1f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/raw/pubmed_abstract_meta.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/project2_glp1')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Replay a previously exported corpus_snapshot.json offline.')
    parser.add_argument('--domain-b', type=Path, help='corpus_snapshot.json of a second domain for the optional two-domain comparison.')
    args = parser.parse_args()
    run(args.manifest, args.output, args.n, args.epochs, args.analysis_only, args.snapshot, args.domain_b)


if __name__ == '__main__': main()
