#!/usr/bin/env python3
"""Run reproducible Homework 2 experiments and export tables, a model and report."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.homework2 import compare_stemming, load_experiment_corpus, rows_csv, training_sentences
from src.embedding_view import pca_2d, projection_rows
from src.word2vec import Word2VecConfig, model_key, train_skipgram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/pubmed_abstract_meta.json")
    parser.add_argument("--n", type=int, default=None, help="Documents to use; defaults to the active manifest's default_count")
    parser.add_argument("--output", type=Path, default=ROOT / "output/homework2")
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-stopwords", action="store_true")
    parser.add_argument("--stem-embeddings", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        limit = args.n if args.n is not None else int(manifest.get("default_count", 100))
        corpus = load_experiment_corpus(args.manifest, limit)
        comparison = compare_stemming(corpus, remove_stopwords=not args.keep_stopwords)
        config = Word2VecConfig(dimensions=args.dimensions, window=args.window, epochs=args.epochs,
                               min_count=args.min_count, negatives=args.negatives, seed=args.seed)
        preprocessing = {"remove_stopwords": not args.keep_stopwords, "stem": args.stem_embeddings,
                         "stemmer": "local Porter 1980"}
        print(f"Corpus: {len(corpus.documents)} articles, {corpus.query}", flush=True)
        model = train_skipgram(training_sentences(corpus, remove_stopwords=not args.keep_stopwords, stem=args.stem_embeddings),
                               config, corpus_metadata=corpus.metadata(), preprocessing=preprocessing,
                               progress=lambda epoch, total, loss: print(f"Epoch {epoch}/{total}: fixed training probe loss = {loss:.4f}", flush=True))
        args.output.mkdir(parents=True, exist_ok=True)
        for field in ("before", "after", "spectrum_before", "spectrum_after", "merged_groups"):
            (args.output / f"{field}.csv").write_text(rows_csv(comparison[field]), encoding="utf-8")
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output / "experiment_corpus.json").write_text(json.dumps(corpus.metadata(), indent=2), encoding="utf-8")
        model.save(args.output / "skipgram.npz")
        key = model_key(config, corpus.fingerprint, preprocessing)
        model.save(ROOT / "data/homework2" / f"skipgram_{key}.npz")
        (args.output / "training.json").write_text(json.dumps(model.metadata, indent=2), encoding="utf-8")
        examples = [word for word in ("diabetes", "insulin", "glucose", "treatment", "patients", "diabet", "patient") if word in model.words][:5]
        if not examples:
            examples = model.words[:3]
        neighbors = [dict(query=word, **row) for word in examples for row in model.most_similar(word, 10)]
        (args.output / "neighbors.csv").write_text(rows_csv(neighbors), encoding="utf-8")
        coordinates, variance = pca_2d(model.vectors)
        points = projection_rows(model, coordinates, mode="frequent", limit=len(model.words))
        (args.output / "pca_coordinates.csv").write_text(rows_csv(points), encoding="utf-8")
        (args.output / "pca_metadata.json").write_text(json.dumps({
            "method": "PCA via SVD", "input": "All L2-normalized, mean-centered center vectors",
            "model_key": key, "explained_variance_ratio": variance.tolist(),
            "vocabulary_size": len(model.words)}, indent=2), encoding="utf-8")
        before, after = comparison["stats_before"], comparison["stats_after"]
        history = model.metadata["history"]
        report = [
            "# Helix Homework 2: running results", "",
            f"Subject query: `{corpus.query}`. Documents: **{len(corpus.documents)}**.", "",
            "Scope: abstracts plus keywords; titles and bodies excluded. Keywords labels are removed. "
            "Corpus selection follows the saved PMID manifest rather than the mixed search index.", "",
            f"Corpus fingerprint: `{corpus.fingerprint}`", "",
            f"Stopword removal: **{preprocessing['remove_stopwords']}**. Both Zipf curves use exactly the same corpus and filtering.", "",
            "## Zipf and local Porter 1980", "",
            "| Metric | Before Porter | After Porter |", "|---|---:|---:|",
            *[f"| {field} | {before[field]} | {after[field]} |" for field in ("tokens", "vocabulary", "hapax", "log_log_slope", "log_log_r_squared")], "",
            f"Vocabulary reduction: **{comparison['vocabulary_reduction_percent']:.2f}%**. "
            f"Groups merging multiple word forms: **{len(comparison['merged_groups'])}**.", "",
            "Frequency tables contain every term, rank, occurrence count and relative frequency. "
            "The spectrum counts how many terms have each frequency. The UI plots both rank-frequency curves "
            "with linear/log axes and a 1/r reference. Log-log regression is descriptive, not a goodness-of-fit test for a power law.", "",
            "## Word2Vec", "",
            "NumPy Skip-gram with negative sampling, trained locally from random initialization. "
            "Context windows stay within sentences and documents after filtering and subsampling.", "",
            "```json", json.dumps(model.metadata["config"], indent=2), "```", "",
            f"Embedding stemming: **{args.stem_embeddings}**. Vocabulary: **{len(model.words)}**. "
            f"Pairs processed across epochs: **{model.metadata['training_pairs']:,}**. "
            f"Training time on this machine: **{model.metadata['elapsed_seconds']:.2f} seconds**.", "",
            f"Fixed training probe loss: **{history[0]['probe_loss']:.4f} → {history[-1]['probe_loss']:.4f}**. "
            "This uses training examples with fixed negative samples, not held-out accuracy.", "",
            "| Query | Five closest words (cosine similarity) |", "|---|---|",
            *[f"| {word} | " + ", ".join(f"{row['word']} ({row['cosine_similarity']:.3f})" for row in model.most_similar(word, 5)) + " |" for word in examples], "",
            "The small corpus gives noisy neighbors. They are context associations, not guaranteed biomedical synonyms. "
            "Increasing epochs can fit the training corpus without improving generalization.", "",
            "## Demonstration and artifacts", "",
            f"PCA 2D retains **{sum(variance):.2%}** of variance in the normalized vocabulary "
            f"(PC1 {variance[0]:.2%}, PC2 {variance[1]:.2%}). The interactive Word2Vec map "
            "supports nearest/frequent words, labels, tooltips and zoom/pan. "
            "Neighbor ranking uses original vectors, not 2D distances.", "",
            "Open the Homework 2 page: inspect the corpus, compare Zipf/Porter, explore the saved Word2Vec model, "
            "and try `diabtes treatment` in the spelling-correction panel. "
            "The search page also offers direct Partial/Fuzzy matching and Open & locate for every "
            "occurrence's character range, sentence and section, with navigation and CSV export.", "",
            "- `before.csv`, `after.csv`: complete rank-frequency tables.",
            "- `spectrum_before.csv`, `spectrum_after.csv`: term-frequency spectra.",
            "- `merged_groups.csv`: Porter groups and source-form counts.",
            "- `comparison.json`, `experiment_corpus.json`: settings, counts and selected PMIDs.",
            "- `training.json`, `neighbors.csv`, `skipgram.npz`: training history, examples and both learned matrices.", "",
            "- `pca_coordinates.csv`, `pca_metadata.json`: complete 2D map and retained variance.", "",
            "System description and reproducibility instructions: [docs/homework2.md](../../docs/homework2.md).", "",
            "Sources: [Porter (1980)](https://tartarus.org/martin/PorterStemmer/def.txt), "
            "[Mikolov et al. (2013)](https://arxiv.org/abs/1310.4546).", "",
        ]
        (args.output / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
        print(f"Saved comparison, Word2Vec model and RESULTS.md to {args.output}")
    except (ValueError, OSError, KeyError, LookupError) as exc:
        print(f"Homework 2 failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
