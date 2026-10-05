# Helix Homework 2: running results

Subject query: `diabetes mellitus[Title/Abstract] AND hasabstract[text]`. Documents: **100**.

Scope: abstracts plus keywords; titles and bodies excluded. Keywords labels are removed. Corpus selection follows the saved PMID manifest rather than the mixed search index.

Corpus fingerprint: `c8aa5e6c8931f8d386e2c65fa697d53fd7eba5ec4fbd5ec0cec96d3f318b6ff5`

Stopword removal: **True**. Both Zipf curves use exactly the same corpus and filtering.

## Zipf and local Porter 1980

| Metric | Before Porter | After Porter |
|---|---:|---:|
| tokens | 19639 | 19639 |
| vocabulary | 5170 | 4185 |
| hapax | 2656 | 2060 |
| log_log_slope | -0.896664828773504 | -0.98782805525661 |
| log_log_r_squared | 0.9525813473086111 | 0.9529811015643634 |

Vocabulary reduction: **19.05%**. Groups merging multiple word forms: **639**.

Frequency tables contain every term, rank, occurrence count and relative frequency. The spectrum counts how many terms have each frequency. The UI plots both rank-frequency curves with linear/log axes and a 1/r reference. Log-log regression is descriptive, not a goodness-of-fit test for a power law.

## Word2Vec

NumPy Skip-gram with negative sampling, trained locally from random initialization. Context windows stay within sentences and documents after filtering and subsampling.

```json
{
  "dimensions": 64,
  "window": 5,
  "min_count": 2,
  "negatives": 5,
  "epochs": 15,
  "learning_rate": 0.025,
  "sample": 0.001,
  "batch_size": 128,
  "seed": 42
}
```

Embedding stemming: **False**. Vocabulary: **2514**. Pairs processed across epochs: **1,801,946**. Training time on this machine: **10.83 seconds**.

Fixed training probe loss: **4.1558 → 2.3048**. This uses training examples with fixed negative samples, not held-out accuracy.

| Query | Five closest words (cosine similarity) |
|---|---|
| diabetes | mellitus (0.886), man (0.868), korean (0.868), hypothyroidism (0.859), type (0.853) |
| insulin | resistance (0.932), impaired (0.918), metabolism (0.889), tolerance (0.887), secretion (0.885) |
| glucose | blood (0.918), fasting (0.917), metabolism (0.908), pressure (0.904), lipid (0.903) |
| treatment | twice (0.884), postoperative (0.875), concerns (0.866), corticosteroids (0.862), preoperative (0.862) |
| patients | ami (0.861), treated (0.856), ptsd (0.852), three (0.847), occurred (0.829) |

The small corpus gives noisy neighbors. They are context associations, not guaranteed biomedical synonyms. Increasing epochs can fit the training corpus without improving generalization.

## Demonstration and artifacts

Open the Homework 2 page: inspect the corpus, compare Zipf/Porter, explore the saved Word2Vec model, and try `diabtes treatment` in the spelling-correction panel. The existing search page also offers correction, highlighted snippets and BM25-ranked matches.

- `before.csv`, `after.csv`: complete rank-frequency tables.
- `spectrum_before.csv`, `spectrum_after.csv`: term-frequency spectra.
- `merged_groups.csv`: Porter groups and source-form counts.
- `comparison.json`, `experiment_corpus.json`: settings, counts and selected PMIDs.
- `training.json`, `neighbors.csv`, `skipgram.npz`: training history, examples and both learned matrices.

System description and reproducibility instructions: [docs/homework2.md](../../docs/homework2.md).

Sources: [Porter (1980)](https://tartarus.org/martin/PorterStemmer/def.txt), [Mikolov et al. (2013)](https://arxiv.org/abs/1310.4546).

## PCA 2D and keyword locations

The saved model projects 2,514 normalized vectors to 2D. PC1 retains 33.66%, PC2 17.49%, for a total of 51.15%. PCA is fitted to the entire vocabulary; the UI selects nearest or frequent words after projection. Cosine neighbors use the original vectors. Complete coordinates and settings are in `pca_coordinates.csv` and `pca_metadata.json`.

Local Search supports direct Partial (`diabe ins`) and Fuzzy (`diabtes`, 1 edit) retrieval. Open & locate shows all occurrences with zero-based character ranges, sentence/section locations, context, navigation and CSV export.
