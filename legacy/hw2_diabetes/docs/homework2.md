# Helix: Homework 2 system description

Course brief: `Homework-2-26-20260922.pdf`, due October 6, 2026. The assignment
requires a term distribution experiment, a Porter comparison, one Word2Vec
architecture, spelling correction, running results and an individual demo.
Helix implements Skip-gram; CBOW is an alternative in the brief, not an additional requirement.

## Corpus and preprocessing

The default experiment reads 100 diabetes PubMed records selected by
`data/raw/pubmed_abstract_meta.json`. It loads their `PMID<id>.xml` source files
directly, so unrelated papers in the retrieval index do not enter the experiment.
The UI allows selecting 10 to 1000 documents, limited to the manifest's actual size.
Selection preserves manifest order; it is not a new random sample on each run.
Missing files and invalid IDs produce explicit errors instead of silently changing the corpus.

Both distribution experiments and Word2Vec use the same selected documents:
abstract text plus keywords, excluding titles and full bodies. Markup is removed,
and the literal `Keywords:` label is removed while the keywords themselves remain.
The existing tokenizer lowercases text and preserves biomedical alphanumeric,
hyphenated, Unicode, decimal and percentage tokens. English stopword removal is
optional and is applied identically before and after stemming.

The manifest query documents the selection topic; it does not prove that every
article has identical subject matter. A SHA-256 fingerprint records the query,
ordered PMIDs, extracted texts and preprocessing implementation version.

## Zipf distribution and Porter comparison

`src/homework2.py` aggregates occurrence counts across the entire selected corpus.
It sorts terms by decreasing frequency (alphabetically for ties), assigns ranks
starting at 1, and calculates relative frequencies. The frequency spectrum is
the number of distinct terms with exactly k occurrences.

The page shows rank-frequency curves before and after stemming on linear or
log-log axes, plus the reference f(r) = f(1)/r. It also shows total token counts,
vocabulary size, hapax counts, vocabulary reduction and merged word-form groups.
The slope and R² use ordinary least squares on log10(rank) and log10(frequency)
over the complete vocabulary. These are descriptive statistics, not a formal
test that the corpus follows a power law.

`src/porter.py` implements the original Porter 1980 rules locally: measure m,
vowel/consonant conditions, steps 1a–1c, suffix transformations in steps 2–3,
removal in step 4, and final e/l adjustments in step 5. This uses the original
ABLI rule, not the later NLTK extensions. Only ASCII alphabetic tokens are stemmed;
biomedical identifiers such as `GLP-1` retain their tokenized form. Empty results
from original rules are excluded and counted explicitly. The existing retrieval
pipeline continues to use NLTK's default Porter variant, so its vocabulary can
differ slightly from the Homework 2 experiment.

## Word2Vec implementation

`src/word2vec.py` implements Skip-gram with negative sampling using NumPy, with
no pretrained vectors or Word2Vec library. A center word predicts context words
within the selected window on either side. Sentences and documents are kept
separate. Vocabulary filtering and optional frequent-word subsampling happen
before forming windows, so distances refer to the retained token sequence.

Two matrices represent center and context words. The objective for a positive
center/context pair is log sigmoid of their dot product, plus log sigmoid of the
negated dot products for noise words. Noise is drawn with replacement from
frequency^0.75; a noise draw equal to the positive context is ignored. Small
batches accumulate SGD updates, correctly adding repeated word IDs. The learning
rate decreases linearly by epoch to 10% of its initial value.

Defaults: 64 dimensions, window 5 on each side, minimum count 2, 5 negative
samples, 15 epochs, learning rate 0.025, batch size 128 and random seed 42.
Frequent-word keep probability is min(1, (sqrt(f/t)+1)*t/f), with t = 0.001 and
f the retained vocabulary's relative frequency. Subsampling can be disabled.
Word2Vec preserves unstemmed word forms by default; local Porter is optional.

The loss chart evaluates fixed training pairs with fixed noise after each epoch,
including initialization at epoch 0. It is not validation loss, retrieval
accuracy or a clinical measure. Nearest words are ranked by cosine similarity
between learned center vectors, excluding the query itself. Out-of-vocabulary
queries receive an explanation rather than invented vectors.

Models save both matrices, vocabulary, counts, configuration, corpus metadata
and loss history in a compressed NPZ archive loaded without Python pickle.
The UI reuses a saved model only when the fingerprint, preprocessing, model
version and every training setting match. A seed makes reruns reproducible in
the same numerical environment; different NumPy/platform versions can differ.

### PCA 2D map

The Word2Vec tab projects the saved center vectors with `src/embedding_view.py`.
It L2-normalizes every vector, subtracts the vocabulary mean, and computes the
first two principal components by NumPy SVD. PCA is fitted to the complete model
vocabulary; choosing another query or displayed subset never refits the axes.
The page reports the variance retained by each component and their sum.

Choose nearest neighbors or frequent words, adjust the point count, toggle
labels, hover for frequency/cosine, and zoom/pan the chart. Clicking a point shows
its neighbors. The query is included in the point limit and colored orange.
Out-of-vocabulary queries can still view frequent words. CSV exports contain the
displayed coordinates. Cosine values and neighbor selection always use the
original vectors; proximity in a 2D projection is only an approximation.
Constant vectors produce a zero map with an explicit message. This implements
the optional PCA visualization; t-SNE is not needed to run the map.

## Spelling correction and retrieval

The existing spelling module uses dynamic programming with insertions,
deletions, substitutions and adjacent transpositions (restricted
Damerau-Levenshtein / optimal string alignment distance). Candidate distance
is the primary criterion and corpus frequency breaks ties. Short words,
all-uppercase biomedical abbreviations and known vocabulary forms are retained.

The Homework 2 demo applies the suggested query to OR matching on the selected
corpus and highlights matched terms. The full retrieval page leaves correction
under the user's control through its “Did you mean” link and provides
AND/OR/PHRASE matching, BM25 ranking and document inspection.

The **Word matching** control adds direct **Partial / substring** and
**Fuzzy / edit distance** retrieval. Partial matches case-insensitive substrings
inside tokens (`diabe` matches `diabetes`); fuzzy compares entire surface tokens
using the same dynamic-programming distance with a selectable maximum of 1 or
2 edits (`diabtes` matches `diabetes`). These modes do not stem or remove query
stopwords. AND requires every query token to match; OR requires at least one.
BM25 ranks the matched terms using the existing index. Terms omitted from the
index, such as stopwords, contribute zero to BM25 but remain inspectable matches.
PHRASE stays in standard token mode and cannot cross paragraphs.

Open a result with **Open & locate → Keyword locations** to see every occurrence:
matched substring/word, query term, abstract/body section, sentence number,
document character range, range within the sentence, and edit distance. A
selector and previous/next buttons highlight one occurrence in context; the
**Search scope** tab highlights all occurrences. Results remain available when
returning from the reader. CSV and the exact searchable plaintext can be saved.

Character ranges use zero-based Unicode positions `[start, end)`, where end is
exclusive. They refer to the displayed plaintext, excluding title and sup/sub
markup, with whitespace normalized inside paragraphs and two newlines between
paragraphs. Sentence numbers begin at 1 and follow the existing heuristic EOS
rules; phrases spanning sentence boundaries report the starting sentence and
include the complete phrase in the context. Partial matches include repeated,
overlapping substrings; merged highlighting can cover more than one occurrence.

## Run and reproduce

From the project root, using the existing portable Python:

```powershell
.\.venv\python.exe scripts\run_homework2.py
.\.venv\python.exe -m streamlit run app.py
```

The command writes tables, configuration, learned vectors, nearest-word examples
and a report to `output/homework2/`. It also caches the default model for the UI.
Open **Homework 2 · Zipf & Word2Vec** from the homepage or sidebar.

To prepare a different subject in a separate directory:

```powershell
.\.venv\python.exe scripts\download_pubmed_abstracts.py --query "GLP-1[Title/Abstract] AND hasabstract[text]" --n 100 --raw data/glp1 --no-index
.\.venv\python.exe scripts\run_homework2.py --manifest data/glp1/pubmed_abstract_meta.json --n 100 --output output/homework2_glp1
```

The web page uses `data/raw/pubmed_abstract_meta.json`; the CLI supports other
manifests without replacing the existing dataset. GLP-1 and approximately 3000
records are examples in the assignment, not required dataset choices.

## Individual demonstration

1. Identify the PubMed subject query, number of articles and exact PMID list.
2. Show the Zipf curves; toggle axes and explain the frequency spectrum.
3. Compare vocabulary size, hapax and grouped forms before/after local Porter.
4. Explain Skip-gram with a center word and surrounding context; show settings,
   train, inspect the loss curve and explore `diabetes`, `insulin` or another
   word retained in the model vocabulary.
5. Enter `diabtes treatment` in the spelling demo. Explain edit operations,
   inspect the suggested query and highlighted results.
6. In Local Search choose Partial and enter `diabe ins`, then Fuzzy and `diabtes`.
   Open a result, inspect its keyword locations and navigate between occurrences.
7. In Word2Vec, show the PCA map for `diabetes`, switch to frequent words, inspect
   a point and explain the retained variance and projection distortion.
8. Download the experiment tables, coordinates and model; explain the limits below.

## Limits

A corpus of 100 abstracts is suitable for demonstrating the computation, but
nearest neighbors can be noisy and are not guaranteed synonyms. Stemming can
merge unrelated forms, and corpus-based correction cannot recover a word absent
from the vocabulary. No labeled relevance judgments or external embedding
benchmark are provided, so the system does not claim measured retrieval accuracy
or general semantic quality. Existing Homework 1 retrieval remains available.

## Sources

- [Porter (1980), original suffix-stripping specification](https://tartarus.org/martin/PorterStemmer/def.txt).
- [Mikolov et al. (2013), Skip-gram and negative sampling](https://arxiv.org/abs/1310.4546).
- Homework requirements: the local `Homework-2-26-20260922.pdf`.
