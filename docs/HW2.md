# Homework 2 — system description, requirement map and demo guide

Due **October 6, 2026**. Two course documents define the work:

- **Brief**: `Homework-2-26-20260922.pdf`. Covers Zipf / frequency spectrum, an own Porter implementation with a before/after comparison, Word2Vec (CBOW *or* Skip-gram), dynamic-programming edit distance, required spelling correction, keyword locations and an individual demo.
- **Spec**: `project-2-details-20260929.pdf`. Covers about 1,000 English abstracts, Parts I–IX, the optional two-domain challenge, the Final Challenge, a one-page Executive Summary and RQ1–RQ5.

## Run

```powershell
.\.venv\python.exe -m streamlit run app.py
.\.venv\python.exe scripts\run_project2.py --domain-b data\experiments\1649fed9869439267d9a\corpus_snapshot.json
.\.venv\python.exe -m unittest discover -s tests
```

- The first command starts the app. Open **Homework 2 · demo → 0. Overview** in the sidebar.
- The second command regenerates `output/project2_glp1/`: tables, figures, model, `report.pdf`, `executive_summary.pdf`, `report.md`, `submission.zip` and the domain comparison. An existing compatible Skip-gram model is reused.
- The third command runs the tests.

## Requirement → page → code

The demo has five minutes, so there are **six pages**: an overview, then five pages of about one minute each. Every page starts with its requirement badges, the minutes it gets and a collapsible **Demo script** (what to say). Each page is one scroll, laid out like the reference course site (http://140.116.247.161:5173):

- **Cards**: one bordered card per question, with key numbers as chips. This is what you show during the minute.
- **More details**: collapsed blocks at the bottom with every table, plot and control, for follow-up questions.
- **Answers**: the brief's questions, answered from the measured numbers (`src/hw2_answers.py`).

| Page | Time | Requirements | Cards (shown in the demo) | More details / Answers | Code |
|---|---|---|---|---|---|
| 0 Overview & results | 0:00–0:30, 4:45–5:00 | Brief · demo; Spec · Summary, Part IX, RQ1–RQ5 | Headline numbers, Finding 1–4 | System pipeline and modules; Part IX essay, RQ1–RQ5, final challenge (report files: `output/project2_glp1/`) | `ui/hw2/overview.py` |
| 1 Data & preprocessing | 0:30–1:30 | Spec · Dataset, Parts I, II, VI; Brief · Porter | Dataset checks, A–D vocabulary table, one sentence through steps 1–5, Porter rules for one word, 100% agreement with NLTK | Browse documents, abstract trace, A–D top terms and curves, all Porter steps, C vs D, merged forms, corpus management; Part VI answers | `ui/hw2/data.py` (+ `corpus.py`, `preprocessing.py`, `porter.py`) |
| 2 Zipf's law | 1:30–2:30 | Brief · Zipf; Spec · Parts III, V, optional domains | Chips k · R² · RMSE · slope · intercept with linear vs log-log plots; **B → C** and **C → D** cards with side-by-side curves | Regions, residuals, Top 50, spectrum, figures, two domains; Q1–Q4 and domain answer | `ui/hw2/zipf.py`, `ui/hw2/domains.py` |
| 3 CF · DF · IDF | 2:30–3:15 | Spec · Parts IV, VII, VIII | 20 terms with CF, DF, CF/DF, IDF; CF–DF plot colored by IDF | DF of every term, one term across documents with reader; Part VII and VIII answers | `ui/hw2/terms.py` |
| 4 Word2Vec | 3:15–4:00 | Brief · Word2Vec | Training examples of a real sentence, viewed as Skip-gram (word → each context) or CBOW (mean of context → word); neighbours; PCA map; **Skip-gram vs CBOW** neighbours side by side | Model selection, **background training** (Skip-gram or CBOW, progress bar while the page stays usable), loss, σ(v·u), config | `ui/hw2/word2vec.py`, `ui/hw2/training.py` |
| 5 Edit distance & search | 4:00–4:45 | Brief · matching, spelling, locations | Query with auto-correct; **keyword interpretation** chips (`semaglutde → semaglutide (d=1)`); **publication-year filter**; top 5 hits with **location chips** (`word @offset (d=…)`); DP table of the correction | Spelling candidate ranking | `ui/hw2/search.py` |

Shared code:
- `ui/hw2/registry.py`: pages, minutes, requirements and demo script.
- `ui/hw2/common.py`: state, loading, widgets and the reader.
- `ui/hw2/shell.py`: page frame, corpus selector in the sidebar and previous/next navigation.
- `pages/10–15_HW2_*.py`: entry points.

## Method summary

- **Corpus.** The corpus is the PubMed query for GLP-1 / glucagon-like peptide-1, restricted to English records that have an abstract. It is frozen in PubMed's returned order. Only AbstractText is analysed. The fingerprint is a SHA-256 of the query and texts.
- **Conditions.** They are cumulative:
  - A: whitespace tokens + lowercase.
  - B: A + boundary punctuation removed. Internal marks are kept (GLP-1, 0.5, 24%).
  - C: B + NLTK English stopwords removed.
  - D: C + own Porter 1980 on ASCII alphabetic words.
- **Statistics.** CF counts occurrences. DF counts abstracts. IDF = ln(N/DF).
- **Zipf fit.** Unweighted OLS of log10 CF on log10 rank. The exponent is k = −slope. Regions are head 10%, middle 70% and tail 20% of the ranks.
- **Porter.** Original rules, steps 1a–5b. `porter_steps()` returns each step's rule. It agrees with NLTK `PorterStemmer(ORIGINAL_ALGORITHM)` on all 10,799 alphabetic words of the condition C vocabulary.
- **Word2Vec (Skip-gram and CBOW).**
  - Negative sampling in NumPy. Training text is condition C, and windows stay within one sentence.
  - Skip-gram: each word predicts every word in its window. CBOW: the mean of the window's input vectors predicts the center word; as in the original word2vec (`cbow_mean=1`), the full hidden gradient is applied to each context word.
  - Settings: 64 dimensions, window 5, min count 2, 5 negatives, 15 epochs, seed 42. The report uses Skip-gram; its model key is unchanged by the architecture option.
  - The loss is a fixed training probe, not held-out accuracy.
  - Training from the page runs in a background thread (`ui/hw2/training.py`); the same settings are never trained twice at once.
- **Edit distance.** Optimal string alignment by dynamic programming: insert, delete, substitute and adjacent transpose, each costing 1.
  - Spelling correction ranks candidates by distance, then by corpus frequency. The distance limit is 1 for words of ≤ 4 letters and 2 otherwise.
  - FUZZY search matches whole tokens within 1–2 edits. PARTIAL search matches substrings.
- **Domains.** The first *n* abstracts of each corpus are used, with the same *n* for both and condition C. Domain B is 500 PubMed abstracts from `("deep learning"[Title] OR "machine learning"[Title] OR "neural network"[Title]) AND (algorithm[Title/Abstract] OR architecture[Title/Abstract])`, excluding GLP-1 PMIDs.

## Measured results (current run)

| Condition | Tokens | Vocabulary | Hapax | k | R² | RMSE log10 |
|---|---:|---:|---:|---:|---:|---:|
| A | 245,416 | 28,792 | 16,855 | 1.0915 | 0.9571 | 0.1003 |
| B | 243,843 | 19,231 | 9,692 | 1.2508 | 0.9687 | 0.0975 |
| C | 177,052 | 19,115 | 9,688 | 1.2095 | 0.9633 | 0.1024 |
| D | 177,052 | 15,569 | 8,241 | 1.2663 | 0.9613 | 0.1101 |

**Two domains (condition C, 500 abstracts each):**

| Measure | GLP-1 | Machine learning |
|---|---:|---:|
| Tokens | 88,036 | 93,131 |
| Vocabulary | 12,626 | 15,191 |
| k | 1.125 | 1.073 |
| R² | 0.961 | 0.962 |
| Shared vocabulary (Jaccard) | 22.4% | 22.4% |

The two domains share none of their top-10 terms.

**Word2Vec:** 9,427 words × 64 dimensions, same settings for both architectures.

| | Skip-gram | CBOW |
|---|---|---|
| Training time | ~205 s | ~48 s |
| Probe loss (final; not comparable across architectures, different objectives) | 4.16 → 1.86 | 1.74 |
| Nearest to `semaglutide` | liraglutide, dulaglutide, once-weekly, 2·4, avexitide | tirzepatide, liraglutide, dulaglutide, subcutaneous, orforglipron |

**Year filter:** 995 of the 1,000 abstracts are from 2026 (PubMed returns the newest first), so the filter appears only when a result set spans several years.

**Search on the course corpus:**

| Example | Result |
|---|---|
| Spelling suggestion `semaglutde` | `semaglutide` (edit distance 1, one insertion) |
| Standard query `glp-1 receptor` | 493 papers |
| Partial query `semaglut` | 238 papers |
| Fuzzy query `semaglutde` | 232 papers |

## Demo route (5 minutes)

Scroll through the cards; open More details or Answers only when asked.

| Time | Page | Show | Say |
|---|---|---|---|
| 0:00–0:30 | Overview | Metrics row | 1,000 GLP-1 abstracts; the five pages follow the two briefs. |
| 0:30–1:30 | Data | A–D table, the sentence trace, `generalizations → gener` | Steps 1–5; vocabulary 28,792 → 15,569; our Porter matches the reference on all 10,799 words. |
| 1:30–2:30 | Zipf | Metrics, both plots | Nearly straight in log-log, k ≈ 1.25, R² 0.97, but R² is not proof (Q3); the tail is tied counts. |
| 2:30–3:15 | CF · DF · IDF | Table and plot | `and`: DF ≈ N so IDF ≈ 0; a concentrated term has high CF, DF 1 and maximum IDF. |
| 3:15–4:00 | Word2Vec | Move the center word, switch Skip-gram ↔ CBOW; neighbours of `semaglutide` in both models | Skip-gram predicts the window from the word, CBOW the word from the window; both place the GLP-1 drugs together. |
| 4:00–4:45 | Edit distance & search | Search `semaglutde receptor`, open a hit; DP table for `livr → liver` | The typo is auto-corrected (0 → 193 papers); the popup highlights every match. The DP table shows one insertion; `livr` is also one edit from `live`, and `liver` wins by corpus frequency (233 vs 7). |
| 4:45–5:00 | Overview | Finding 1–4 | Conclusion. |

## Limits

- The corpus is a single topic in PubMed order, not a random sample. R² describes a log-space line; it is not a power-law test.
- Cosine similarity is not synonymy.
- The comparison covers only two single-topic samples.
- No relevance judgments, compression measurements or embedding benchmarks were collected.

## Legacy

The first Homework 2 version (100 diabetes abstracts, one page, before/after stemming only) is kept in `legacy/hw2_diabetes/`. This includes `src/homework2.py`, `run_homework2.py`, its outputs and documents. The seven-screen single-page Project 2 workspace and its documentation are kept in `legacy/project2_single_page/`. A ten-page variant preceded the current six pages. Neither is imported by the app or the tests.
