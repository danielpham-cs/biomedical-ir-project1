# Project 2: implementation, results and demo

Implemented September 30, 2026. The detailed brief is `project-2-details-20260929.pdf`;
the original homework requirements remain supported. This document describes the
actual implementation; the earlier two plan documents retain the design rationale.

## Run

From the repository root, use `.\.venv\python.exe -m streamlit run app.py`, then
open **3. Project 2 · GLP-1 experiments**. If setting up a new environment, install
`requirements.txt` and the NLTK English stopwords resource. No database server is needed.

The supplied environment has 1,000 GLP-1 records in the active SQLite catalog and
104 older records marked by `deleted_at`. Source XML/JSON is retained. The collection
page has an Archived papers section with explicit restore and index rebuild.

## System and data flow

1. **Corpus:** `src/zipf_experiment.py` validates unique PMIDs, English language and
   nonempty AbstractText from raw XML. Titles, Keywords and article bodies are excluded.
   The query and ordered abstract texts determine the experiment fingerprint; raw
   XML hashes record source provenance.
2. **A–D analysis:** A lowercases whitespace tokens; B normalizes Unicode punctuation
   and strips boundary punctuation while preserving internal biomedical forms;
   C removes the saved English stopword list; D applies local Porter 1980 to ASCII
   alphabetic tokens. CF counts tokens, DF counts documents, and IDF is `ln(N/DF)`.
3. **Zipf:** unweighted OLS of `log10(CF)` on `log10(rank)`. Exponent `k=-slope`;
   R² and RMSE use log space. Regions are fixed at 10%/70%/20% of vocabulary ranks,
   with ceiling boundaries. A constant tail has undefined R², displayed as N/A.
4. **Word2Vec:** NumPy Skip-gram with negative sampling, condition C, sentence-bounded
   windows, fixed seed. Compatibility keys include corpus fingerprint, pipeline and
   every training parameter. Saved compatible models can be selected; training is explicit.
   Neighbors use full-vector cosine; PCA uses normalized vectors and shows retained variance.
5. **Search:** an abstract-only inverted index supports AND/OR/PHRASE with BM25 ranking.
   This uses the existing retrieval tokenizer and NLTK Porter, explicitly distinct
   from the A–D experiment. Partial and fuzzy modes are available; fuzzy allows one
   or two edits. Suggested spelling correction requires Apply. Reader highlights and
   match locations use the same matching plan as retrieval. CF/DF links instead use
   exact selected-condition token spans, so an old query cannot contaminate the reader.
6. **Report:** `scripts/run_project2.py` exports full frequency/regression tables,
   figures, model metadata, PCA coordinates, RQ1–RQ5, the IR discussion, Final Challenge,
   full report and one-page executive summary. A frozen JSON snapshot supports offline
   replay using `--snapshot`. Run metadata records dependency versions, code hashes
   and artifact checksums; `submission.zip` packages the run.

## UI and reference website

The user-provided [Biomedical IR reference](http://140.116.247.161:5173/corpus) was
inspected through its publicly served UI modules. Its useful flows are implemented
in the existing Streamlit app: PubMed query/maximum documents, a persistent list of
corpora, shared corpus selection, document filtering, A–D/Zipf/CF–DF exploration,
fuzzy search and matched locations, trained-model selection, neighbors and PCA.
This is a functional adaptation, not a React rewrite or a claim of pixel equivalence.

Additional corpora use a SQLite registry and separate local XML/snapshots under
`data/experiments/`. Fetching validates complete batches before registration,
supports retry/resume, and records actual size when fewer valid records exist.
Archived corpora retain all files and can be restored. The course reference remains
pinned; adding a corpus does not overwrite the 1,000-paper submission.

The page keeps Helix's teal palette. Paper titles have normal text color and teal
hover/focus; they open the internal reader. URL state carries the corpus, filters,
page, query/matching settings, condition/term and theme through reader/back links.
Permanent session state preserves settings when lazy screens unmount their widgets.

## Measured course results

| Condition | Tokens | Vocabulary | Mean tokens / abstract | Hapax | k | R² | RMSE log10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 245,416 | 28,792 | 245.416 | 16,855 | 1.0915 | 0.9571 | 0.1003 |
| B | 243,843 | 19,231 | 243.843 | 9,692 | 1.2508 | 0.9687 | 0.0975 |
| C | 177,052 | 19,115 | 177.052 | 9,688 | 1.2095 | 0.9633 | 0.1024 |
| D | 177,052 | 15,569 | 177.052 | 8,241 | 1.2663 | 0.9613 | 0.1101 |

Condition B examples: `and` has CF=10,988 and DF=997; `glp-1` has CF=1,606 and
DF=573; `cdr` has CF=15 and DF=1. All 20 selected terms have textbook IDF, with
per-document TF/TF×IDF and condition-aware reader links available in the UI.

The published Skip-gram model has 9,427 words × 64 dimensions, window 5,
minimum count 2, five negative samples, 15 epochs and seed 42. Final fixed training
probe loss is about 1.8601. This is a training diagnostic, not held-out accuracy.

Verified demo queries on this snapshot: `glp-1 receptor` (standard AND) returns
493 papers; `semaglutide` returns 232; partial `semaglut` returns 238; fuzzy
`semaglutde` at one edit returns 232. The spelling suggestion for `semaglutde`
is `semaglutide`. These counts refer to the abstract-only Project 2 index.

## Individual demo: 10–12 minutes

1. Corpus (1 min): show the source query, 1,000 unique PMIDs, title/PMID filter and internal reader.
2. Preprocessing (1.5 min): trace one actual abstract through A–D; inspect Porter groups.
3. Zipf (2.5 min): compare linear/log views and observed/reference/fitted curves;
   explain slope, k, R², RMSE, regional fits and tied-frequency tails.
4. CF/DF/IDF (1.5 min): compare `and`, `glp-1` and `cdr` in B; open a paper and verify TF.
5. Word2Vec (1.5 min): explain center/context, inspect `glp-1` or `semaglutide`,
   show neighbors, loss and PCA variance. Use the existing full model during the demo.
6. Search (2 min): enter `semaglutde`, apply correction, try partial `semaglut`,
   open a hit and use Previous/Next match; return with settings intact.
7. Report (1 min): explain RQ1–RQ5/Final Challenge and download the PDF summary/bundle.

The UI includes this walkthrough, but the individual presentation still needs to
be performed and explained by the student.

## Validation and limits

- All 96 backend and UI regression tests pass. Run them with `.\.venv\python.exe -m unittest discover -s tests -v`.
- Synthetic power laws test exponent recovery; constant tails test undefined R².
  CF/DF/IDF, abstract-only extraction, spelling and match offsets have numerical tests.
- AppTest exercises all seven lazy screens, retained filters/theme, internal reader
  roundtrips, explicit spelling correction, Partial/Fuzzy, term-aware highlights,
  match navigation, stale-report handling and archived/unavailable records.
- Corpus-store tests cover duplicate IDs, non-English exclusion, network failure,
  staged resume, exact-request reuse, snapshot replay/tampering and archive/restore.
- Report PDF pages were rendered and inspected; the executive summary is exactly one page.
- Native-browser visual QA was stopped by Computer Use because it could not reliably
  determine the browser URL. Dark-mode CSS is implemented and theme persistence is
  tested; final browser color/layout verification remains manual.

This is one topic selected in PubMed's returned order, not a random cross-domain
sample. R² does not prove a power law, cosine does not establish synonymy, and no
relevance labels or compression benchmark were collected. CBOW and a second-domain
comparison are not required because Skip-gram satisfies the alternative in the brief.
