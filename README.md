# Helix — Biomedical IR Projects #1 & #2

**Helix** is a course workspace for keyword-based retrieval over **PubMed/PMC XML**. Its default workflow is: PMID/PMCID → download local XML/JSON + SQLite catalog → keyword search.

Course: Biomedical Information Retrieval · HW1 due September 22, 2026 · HW2 due October 6, 2026

## Features

### Active corpus: GLP-1 (September 29, 2026)

The active library now contains **1,000 English GLP-1 PubMed abstracts** in SQLite,
XML/JSON and the retrieval index. The previous **104 papers are soft-deleted** using
`articles.deleted_at`; their source files remain intact. Rebuilding the index does
not restore hidden papers. The Homework 2 page defaults to the active 1,000-paper
manifest. The detailed A–D analysis, 15-epoch Skip-gram model and report have now
been generated on these 1,000 abstracts. Earlier diabetes outputs are archived under
`legacy/hw2_diabetes/`.

```powershell
.\.venv\python.exe scripts\download_glp1_corpus.py --n 1000
```

Downloads resume from `data/staging/glp1`; a successful rerun validates the current
collection without replacing it. The script keeps a snapshot in `data/archives/`
before switching corpora. To restore the previous active collection while keeping
all downloaded files:

```powershell
.\.venv\python.exe scripts\download_glp1_corpus.py --restore data\archives\before_glp1_20260929T080819389728Z
```

### Homework 2 demo (restructured September 30)

Start the app from the repository root:

```powershell
.\.venv\python.exe -m streamlit run app.py
```

The sidebar group **Homework 2 · demo** has six pages sized for a five-minute demo:

0. Overview & results
1. Data & preprocessing
2. Zipf's law
3. CF · DF · IDF
4. Word2Vec
5. Edit distance & search

Each page is one scroll of cards, one per question, with key numbers as chips; collapsed **More details** and **Answers** blocks at the bottom hold everything else. The two-domain comparison is under Zipf → More details; the report, RQ answers and downloads are on the Overview.

Opening a page never trains a model or downloads from PubMed. The Data page (Details → Manage corpora) can fetch or archive additional corpora; these live in `data/experiments/` and never replace the pinned 1,000-paper course corpus.

Generate or reuse the complete analysis, model, PDF report, one-page summary and two-domain comparison:

```powershell
.\.venv\python.exe scripts\run_project2.py --domain-b data\experiments\1649fed9869439267d9a\corpus_snapshot.json
```

The outputs go to `output/project2_glp1/`:
- `report.pdf`, `executive_summary.pdf`, `report.md`
- `analysis.json`, CSV tables and figures
- `skipgram.npz`, `corpus_snapshot.json`
- `domain_comparison.json`
- checksums and `submission.zip`

To replay the frozen texts offline, add `--snapshot output\project2_glp1\corpus_snapshot.json`.

See [docs/HW2.md](docs/HW2.md) for the requirement → page → code map, measured results and the 12-minute demo route. Earlier Homework 2 versions are in `legacy/`.

The existing retrieval pages below continue to support Homework 1.

- Parse PMC XML → title, abstract, body, PMID/PMCID, journal, year
- Preprocessing: lowercase, tokenize, English **stop-words**, **Porter stemming** (NLTK / Porter 1980)
- **Inverted index** with term frequencies and positions
- Matching: **AND/OR/PHRASE** candidate selection and **BM25** ranking
- Document statistics: characters, words, sentences, vocabulary
- Rule-based **EOS** sentence splitter (abbreviation-aware)
- Streamlit course UI: dataset overview, XML import/index settings, retrieval comparison, document analysis, and a methods/EOS playground

## Simple workflow

1. Open **Search and analyze abstract** and enter a PMID (for example `30153250`) or PMCID (for example `PMC13559554`).
2. Click **Search abstract**. Helix retrieves the PubMed record and calculates character, word, unique-word, term-frequency, and EOS sentence statistics from its abstract.
3. Click **Import paper to library** to store the displayed paper and update the search index in one step. **Download PubMed XML** remains available for a browser download.

`data/library.sqlite3` is a local SQLite database built into Python; no database server, account, or separate setup is needed. The database tracks IDs, metadata, saved-file paths, full-text availability, and save time. XML/JSON stay as normal local files so the index can be rebuilt later.

## Website structure aligned with the homework

- **Overview:** actual corpus/index counts and the path from XML to results.
- **01 · XML collection:** upload article XMLs, inspect full-text coverage and corpus statistics,
  choose stopword/stemming settings, and build the index.
- **02 · Search & document analysis:** keyword retrieval, AND/OR/PHRASE selection and BM25 ranking, matching
  time, full-text snippets, document statistics/charts, and sentence exploration.
- **03 · Methods & EOS demo:** system components, weighting formulas, preprocessing trace,
  editable sentence-splitting examples, and an individual-demo guide.
- **Download & save article:** one-step online XML/JSON acquisition and SQLite/local-file storage.

See [the requirement-to-feature mapping](docs/homework_alignment.md). The supplied
one-page homework PDF does not require a particular corpus size, a two-paper
presentation, or a specific website style. Existing paper notes are optional background.

The full article body is no longer truncated. **Rebuild existing indexes from the
original XML** to apply full-text parsing and retain numeric terms such as `type 2`.

## Quick start

### Abstract-only PubMed corpus (recommended)

The homework statistics use abstracts and keywords. Download a reproducible
abstract-only dataset without requesting PMC full text:

```powershell
.\.venv\python.exe scripts\download_pubmed_abstracts.py --n 100
```

The command saves individual `PMID<id>.xml` files under `data/raw/`, writes
`pubmed_abstract_meta.json`, and rebuilds the local BM25 index. Change the
topic with `--query`, for example:

```powershell
.\.venv\python.exe scripts\download_pubmed_abstracts.py --query "asthma treatment[Title/Abstract] AND hasabstract[text]" --n 100
```

```bash
cd biomedical-ir-project1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Download ~150 PMC Open Access articles (diabetes topic by default)
python scripts/download_sample.py --n 150

# 2) Parse XML and build inverted index
python scripts/build_index.py

# 3) Launch product UI (homepage → Search)
streamlit run app.py
```

Opens the **Helix** homepage. Enter a query or choose **Open search workspace** to start.

### Windows / PowerShell

With a standard Python installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_sample.py --n 150
.\.venv\Scripts\python.exe scripts/build_index.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you use a portable Python stored directly in `.venv`, use
`.\.venv\python.exe` instead of `.\.venv\Scripts\python.exe`.
Open http://localhost:8501. Download and build are only needed for the initial
collection or when changing the articles or preprocessing options.

The homepage accepts a query directly. The search workspace provides matching
preferences, paginated results, and a document reader that preserves the submitted
query when returning to results. Missing collections show setup instructions.
The UI uses local fonts and CSS and does not need an external image or font service.

## Find and download a paper by PMID or PMCID

1. Open **Find by PMID / PMCID** from the homepage or search sidebar. This works without
   an existing local corpus or index.
2. Enter a numeric PMID, `PMID: 30153250`, a PMCID such as `PMC13559554`, or a
   PubMed/PMC article URL, then choose
   **Find on PubMed**. The app shows the title, authors, journal, year, DOI, and abstract.
   PMC IDs are resolved using the official [NCBI ID Converter](https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/).
   Keep the `PMC` prefix: plain numbers are interpreted as PMIDs, not PMCIDs.
   If a PMC record has no linked PMID, the app explains that its PubMed record
   is not available rather than looking up an unrelated numeric PMID.
3. Click **Import paper to library** to save the record directly in Helix. This
   uses the displayed record without another PubMed request. Under **Import options**,
   optionally include PMC full text when available. Browser download is offered for **PubMed XML**.
4. Imported articles are cataloged in SQLite. Their files are
   stored as `data/raw/PMID<id>.xml`, `data/raw/PMID<id>.json`, and, if fetched,
   `data/raw/PMC<id>.xml`. Saving the same ID updates these files.
5. The search index updates automatically; use **Search your library** to find the paper.
   Existing indexed papers and stemming/stopword settings are preserved. PubMed
   and PMC records of the same paper are merged, preferring the full text.

JSON contains the existing `Document` fields (`doc_id`, `pmid`, `pmcid`, `title`,
`abstract`, `body`, `journal`, `year`, `full_text`) plus `authors`, `doi`,
`pubmed_url`, `source`, and `has_full_text`. Without PMC full text, `body` is empty
and local search uses the abstract and keywords. Stored IR JSON includes the complete parsed
body, including section headings and table/caption text; source XML remains unchanged.
The command-line `scripts/build_index.py` also accepts these saved PubMed XMLs.

The lookup uses the official [NCBI EFetch API](https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch).
Optionally set `NCBI_EMAIL` to a contact email and `NCBI_API_KEY` to your API key
in the server environment. Lookup requests are spaced at least 0.35 seconds apart;
successful PubMed lookups are cached for one hour.

Lookup first reuses an identity-checked `PMID<id>.xml` already in `data/raw/`.
Select **Fetch latest from NCBI** to bypass local XML and the one-hour cache.
The result shows its source and elapsed lookup time. XML browser downloads use
the prepared bytes and do not repeat the NCBI request or rerun the page.

Transient timeouts, connection failures, HTTP 429 and selected 5xx errors get at
most two attempts. Single-record lookups use 5-second connect and 12-second read
timeouts; larger PMC/batch responses keep a 30-second read timeout. These are
socket inactivity limits, not a total wall-clock deadline. Permanent HTTP errors
fail immediately. Long `Retry-After` values are shown instead of making the UI
sleep. Optional local saves show separate timings for PMC retrieval, file/catalog
storage and index rebuilding; repeated full-text retrievals are cached for one hour.

Offline checks:

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

Re-run with another topic:

```bash
python scripts/download_sample.py --query "covid-19 vaccine" --n 120
python scripts/build_index.py
```

## Sample demo queries

| Query | What to show |
|---|---|
| `diabetes insulin treatment` | BM25-ranked list + highlights |
| `metformin type 2` | Compare Boolean AND vs OR |
| `glucose blood pressure` | Open a doc → stats + sentences |
| `inflammation cytokine` | Preprocessed stemmed terms in UI |
| `diabe ins` | Choose Partial / substring; open a result to inspect character/sentence locations |
| `diabtes` | Choose Fuzzy / edit distance (1 edit) for direct typo-tolerant retrieval |

Search results include **Open & locate → Keyword locations**, with every match's
section, sentence, character range, previous/next navigation, CSV and plaintext
exports. Offsets are zero-based `[start, end)` in the displayed normalized
abstract + keywords + body; the title is excluded.

On **Homework 2 · demo → 4. Word2Vec**, the **PCA 2D** map uses the saved model without
retraining. Explore a query's neighbors or frequent words, hover/click points,
zoom/pan, toggle labels and download coordinates. PCA is fitted to the complete
normalized vocabulary and reports the variance retained in two dimensions.

## Project layout

```
app.py                 # Helix homepage
pages/1_Search.py      # Search + document workspace
pages/2_PubMed.py      # Online PMID lookup + XML/JSON downloads
pages/3_Collection.py  # XML import, corpus statistics, preprocessing/index settings
pages/4_Methods.py     # System explanation, preprocessing trace, EOS playground
ui/brand.py            # Product theme / CSS
scripts/
  download_sample.py   # NCBI E-utilities PMC fetch
  build_index.py       # XML → docs.jsonl + index.pkl
src/
  parse_pmc.py         # PMC XML parser
  pubmed.py            # PubMed API client, metadata parsing, local exports
  corpus.py            # Validated single-article XML imports and corpus statistics
  preprocess.py        # tokenize / stopwords / Porter
  index.py             # inverted index
  search.py            # AND / OR / PHRASE + BM25
  eos.py               # rule-based sentence splitter
  stats.py             # document statistics
data/raw/              # PMC XML
data/processed/        # docs.jsonl, index.pkl
papers/                # 2 paper notes + slide outline
```

## Architecture

```
PMC XML → Parser → Document store
                ↓
         Preprocess (stop + Porter)
                ↓
         Inverted index + IDF
                ↓
Query → Preprocess → AND / OR / PHRASE → BM25 → Streamlit (results, stats, EOS)
```

## Optional background papers

1. [Porter (1980) — An algorithm for suffix stripping](papers/paper1_porter_stemming.md)
2. [Fiorini et al. (2018) — PubMed Best Match](papers/paper2_pubmed_best_match.md)

Slide outline: [papers/SLIDES.md](papers/SLIDES.md)

## Notes for grading / demo

- Matching is implemented from scratch (no Elasticsearch/Lucene) to reflect basic IR course schemes.
- Porter stemmer is the standard public-domain algorithm via NLTK.
- NCBI asks for a contact email in E-utilities; change `--email` in the download script if needed.
- The course submission uses 1,000 English GLP-1 abstracts; extra corpora have their own explicit sample sizes.
# Run on the NCKU local network

To make the app reachable from another device on the same NCKU network, run this
from PowerShell on the server machine:

```powershell
.\start_ncku_lan.ps1
```

Then open `http://SERVER-IP:8501` from another device. The server machine must
remain powered on. Allow inbound TCP port 8501 in Windows Firewall if it is not
already allowed.
