# Helix — Biomedical IR Project #1

**Helix** is a product-style keyword search workspace over **PubMed Central (PMC)** XML documents: classical IR matching, ranked results, and document statistics.

Course: Biomedical Information Retrieval · Due: Sept 22, 2026

## Features

- Parse PMC XML → title, abstract, body, PMID/PMCID, journal, year
- Preprocessing: lowercase, tokenize, English **stop-words**, **Porter stemming** (NLTK / Porter 1980)
- **Inverted index** with term frequencies and positions
- Matching: **Boolean AND/OR** and **TF-IDF cosine** ranking
- Document statistics: characters, words, sentences, vocabulary
- Rule-based **EOS** sentence splitter (abbreviation-aware)
- Streamlit product UI: **Helix** homepage + search workspace with scores, highlights, document stats

## Quick start

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

Opens the **Helix** homepage. Use **Start searching** to enter the retrieval workspace.

## Deploy on Streamlit Community Cloud (via GitHub)

Repo: [danielpham-cs/biomedical-ir-project1](https://github.com/danielpham-cs/biomedical-ir-project1)

1. Push this project to GitHub (already configured as `origin`).
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Connect the GitHub repo `danielpham-cs/biomedical-ir-project1`.
4. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**.

The processed corpus (`data/processed/docs.jsonl` + `index.pkl`) is included so the cloud app can search without re-downloading PMC XML.

Re-run with another topic:

```bash
python scripts/download_sample.py --query "covid-19 vaccine" --n 120
python scripts/build_index.py
```

## Sample demo queries

| Query | What to show |
|---|---|
| `diabetes insulin treatment` | TF-IDF ranked list + highlights |
| `metformin type 2` | Compare Boolean AND vs OR |
| `glucose blood pressure` | Open a doc → stats + sentences |
| `inflammation cytokine` | Preprocessed stemmed terms in UI |

## Project layout

```
app.py                 # Helix homepage
pages/1_Search.py      # Search + document workspace
ui/brand.py            # Product theme / CSS
scripts/
  download_sample.py   # NCBI E-utilities PMC fetch
  build_index.py       # XML → docs.jsonl + index.pkl
src/
  parse_pmc.py         # PMC XML parser
  preprocess.py        # tokenize / stopwords / Porter
  index.py             # inverted index
  search.py            # Boolean + TF-IDF
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
Query → Preprocess → Boolean / TF-IDF → Streamlit (results, stats, EOS)
```

## Papers (presentation)

1. [Porter (1980) — An algorithm for suffix stripping](papers/paper1_porter_stemming.md)
2. [Fiorini et al. (2018) — PubMed Best Match](papers/paper2_pubmed_best_match.md)

Slide outline: [papers/SLIDES.md](papers/SLIDES.md)

## Notes for grading / demo

- Matching is implemented from scratch (no Elasticsearch/Lucene) to reflect basic IR course schemes.
- Porter stemmer is the standard public-domain algorithm via NLTK.
- NCBI asks for a contact email in E-utilities; change `--email` in the download script if needed.
- Corpus size is intentionally ~100–200 docs for a stable classroom demo.
