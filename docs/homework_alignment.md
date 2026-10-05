# Project #1: implementation aligned with the homework brief

For Project #2 (Zipf, local Porter comparison, Word2Vec and spelling correction),
see [the Homework 2 system description](HW2.md) and the **Homework 2 · demo** pages.

Source: `homework-1-26-20260908.pdf`, one page, due September 22, 2026.

The brief specifies individual work, permits ChatGPT/Claude/Gemini, and evaluates
system performance and the individual demonstration. It does not specify a visual
theme, corpus size, a required ranking formula, or a two-paper presentation.

| Actual requirement | Workspace behavior | Implementation |
|---|---|---|
| Biomedical documents from PubMed/PMC in XML | Import one article XML per file, retrieve records by PMID/PMCID, inspect actual full-text coverage | `pages/3_Collection.py`, `pages/2_PubMed.py`, `src/corpus.py`, `src/pubmed.py` |
| Full-text retrieval using specified keywords | Index abstract, Keywords, and complete parsed body; the title is excluded and numeric terms such as type 2 are retained | `src/parse_pmc.py`, `src/preprocess.py`, `src/index.py` |
| Basic IR matching schemes | AND/OR/PHRASE candidate selection followed by BM25 ranking | `src/search.py`, `pages/1_Search.py` |
| Easily visualized results | Ranked cards with snippets, scores, pagination, matching time, and document inspection | `pages/1_Search.py` |
| Document statistics, including characters, words, and sentences | Corpus table/length chart, document metrics, section counts, frequent words, and statistics export | `src/stats.py`, `src/corpus.py`, collection and reader pages |
| Smart EOS, for example rule-based | Sentence list and an editable playground demonstrating abbreviations, initials, decimals, and closing quotes | `src/eos.py`, `pages/4_Methods.py` |
| Public-domain preprocessing algorithms may be used | NLTK English stopwords and Porter stemmer; trace each step and choose settings before indexing | `src/preprocess.py`, collection and methods pages |
| Explain the system, show running results, and demonstrate it | Actual dataset metrics, methods/formulas, and a demo sequence linked to the working features | `app.py`, `pages/4_Methods.py` |

## Scope and interpretation

- Full text means text extracted from the XML. Non-text figures are not indexed.
- PubMed-only records can contain just a title/abstract. They are explicitly labeled;
  use PMC full-text XML for the full-text portion of the course demonstration.
- Long article bodies are no longer truncated. Rebuild old indexes from their XML
  to apply the complete-text parser and updated tokenization.
- Words are lowercase alphanumeric/hyphenated tokens. Document vocabulary is counted
  before stemming; index vocabulary is counted after the selected preprocessing.
- EOS is heuristic. Abbreviations at sentence ends and unusual notation remain
  ambiguous; the UI makes this limitation visible rather than claiming perfect accuracy.
- Query timing measures the local matching call, not network access or browser rendering.
- Reference paper notes in `papers/` are optional background, not a PDF requirement.
