# Biomedical IR Project #1 — Demo Slides Outline

Use this as a 5–8 slide talk script (copy into PowerPoint / Google Slides).

---

## Slide 1 — Title
**Keyword-based Full-Text Matching on PubMed/PMC Documents**  
Biomedical Information Retrieval — Project #1  
Your name · Due Sept 22, 2026

## Slide 2 — Problem & goals
- Build a **search engine** over PMC XML biomedical articles
- Keyword query → ranked documents + clear visualization
- Document statistics + **smart sentence (EOS)** splitting
- Classic IR preprocessing: stop-words + **Porter stemming**

## Slide 3 — System pipeline
PMC XML → Parse (title/abstract/body) → Preprocess → **Inverted index**  
Query → same preprocess → **AND/OR/PHRASE selection** → **BM25 ranking** → Streamlit UI

## Slide 4 — Matching schemes
- **Candidate selection:** AND intersects posting lists, OR unions them, and PHRASE requires adjacent stemmed terms in one paragraph
- **BM25:** combines term frequency, inverse document frequency, and document-length normalization for ranking
- Show one example query and top-3 titles + scores

## Slide 5 — Document stats & EOS
- Characters, words, sentences, vocabulary, avg sentence length
- Rule-based EOS: do not split on `Dr.`, `Fig.`, `et al.`, `i.e.`, decimals `1.2`

## Slide 6 — Live demo checklist
1. Query: `diabetes insulin treatment` (BM25)
2. Same query with AND vs OR, then PHRASE
3. Open a hit → show stats + sentence list
4. Point to preprocessing terms in the UI

## Slide 7 — Two papers
1. Porter (1980) — suffix stripping / stemming  
2. Fiorini et al. (2018) — PubMed Best Match relevance ranking  
(See `papers/` for one-page notes.)

## Slide 8 — Summary & future projects
- Delivered: keyword IR + PMC visualization + stats/EOS  
- Next course projects: query expansion, indexing models, multi-modal, LLMs
