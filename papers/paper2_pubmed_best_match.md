# Paper 2 — PubMed / Biomedical Literature Retrieval

## Citation

Fiorini, N., Canese, K., Starchenko, G., et al. (2018). **Best Match: New relevance search for PubMed**. *PLOS Biology*, 16(8), e2005343.  
https://doi.org/10.1371/journal.pbio.2005343  
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005343

## Problem

PubMed historically defaulted to **recency-heavy** ranking. Users searching biomedical literature often need **relevance** (topical match to keywords), not only the newest papers. Scaling relevance ranking over tens of millions of citations is hard.

## Method

**Best Match** learns a relevance ranking model over PubMed (machine-learned ranking with many signals: term matching, past usage, publication type, etc.), and became a user-selectable / default relevance sort. It builds on classical term matching while adding learned signals.

## Results

Authors report improved retrieval quality versus date-sorted baselines in online evaluation with PubMed users — showing that keyword matching alone is necessary but ranking quality matters for biomedical search UX.

## Link to our system

Our demo focuses on the **classic IR core** that underpins tools like PubMed: inverted index, Boolean matching, and TF-IDF cosine ranking over PMC XML, plus visualization (snippets, scores, document stats). Best Match motivates *why* ranking beyond Boolean AND is useful — which we illustrate by comparing Boolean vs TF-IDF side-by-side in the Streamlit UI.
