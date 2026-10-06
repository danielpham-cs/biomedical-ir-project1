"""Homework 2 demo pages in presentation order, each mapped to the brief's requirements.

Two course documents define the work:
- "Brief"  = Homework-2-26-20260922.pdf (Zipf, own Porter, Word2Vec, edit distance, spelling, demo)
- "Spec"   = project-2-details-20260929.pdf (Parts I–IX, RQ1–RQ5, optional domains, summary)

The demo is five minutes: each page stacks one card per question for its minute;
collapsed "More details" and "Answers" blocks at the bottom serve follow-up questions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    key: str
    file: str
    title: str
    question: str
    minutes: str
    requirements: tuple[tuple[str, str, str], ...]  # (document, reference, requirement)
    script: tuple[str, ...] = ()  # what to say during the page's minute

    @property
    def url_path(self) -> str:
        # Streamlit drops the numeric prefix: pages/11_HW2_Data.py -> /HW2_Data
        return self.file.split("/")[-1].removesuffix(".py").split("_", 1)[1]


PAGES = (
    Page("overview", "pages/10_HW2_Overview.py", "Overview & results",
         "What was required, what did the experiment show, and what is submitted?", "0:00–0:30 · 4:45–5:00",
         (("Brief", "Demo", "System description, running results and system demo"),
          ("Spec", "Summary", "One-page executive summary, Finding 1–4"),
          ("Spec", "Part IX", "300–500 words: why Zipf matters to IR"),
          ("Spec", "RQ1–RQ5", "Research questions and final challenge")),
         ("Open: 1,000 GLP-1 abstracts; five pages follow the two briefs.",
          "Close: return here and read Finding 1–4.")),
    Page("data", "pages/11_HW2_Data.py", "Data & preprocessing",
         "What are the documents, and what does each preprocessing step — including our Porter — do?", "0:30–1:30",
         (("Spec", "Dataset", "About 1,000 English abstracts with unique PMIDs"),
          ("Spec", "Part I", "Tokenize, case fold, punctuation, stopwords kept/removed, stemming"),
          ("Spec", "Part II", "Documents, tokens, unique terms, average tokens per document"),
          ("Brief", "Porter", "Own Porter implementation, compared before/after"),
          ("Spec", "Part VI", "Effect of conditions A–D")),
         ("Table: 1,000 docs; vocabulary shrinks A → D.",
          "Sandbox: one sentence through steps 1–5.",
          "Porter: generalizations → gener, rule by rule; 100% agreement with the reference.")),
    Page("zipf", "pages/12_HW2_Zipf.py", "Zipf's law",
         "Does the collection follow Zipf's Law, and with which exponent?", "1:30–2:30",
         (("Brief", "Zipf", "Zipf distribution / frequency spectrum"),
          ("Spec", "Part III", "CF, Top 50 terms"),
          ("Spec", "Part V", "Rank–frequency, log-log regression: slope, intercept, k, R², RMSE; Q1–Q4"),
          ("Spec", "Optional", "Two-domain comparison")),
         ("Log-log is almost straight: k and R² on top.",
          "Linear plot: a few terms hold most tokens.",
          "R² is not proof (Q3); the tail is a staircase of ties.",
          "A–D table: punctuation −33% vocabulary, stopwords −27% tokens, stemming −19% vocabulary; k 1.09–1.27.")),
    Page("terms", "pages/13_HW2_CF_DF_IDF.py", "CF · DF · IDF",
         "How do collection frequency, document frequency and IDF differ?", "2:30–3:15",
         (("Spec", "Part IV", "DF for every term"),
          ("Spec", "Part VII", "CF vs DF for ≥20 terms; Q1–Q3"),
          ("Spec", "Part VIII", "IDF = log(N/DF) for ≥10 terms; relation to TF-IDF")),
         ("'and': DF ≈ N, so IDF ≈ 0.",
          "A concentrated term: high CF, DF of 1, maximum IDF.")),
    Page("word2vec", "pages/14_HW2_Word2Vec.py", "Word2Vec · Skip-gram",
         "What does Skip-gram learn from the context of each word?", "3:15–4:00",
         (("Brief", "Word2Vec", "Skip-gram on same-subject PubMed documents"),),
         ("A center word predicts its window: these are the training pairs.",
          "Neighbours of glp-1 and the PCA map.")),
    Page("search", "pages/15_HW2_Search.py", "Edit distance & search",
         "How are misspelled or partial queries matched, and where do they occur?", "4:00–4:45",
         (("Brief", "Matching", "Exact / partial matching with DP edit distance"),
          ("Brief", "Spelling", "Spelling correction is necessary"),
          ("Brief", "Locations", "Location(s) of query keywords in each document")),
         ("DP table: livr → liver = 1 insertion (also doze → dose, weigth → weight).",
          "Apply the suggestion, open a hit, step through its match locations.")),
)
BY_KEY = {page.key: page for page in PAGES}
