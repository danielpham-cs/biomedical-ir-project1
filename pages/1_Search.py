"""Helix Search — keyword retrieval workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import index_summary, load_docs, load_index
from src.preprocess import ensure_nltk_data, preprocess
from src.search import highlight, search
from src.stats import compute_stats
from ui.brand import BRAND, TAGLINE, inject_css

DOCS_PATH = ROOT / "data" / "processed" / "docs.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"

SAMPLE_QUERIES = [
    "diabetes insulin treatment",
    "glucose blood pressure",
    "metformin type 2",
    "inflammation cytokine",
]


@st.cache_resource
def load_resources():
    ensure_nltk_data()
    if not DOCS_PATH.exists() or not INDEX_PATH.exists():
        return None, None
    return load_docs(DOCS_PATH), load_index(INDEX_PATH)


def pubmed_url(pmid: str) -> str:
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return "https://pmc.ncbi.nlm.nih.gov/"


def _apply_sample_query(q: str) -> None:
    st.session_state.query = q
    st.session_state.view = "Search"


def _open_document(doc_id: str) -> None:
    st.session_state.selected_doc = doc_id
    st.session_state.view = "Document"


st.set_page_config(
    page_title=f"{BRAND} Search",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown(
    """
    <style>
      /* —— Streamlit chrome: no empty header strip —— */
      header[data-testid="stHeader"],
      .stAppHeader {
        background: transparent !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        border: none !important;
        z-index: 99990 !important;
      }
      .stAppToolbar,
      [data-testid="stToolbar"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
      }
      /* Remove Deploy + hamburger menu */
      .stAppDeployButton,
      [data-testid="stAppDeployButton"],
      #MainMenu,
      [data-testid="stMainMenu"],
      [data-testid="stToolbarActions"],
      .stToolbarActions {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
      }
      /* Sidebar open control = floating chip on the nav, not a header row */
      [data-testid="stExpandSidebarButton"],
      button[data-testid="stExpandSidebarButton"] {
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 100000 !important;
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1px solid #c5d5d1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(7, 20, 18, 0.14) !important;
        color: #071412 !important;
      }
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 100000 !important;
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1px solid #c5d5d1 !important;
        border-radius: 8px !important;
        padding: 0.2rem !important;
        box-shadow: 0 4px 14px rgba(7, 20, 18, 0.14) !important;
      }

      /* Full-width page canvas */
      section.main > div.block-container {
        max-width: 100% !important;
        padding: 0 !important;
      }

      /* Full-bleed nav iframe */
      div[data-testid="stElementContainer"]:has(iframe) {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      div[data-testid="stElementContainer"]:has(iframe) iframe {
        display: block !important;
        width: 100% !important;
      }

      /* Section 2 — centered white panel under full-width nav */
      .block-container > div > div > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid rgba(15, 118, 110, 0.18) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 24px rgba(7, 20, 18, 0.06) !important;
        padding: 0.5rem 0.35rem 0.75rem !important;
        margin: 1rem auto 1.5rem !important;
        max-width: 1100px !important;
        width: calc(100% - 2rem) !important;
      }
      div[data-testid="stVerticalBlockBorderWrapper"]
        div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: none !important;
        background: #fafcfb !important;
        border-radius: 8px !important;
        padding: 0 !important;
        margin: 0.35rem 0 !important;
        max-width: none !important;
        width: auto !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

docs, index = load_resources()
if docs is None or index is None:
    st.error(
        "Index not found. Run `python scripts/download_sample.py` then "
        "`python scripts/build_index.py`."
    )
    st.stop()

doc_by_id = {d.doc_id: d for d in docs}
summary = index_summary(index)

if "view" not in st.session_state:
    st.session_state.view = "Search"
if "query" not in st.session_state:
    st.session_state.query = "diabetes insulin"

with st.sidebar:
    st.markdown(f"### {BRAND}")
    st.caption("Corpus & matching")
    st.metric("Documents", summary["num_documents"])
    st.metric("Vocabulary", summary["vocabulary_size"])
    st.metric("Avg doc length", summary["avg_doc_length"])
    st.write(
        f"Stemming: **{'on' if index.stem else 'off'}** · "
        f"Stop-words: **{'removed' if index.remove_stopwords else 'kept'}**"
    )
    st.divider()
    st.radio("Workspace", ["Search", "Document"], key="view")
    st.divider()
    st.page_link("app.py", label="← Home")
    st.divider()
    st.markdown("**Try a query**")
    for i, q in enumerate(SAMPLE_QUERIES):
        st.button(
            q,
            key=f"sq_{i}",
            on_click=_apply_sample_query,
            args=(q,),
            use_container_width=True,
        )

# Section 1 — full-width top nav
components.html(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
    <div style="
      width: 100%;
      box-sizing: border-box;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px 20px;
      flex-wrap: wrap;
      padding: 18px 28px 18px 64px;
      margin: 0;
      border-radius: 0;
      background: linear-gradient(115deg, #061614 0%, #0d221f 48%, #0f766e 130%);
      border-bottom: 1px solid rgba(20, 184, 166, 0.22);
      font-family: 'Source Sans 3', sans-serif;
    ">
      <div style="
        font-family: Syne, sans-serif;
        font-weight: 800;
        font-size: 22px;
        letter-spacing: -0.03em;
        color: #f4f8f7;
        line-height: 1.1;
      ">{BRAND}</div>
      <div style="
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 6px 8px;
        font-size: 15px;
        color: rgba(244, 248, 247, 0.78);
        line-height: 1.35;
      ">
        <span style="font-family: Syne, sans-serif; font-weight: 600; color: #f4f8f7;">Search workspace</span>
        <span style="opacity: 0.45;">·</span>
        <span style="color: rgba(244, 248, 247, 0.58);">{TAGLINE}</span>
      </div>
    </div>
    """,
    height=64,
    scrolling=False,
)

# Section 2 — content panel (Streamlit native bordered container)
with st.container(border=True):
    if st.session_state.view == "Search":
        st.markdown("### Find papers")
        st.caption("Keyword retrieval over PMC open-access articles.")

        col_q, col_m, col_k = st.columns([3, 1, 1])
        with col_q:
            query = st.text_input("Keywords", key="query")
        with col_m:
            method = st.selectbox("Matching", ["TF-IDF", "Boolean"])
        with col_k:
            top_k = st.number_input("Top-k", min_value=5, max_value=50, value=10)

        boolean_mode = "AND"
        if method == "Boolean":
            boolean_mode = st.radio("Boolean mode", ["AND", "OR"], horizontal=True)

        hits = search(
            query,
            index,
            docs,
            method="boolean" if method == "Boolean" else "tfidf",
            boolean_mode=boolean_mode,
            top_k=int(top_k),
        )
        processed = preprocess(
            query,
            remove_stopwords=index.remove_stopwords,
            stem=index.stem,
        )
        st.write(
            f"**{len(hits)}** results · preprocessed terms: "
            f"`{'`, `'.join(processed) if processed else '(none)'}`"
        )

        if not hits:
            st.warning("No matching documents. Try OR mode or broader keywords.")
        else:
            for i, hit in enumerate(hits, 1):
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{i}. {hit.title}**")
                        link = pubmed_url(hit.pmid)
                        st.caption(
                            f"doc_id={hit.doc_id} · PMID [{hit.pmid or 'n/a'}]({link}) · "
                            f"matched: {', '.join(hit.matched_terms) or '—'}"
                        )
                        st.markdown(highlight(hit.snippet, query.split()))
                    with c2:
                        st.markdown("**Score**")
                        st.markdown(f"### {hit.score}")
                        st.button(
                            "Open",
                            key=f"open_{hit.doc_id}_{i}",
                            on_click=_open_document,
                            args=(hit.doc_id,),
                            use_container_width=True,
                            type="secondary",
                        )
    else:
        st.markdown("### Document")
        st.caption("Full text, statistics, and rule-based sentence segmentation.")

        options = {f"{d.doc_id} — {d.title[:80]}": d.doc_id for d in docs}
        default_id = st.session_state.get("selected_doc")
        keys = list(options.keys())
        default_idx = 0
        if default_id:
            for i, k in enumerate(keys):
                if options[k] == default_id:
                    default_idx = i
                    break

        choice = st.selectbox("Select document", keys, index=default_idx)
        doc_id = options[choice]
        st.session_state.selected_doc = doc_id
        doc = doc_by_id[doc_id]

        st.subheader(doc.title)
        meta_bits = []
        if doc.pmid:
            meta_bits.append(f"[PubMed {doc.pmid}]({pubmed_url(doc.pmid)})")
        if doc.pmcid:
            meta_bits.append(
                f"[PMC{doc.pmcid}](https://pmc.ncbi.nlm.nih.gov/articles/PMC{doc.pmcid}/)"
            )
        if doc.journal:
            meta_bits.append(doc.journal)
        if doc.year:
            meta_bits.append(doc.year)
        st.caption(" · ".join(meta_bits))

        stats = compute_stats(doc.full_text)
        a, b, c, d, e = st.columns(5)
        a.metric("Characters", stats.num_characters)
        b.metric("Words", stats.num_words)
        c.metric("Sentences (EOS)", stats.num_sentences)
        d.metric("Vocabulary", stats.vocabulary_size)
        e.metric("Avg sent. len", stats.avg_sentence_length)

        tab_abs, tab_body, tab_sent = st.tabs(
            ["Abstract", "Body / full text", "Sentences"]
        )
        with tab_abs:
            st.write(doc.abstract or "(no abstract)")
        with tab_body:
            st.write(doc.body or doc.full_text or "(empty)")
        with tab_sent:
            st.caption("Rule-based EOS (abbreviation-aware)")
            for i, s in enumerate(stats.sentences, 1):
                st.markdown(f"**{i}.** {s}")
