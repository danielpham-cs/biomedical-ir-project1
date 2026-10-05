"""Helix product homepage for biomedical abstract retrieval and analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.index import load_docs, load_index
from src.search import build_spelling_vocabulary
from ui.brand import BRAND, footer, inject_css, masthead, sidebar_nav
from ui.ghost_search import ghost_search
from ui.i18n import algorithm_cards, dark_mode, language, t
from ui.method_details import method_details

ROOT = Path(__file__).resolve().parent


def open_local_search(query: str = "") -> None:
    st.session_state.home_query = query.strip()
    st.switch_page("pages/1_Search.py")


def autocomplete_vocabulary(docs) -> list[str]:
    vocabulary = build_spelling_vocabulary(docs)
    return [word for word, _ in sorted(vocabulary.items(), key=lambda item: (-item[1], len(item[0]), item[0]))]


def load_collection():
    docs_path = ROOT / "data" / "processed" / "docs.jsonl"
    index_path = ROOT / "data" / "processed" / "index.pkl"
    if not docs_path.exists() or not index_path.exists():
        return [], None
    try:
        return load_docs(docs_path), load_index(index_path)
    except (OSError, ValueError, EOFError):
        return [], None


def show_method_detail(step: str, title: str, detail: str) -> None:
    """Show one report-ready method explanation without leaving the overview."""

    @st.dialog(f"{step} · {title}", width="large")
    def detail_dialog() -> None:
        st.markdown(detail)

    detail_dialog()


def main() -> None:
    st.set_page_config(page_title=f"{BRAND} | Biomedical search", page_icon="🔬", layout="wide")
    inject_css()
    sidebar_nav()
    masthead("Biomedical evidence search")
    docs, index = load_collection()
    raw_count = sum((ROOT / "data/raw" / f"PMID{doc.pmid}.xml").exists() for doc in docs)
    abstract_only = sum(not bool(document.body) for document in docs)
    vocabulary_size = index.vocabulary_size() if index else 0

    st.markdown(
        f'''<section class="hx-product-hero">
        <div class="hx-hero-copy"><p class="hx-product-kicker">{t('hero_kicker')}</p>
        <h1>{t('hero_title')}</h1>
        <p>{t('hero_body')}</p>
        <div class="hx-hero-proof"><span></span><p>{t('ready_body')}</p></div></div>
        <aside><span>{t('ready')}</span><strong>{len(docs):,}</strong><p>{t('indexed_records')} · {vocabulary_size:,} {t('vocabulary')}</p><div class="hx-hero-signal" aria-hidden="true"><i></i><i></i><i></i><i></i></div></aside>
        </section>''',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="hx-overview-heading"><span>01</span><h2>{t("search_collection")}</h2></div>', unsafe_allow_html=True)
    with st.container(key="home_search"):
        if "home_search_query" not in st.session_state:
            st.session_state.home_search_query = ""
        event = ghost_search(
            value=st.session_state.home_search_query,
            vocabulary=autocomplete_vocabulary(docs),
            label="",
            placeholder="e.g. diabetes insulin treatment",
            submit_label=t("search_button"),
            dark=dark_mode(),
            key="home_search_ghost",
        )
        if isinstance(event, dict) and event.get("nonce") != st.session_state.get("last_home_search_nonce"):
            st.session_state.last_home_search_nonce = event.get("nonce")
            query = str(event.get("value", "")).strip()
            st.session_state.home_search_query = query
            if query:
                open_local_search(query)
        st.caption(t("search_hint"))

    with st.container(key="overview_metrics"):
        for column, label, value, note in zip(
            st.columns(4),
            [t("abstract_xml"), t("indexed_records"), t("abstract_only"), t("vocabulary")],
            [raw_count, len(docs), abstract_only, vocabulary_size],
            ["local source files", "ready for BM25", "no full text needed", "stemmed search terms"],
        ):
            with column:
                st.metric(label, f"{value:,}")
                st.caption(note)

    st.markdown(f'<div class="hx-overview-heading"><span>02</span><h2>{t("start_task")}</h2></div>', unsafe_allow_html=True)
    primary, secondary = st.columns([1.45, 1])
    with primary:
        with st.container(key="primary_action"):
            st.markdown(
                f'<span class="hx-action-number">01 / PMID · PMCID</span><h3>{escape(t("lookup_title"))}</h3><p>{escape(t("lookup_body"))}</p>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/2_PubMed.py", label=t("lookup_action"))
    with secondary:
        with st.container(key="secondary_action"):
            st.markdown(
                f'<span class="hx-action-number">02 / BM25</span><h3>{escape(t("library_title"))}</h3><p>{escape(t("library_body"))}</p>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/1_Search.py", label=t("library_action"))

    with st.container(border=True):
        st.markdown("### Homework 2 · GLP-1 experiments")
        st.write("Ten demo pages, one per requirement: corpus, preprocessing A–D, Porter, Zipf, CF/DF/IDF, Word2Vec, edit distance & search, two domains and the report.")
        st.page_link("pages/10_HW2_Overview.py", label="Open the Homework 2 demo →")

    st.markdown(f'<div class="hx-overview-heading hx-how-heading"><span>03</span><h2>{t("how")}</h2></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="hx-how-caption">{t("how_caption")}</p>', unsafe_allow_html=True)
    cards = algorithm_cards()
    details = method_details(language())
    selected_card: int | None = None
    for row_start in range(0, len(cards), 3):
        for card_index, column in enumerate(st.columns(3), start=row_start):
            if card_index >= len(cards):
                break
            step, title, description = cards[card_index]
            with column:
                with st.container(key=f"how_card_{card_index}"):
                    if st.button(
                        f"{step}\n\n**{title}**\n\n{description}",
                        key=f"open_how_card_{card_index}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        selected_card = card_index

    if selected_card is not None:
        step, title, _ = cards[selected_card]
        show_method_detail(step, title, details[selected_card])

    with st.expander(t("technical"), expanded=False):
        st.markdown(
            "**Statistics scope:** abstract + Keywords. **Search scope:** abstract + Keywords + available body, excluding the title. "
            "**Ranking:** BM25 with k1 = 1.2 and b = 0.75. **Sentence detection:** rule-based EOS."
        )
    footer()


if __name__ == "__main__":
    main()
