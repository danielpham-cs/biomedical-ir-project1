"""Shared presentation components for the Helix Streamlit application."""

import re
from html import escape
from pathlib import Path

import streamlit as st

from src.markup import render_inline_markup

from .i18n import dark_mode, preferences, t

BRAND = "Helix"
TAGLINE = "A clearer view of biomedical literature."


def inject_css() -> None:
    """Add the project stylesheet to the current Streamlit page."""
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if dark_mode():
        st.markdown(
            """<style>
            .stApp { background: #10191e !important; color: #e8f0f2 !important; }
            header[data-testid="stHeader"] { background: #10191ef2 !important; }
            [data-testid="stSidebar"] { background: #17242b !important; border-color: #31464f !important; }
            h1, h2, h3, h4, p, label, [data-testid="stMetricValue"] { color: #e8f0f2 !important; }
            [data-testid="stMetricLabel"], .stCaption, .hx-masthead-note { color: #aabdc4 !important; }
            .st-key-home_search, .st-key-search_box, .st-key-primary_action, .hx-algorithm-card, .hx-pubmed-abstract, .st-key-overview_metrics { background: #17242b !important; border-color: #31464f !important; box-shadow: none !important; }
            .st-key-secondary_action, .hx-pubmed-abstract-label { background: #20323a !important; border-color: #31464f !important; }
            .hx-product-hero aside { background: #087f78 !important; }
            .hx-product-hero { background: linear-gradient(145deg, #17242b, #20323a) !important; border-color: #31464f !important; box-shadow: none !important; }
            .hx-product-hero h1 { color: #e8f0f2 !important; }.hx-product-hero::before { border-color: #3d6264 !important; }
            .hx-product-hero p, .hx-hero-proof p, .hx-how-caption, .hx-algorithm-card p, .st-key-primary_action p, .st-key-secondary_action p, .hx-pubmed-abstract-body p, .hx-reading { color: #c9d8dc !important; }
            [class*="st-key-how_card_"] [data-testid="stButton"] > button { background: #17242b !important; border-color: #31464f !important; box-shadow: none !important; }
            [class*="st-key-how_card_"] [data-testid="stButton"] > button:hover { background: #17242b !important; border-color: #50cfc0 !important; box-shadow: 0 10px 24px #087f7833 !important; }
            [class*="st-key-how_card_"] [data-testid="stMarkdownContainer"] p { color: #c9d8dc !important; }
            [class*="st-key-how_card_"] [data-testid="stMarkdownContainer"] p::first-line { color: #71d6c9 !important; }
            [class*="st-key-how_card_"] [data-testid="stMarkdownContainer"] strong { color: #e8f0f2 !important; }
            .hx-hero-proof span { box-shadow: 0 0 0 5px #17a58e33 !important; }.hx-overview-heading h2 { color: #e8f0f2 !important; }
            .hx-article-title a, .hx-result-title a { color: #e8f0f2 !important; }.hx-article-title a:hover, .hx-result-title a:hover { color: #71d6c9 !important; }
            .st-key-primary_action [data-testid="stPageLink"] a, .st-key-secondary_action [data-testid="stPageLink"] a { background: #087f78 !important; color: #ffffff !important; }
            .st-key-primary_action [data-testid="stPageLink"] a:hover, .st-key-secondary_action [data-testid="stPageLink"] a:hover { background: #0a958b !important; color: #ffffff !important; }
            [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button, [data-testid="stFormSubmitButton"] > button { background: #087f78 !important; border-color: #087f78 !important; color: #ffffff !important; }
            [data-testid="stButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #0a958b !important; border-color: #0a958b !important; color: #ffffff !important; }
            [data-testid="stButton"] > button:disabled, [data-testid="stDownloadButton"] > button:disabled, [data-testid="stFormSubmitButton"] > button:disabled { background: #31545a !important; border-color: #31545a !important; color: #aabdc4 !important; }
            [data-testid="stSidebar"] [data-testid="stPageLink"] a { color: #e8f0f2 !important; }
            /* Streamlit gives page links a pale background on hover.  Keep the sidebar
               surface stable in dark mode and communicate focus with the teal outline. */
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { background: transparent !important; border: 1px solid transparent !important; box-sizing: border-box !important; }
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover,
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:focus-visible { background: transparent !important; border-color: #087f78 !important; color: #71d6c9 !important; }
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover span,
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:focus-visible span { color: #71d6c9 !important; }
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:has(strong) { background: transparent !important; border-color: #087f78 !important; color: #71d6c9 !important; }
            [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:has(strong) span, [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:has(strong) strong { color: #71d6c9 !important; }
            /* The sidebar toggle lives in Streamlit's header, outside stButton. */
            [data-testid="stExpandSidebarButton"],
            [data-testid="stSidebarCollapseButton"] button { background: #087f78 !important; border: 1px solid #50cfc0 !important; border-radius: 8px !important; color: #ffffff !important; min-width: 2rem !important; min-height: 2rem !important; padding: 0.35rem !important; box-shadow: 0 0 0 1px #087f7855 !important; }
            [data-testid="stExpandSidebarButton"]:hover,
            [data-testid="stSidebarCollapseButton"] button:hover { background: #0a958b !important; border-color: #8ae8da !important; color: #ffffff !important; opacity: 1 !important; }
            [data-testid="stExpandSidebarButton"] svg,
            [data-testid="stExpandSidebarButton"] svg *,
            [data-testid="stSidebarCollapseButton"] button svg,
            [data-testid="stSidebarCollapseButton"] button svg * { color: #ffffff !important; fill: #ffffff !important; opacity: 1 !important; }
            [data-testid="stDataFrame"], [data-testid="stDataFrame"] * { --gdg-bg-cell: #17242b !important; --gdg-bg-cell-medium: #17242b !important; --gdg-bg-header: #20323a !important; --gdg-bg-header-has-focus: #29424b !important; --gdg-bg-header-hovered: #29424b !important; --gdg-bg-bubble: #29424b !important; --gdg-text-dark: #e8f0f2 !important; --gdg-text-medium: #c9d8dc !important; --gdg-text-light: #8fa8b0 !important; --gdg-text-header: #c9d8dc !important; --gdg-text-group-header: #c9d8dc !important; --gdg-border-color: #31464f !important; --gdg-horizontal-border-color: #31464f !important; --gdg-link-color: #71d6c9 !important; }
            [data-testid="stFileUploaderDropzone"] { background: #17242b !important; border-color: #3a5963 !important; }
            [data-testid="stFileUploaderDropzone"] button { background: #087f78 !important; border-color: #087f78 !important; color: #ffffff !important; }
            [data-testid="stFileUploaderDropzone"] button:hover { background: #0a958b !important; border-color: #0a958b !important; color: #ffffff !important; }
            [data-testid="stFileUploaderDropzoneInstructions"] span { color: #aabdc4 !important; }
            .hx-batch-table { background: #17242b !important; border-color: #31464f !important; }.hx-batch-table th { color: #c9d8dc !important; background: #20323a !important; border-color: #31464f !important; }.hx-batch-table td { color: #e8f0f2 !important; border-color: #29424b !important; }.hx-batch-table tbody tr:hover td { background: #20323a !important; }.hx-batch-title a { color: #71d6c9 !important; }
            [data-testid="stExpander"] { background: #17242b !important; border-color: #29424b !important; }
            [data-testid="stExpander"] details { background: #17242b !important; border-color: #29424b !important; }
            [data-testid="stExpander"] summary { background: #20323a !important; border-color: #31464f !important; }
            [data-testid="stExpander"] summary, [data-testid="stExpander"] summary span { color: #e8f0f2 !important; }
            [data-testid="stExpander"] summary svg { color: #71d6c9 !important; fill: #71d6c9 !important; opacity: 1 !important; }
            [data-testid="stDialog"] [role="dialog"], [data-testid="stDialog"] > div { background: #17242b !important; border-color: #31464f !important; }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"], [data-testid="stDialog"] [data-testid="stMarkdownContainer"] p, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] li, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] ul, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] ol { color: #c9d8dc !important; }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"] h1, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] h2, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] h3, [data-testid="stDialog"] [data-testid="stMarkdownContainer"] strong { color: #e8f0f2 !important; }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"] li::marker { color: #71d6c9 !important; }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"] code { background: #20323a !important; border: 1px solid #3a5963 !important; color: #8ae8da !important; }
            [data-testid="stDialog"] [data-testid="stMarkdownContainer"] pre { background: #10191e !important; border-color: #31464f !important; color: #e8f0f2 !important; }
            [data-testid="stDialog"] .katex, [data-testid="stDialog"] .katex * { color: #e8f0f2 !important; }
            [data-testid="stDialog"] button { background: #20323a !important; border-color: #3a5963 !important; color: #71d6c9 !important; opacity: 1 !important; }
            [data-testid="stDialog"] button:hover { background: #087f78 !important; border-color: #50cfc0 !important; color: #ffffff !important; }
            [data-testid="stDialog"] button svg, [data-testid="stDialog"] button svg * { color: currentColor !important; fill: currentColor !important; opacity: 1 !important; }
            .hx-pubmed-abstract-body, .hx-flow { background: #17242b !important; }
            [data-testid="stTextInput"] input, [data-baseweb="select"] > div { background: #10191e !important; color: #e8f0f2 !important; }
            </style>""",
            unsafe_allow_html=True,
        )


def masthead(section: str = "Biomedical literature") -> None:
    st.markdown(
        '<div class="hx-masthead"><a class="hx-wordmark" href="/" target="_self" aria-label="Go to Helix homepage">'
        '<span class="hx-logo" aria-hidden="true">🔬</span>helix'
        '</a><span class="hx-masthead-note">'
        f'{escape(section)}</span></div>',
        unsafe_allow_html=True,
    )


def intro(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="hx-intro"><p class="hx-eyebrow">{escape(kicker)}</p>'
        f'<h1>{escape(title)}</h1><p class="hx-lead">{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def chips(terms: list[str]) -> None:
    content = "".join(f'<span class="hx-chip">{escape(term)}</span>' for term in terms)
    st.markdown(f'<div class="hx-chips">{content}</div>', unsafe_allow_html=True)


def empty_state(title: str, description: str) -> None:
    st.markdown(
        '<div class="hx-empty"><span class="hx-empty-symbol" aria-hidden="true">+</span>'
        f'<h2>{escape(title)}</h2><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def reading_text(text: str) -> None:
    paragraphs = "".join(
        f"<p>{render_inline_markup(paragraph)}</p>" for paragraph in text.split("\n\n") if paragraph.strip()
    )
    st.markdown(f'<div class="hx-reading">{paragraphs}</div>', unsafe_allow_html=True)


def pubmed_abstract(text: str) -> None:
    """Render a structured abstract in a compact PubMed-style reading card."""
    blocks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        section = re.fullmatch(r"([A-Za-z][A-Za-z /-]{1,50}):\s*(.+)", paragraph, re.DOTALL)
        if section:
            blocks.append(
                f"<h3>{escape(section.group(1).upper())}</h3><p>{render_inline_markup(section.group(2))}</p>"
            )
        else:
            blocks.append(f"<p>{render_inline_markup(paragraph)}</p>")
    st.markdown(
        '<section class="hx-pubmed-abstract"><div class="hx-pubmed-abstract-label">Abstract</div>'
        f'<div class="hx-pubmed-abstract-body">{"".join(blocks)}</div></section>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="hx-footer"><span><strong>helix</strong> / Biomedical Information Retrieval / Project 1</span>'
        '<span>XML / Keyword matching / Document statistics / EOS</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_nav() -> None:
    """Render the navigation shared by all pages."""
    with st.sidebar:
        st.markdown("### helix / IR workspace")
        st.caption("LOCAL BIOMEDICAL LIBRARY")
        st.page_link("app.py", label=t("overview"))
        st.page_link("pages/2_PubMed.py", label=f"1. {t('abstract_lookup')}")
        st.page_link("pages/1_Search.py", label=f"2. {t('local_search')}")
        with st.expander(t("advanced")):
            st.page_link("pages/3_Collection.py", label=t("collection"))
            st.page_link("pages/4_Methods.py", label=t("methods"))
        st.divider()
        st.markdown(f"#### {t('settings')}")
        preferences()


def preprocessing_table(text: str, *, remove_stopwords: bool, stem: bool) -> None:
    """Show each configured preprocessing stage."""
    from src.preprocess import preprocess, tokenize

    tokens = tokenize(text)
    filtered = preprocess(text, remove_stopwords=remove_stopwords, stem=False)
    final = preprocess(text, remove_stopwords=remove_stopwords, stem=stem)
    rows = [
        {
            "Stage": stage,
            "Token count": len(values),
            "Preview (first 35 tokens)": " ".join(values[:35]),
        }
        for stage, values in [
            ("Lowercase and tokenize", tokens),
            ("Stopword filtering" if remove_stopwords else "Keep stopwords", filtered),
            ("Porter stemming" if stem else "Keep word forms", final),
        ]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
