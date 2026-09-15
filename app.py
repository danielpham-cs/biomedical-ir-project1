"""Helix — product homepage."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ui.brand import BRAND, TAGLINE, inject_css

HERO_IMG = (
    "https://images.unsplash.com/photo-1579684385127-1ef15d508118"
    "?auto=format&fit=crop&w=2000&q=80"
)


def main() -> None:
    st.set_page_config(
        page_title=f"{BRAND} — Biomedical literature search",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    # Homepage: collapse sidebar by default; keep reopen control visible
    st.markdown(
        """
        <style>
          [data-testid="stSidebarCollapsedControl"],
          [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 100000 !important;
            background: #ffffff !important;
            border: 1px solid #c5d5d1 !important;
            border-radius: 8px !important;
            padding: 0.35rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="hx-hero" aria-label="{BRAND} home">
          <div class="hx-hero-media" aria-hidden="true">
            <img src="{HERO_IMG}" alt="" />
            <div class="hx-hero-shade"></div>
          </div>
          <div class="hx-hero-content">
            <p class="hx-brand">{BRAND}</p>
            <h1 class="hx-headline">Find what the literature already knows.</h1>
            <p class="hx-lede">{TAGLINE} Keyword search over PubMed Central with ranking you can explain.</p>
            <div id="hx-hero-cta-slot"></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hx-home-cta">', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.2, 1.2, 2])
    with c1:
        st.page_link(
            "pages/1_Search.py",
            label="Start searching",
            use_container_width=True,
        )
    with c2:
        st.link_button("See how it works", "#how", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <section class="hx-section" id="how">
          <h2>How it works</h2>
          <p class="lead">A transparent retrieval stack — from PMC XML to ranked results — without a black-box index.</p>
          <div class="hx-pipeline">
            <span>01  Ingest open PMC articles as structured XML</span>
            <span>02  Normalize with stop-words and Porter stemming</span>
            <span>03  Match with Boolean sets or TF–IDF cosine ranking</span>
            <span>04  Inspect documents with stats and rule-based sentences</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="hx-band">
          <div class="hx-section">
            <h2>Built for demo-ready research search</h2>
            <p class="lead">One workspace for keyword retrieval, ranking comparison, and document-level reading statistics.</p>
            <div class="hx-split">
              <div class="hx-feature-block">
                <h3>Classical matching, visible scores</h3>
                <p>Switch Boolean AND/OR and TF–IDF side by side. Every hit shows a score and highlighted snippet — ready for classroom or lab demos.</p>
              </div>
              <div class="hx-feature-block">
                <h3>Document intelligence</h3>
                <p>Characters, words, vocabulary, and sentence counts from an abbreviation-aware EOS splitter tuned for biomedical prose.</p>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="hx-section">
          <h2>Open the product</h2>
          <p class="lead">Jump into the live search workspace over the indexed PMC corpus.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Search.py", label=f"Launch {BRAND} Search", use_container_width=True)

    st.markdown(
        f"""
        <footer class="hx-footer">
          <strong>{BRAND}</strong> · Biomedical information retrieval · PMC open access corpus
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
