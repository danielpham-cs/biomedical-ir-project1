"""Shared product branding and CSS for Helix."""

from __future__ import annotations

import streamlit as st

BRAND = "Helix"
TAGLINE = "Biomedical literature search, built for clarity."

# Cool clinical palette — not purple, not cream+terracotta
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&display=swap');

:root {
  --hx-ink: #071412;
  --hx-ink-soft: #14302b;
  --hx-mist: #e7efed;
  --hx-paper: #f4f8f7;
  --hx-accent: #0f766e;
  --hx-accent-bright: #14b8a6;
  --hx-line: rgba(15, 118, 110, 0.22);
  --hx-muted: #5a736e;
  --font-display: "Syne", sans-serif;
  --font-body: "Source Sans 3", sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
}

.stApp {
  background: var(--hx-paper);
  color: var(--hx-ink);
}

/* Sidebar restored — light mist text on dark panel */
div[data-testid="stSidebar"] {
  background: #0d221f !important;
}
div[data-testid="stSidebarContent"] {
  background: #0d221f;
  color: var(--hx-mist);
}
div[data-testid="stSidebarContent"] p,
div[data-testid="stSidebarContent"] span,
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] .stMarkdown,
div[data-testid="stSidebarContent"] [data-testid="stMetricLabel"],
div[data-testid="stSidebarContent"] [data-testid="stWidgetLabel"],
div[data-testid="stSidebarContent"] [data-testid="stPageLink"] a {
  color: rgba(231, 239, 237, 0.92) !important;
}
div[data-testid="stSidebarContent"] [data-testid="stMetricValue"] {
  color: #e7efed !important;
}
div[data-testid="stSidebarContent"] .stButton > button {
  background: #ffffff !important;
  color: #071412 !important;
  border: 1px solid #c5d5d1 !important;
}
div[data-testid="stSidebarContent"] .stButton > button:hover {
  border-color: #14b8a6 !important;
}

#MainMenu, footer, .stAppDeployButton, [data-testid="stAppDeployButton"] {
  visibility: hidden !important;
  display: none !important;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebarNav"] { display: none !important; }

section.main > div {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  max-width: 100% !important;
}

/* —— Product top bar —— */
.hx-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: rgba(244, 248, 247, 0.92);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--hx-line);
  position: sticky;
  top: 0;
  z-index: 50;
  animation: hx-fade-down 0.7s ease both;
}
.hx-topbar a {
  text-decoration: none;
  color: var(--hx-ink);
  font-family: var(--font-body);
  font-weight: 600;
  font-size: 0.95rem;
}
.hx-topbar-brand {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 1.35rem;
  letter-spacing: -0.03em;
  color: var(--hx-ink);
}
.hx-topbar-nav {
  display: flex;
  gap: 1.25rem;
  align-items: center;
  flex-wrap: wrap;
}
.hx-topbar-cta {
  background: var(--hx-ink) !important;
  color: var(--hx-mist) !important;
  padding: 0.45rem 0.95rem !important;
  border-radius: 4px !important;
  transition: background 0.25s ease, transform 0.25s ease;
}
.hx-topbar-cta:hover {
  background: var(--hx-accent) !important;
  transform: translateY(-1px);
}

/* —— Homepage hero —— */
.hx-hero {
  position: relative;
  min-height: 100vh;
  min-height: 100dvh;
  display: grid;
  grid-template-columns: 1fr;
  overflow: hidden;
  background: var(--hx-ink);
}
.hx-hero-media {
  position: absolute;
  inset: 0;
  z-index: 0;
}
.hx-hero-media img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
  transform: scale(1.06);
  animation: hx-kenburns 18s ease-out both;
  filter: saturate(0.85) contrast(1.05);
}
.hx-hero-shade {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(105deg, rgba(7, 20, 18, 0.92) 0%, rgba(7, 20, 18, 0.72) 42%, rgba(7, 20, 18, 0.35) 100%),
    linear-gradient(to top, rgba(7, 20, 18, 0.55), transparent 45%);
  z-index: 1;
}
.hx-hero-content {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  min-height: 100vh;
  min-height: 100dvh;
  padding: clamp(1.5rem, 4vw, 3.5rem);
  padding-bottom: clamp(2.5rem, 8vh, 5rem);
  max-width: 920px;
  color: var(--hx-mist);
}
.hx-brand {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: clamp(3.4rem, 12vw, 7.5rem);
  line-height: 0.92;
  letter-spacing: -0.045em;
  margin: 0 0 1rem 0;
  animation: hx-rise 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
}
/* Compact top nav styles live page-local on Search; keep homepage-safe defaults */
.hx-brand-sm {
  font-size: clamp(2.2rem, 6vw, 3.6rem);
  margin-bottom: 0.65rem;
}
.hx-headline {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(1.35rem, 3.2vw, 2rem);
  line-height: 1.25;
  letter-spacing: -0.02em;
  margin: 0 0 0.75rem 0;
  max-width: 18ch;
  animation: hx-rise 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}
.hx-lede {
  font-size: clamp(1rem, 2vw, 1.15rem);
  line-height: 1.55;
  color: rgba(231, 239, 237, 0.82);
  margin: 0 0 1.75rem 0;
  max-width: 36ch;
  animation: hx-rise 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.22s both;
}
.hx-cta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  animation: hx-rise 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.32s both;
}
.hx-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-body);
  font-weight: 600;
  font-size: 1rem;
  padding: 0.85rem 1.35rem;
  border-radius: 4px;
  text-decoration: none !important;
  border: 1px solid transparent;
  transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
  cursor: pointer;
}
.hx-btn-primary {
  background: var(--hx-accent-bright);
  color: var(--hx-ink) !important;
}
.hx-btn-primary:hover {
  transform: translateY(-2px);
  background: #2dd4bf;
}
.hx-btn-ghost {
  background: transparent;
  color: var(--hx-mist) !important;
  border-color: rgba(231, 239, 237, 0.35);
}
.hx-btn-ghost:hover {
  border-color: var(--hx-mist);
  transform: translateY(-2px);
}

/* —— Below-fold sections (one job each) —— */
.hx-section {
  padding: clamp(3rem, 8vw, 5.5rem) clamp(1.5rem, 4vw, 3.5rem);
  max-width: 1100px;
  margin: 0 auto;
}
.hx-section h2 {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: clamp(1.6rem, 3.5vw, 2.35rem);
  letter-spacing: -0.03em;
  color: var(--hx-ink);
  margin: 0 0 0.6rem 0;
}
.hx-section p.lead {
  color: var(--hx-muted);
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 42ch;
  margin: 0 0 2rem 0;
}
.hx-split {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
}
@media (min-width: 800px) {
  .hx-split {
    grid-template-columns: 1fr 1fr;
    gap: 2.5rem;
    align-items: start;
  }
}
.hx-feature-block h3 {
  font-family: var(--font-display);
  font-size: 1.15rem;
  margin: 0 0 0.35rem 0;
  color: var(--hx-ink-soft);
}
.hx-feature-block p {
  margin: 0;
  color: var(--hx-muted);
  line-height: 1.55;
}
.hx-band {
  background:
    radial-gradient(ellipse at 20% 0%, rgba(20, 184, 166, 0.12), transparent 50%),
    linear-gradient(180deg, #dfeae7 0%, var(--hx-paper) 100%);
  border-top: 1px solid var(--hx-line);
  border-bottom: 1px solid var(--hx-line);
}
.hx-pipeline {
  display: grid;
  gap: 0.35rem;
  font-family: var(--font-display);
  font-weight: 600;
  font-size: clamp(1rem, 2.2vw, 1.25rem);
  color: var(--hx-ink-soft);
  letter-spacing: -0.02em;
}
.hx-pipeline span {
  display: block;
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--hx-line);
  animation: hx-rise 0.7s ease both;
}
.hx-pipeline span:nth-child(1) { animation-delay: 0.05s; }
.hx-pipeline span:nth-child(2) { animation-delay: 0.12s; }
.hx-pipeline span:nth-child(3) { animation-delay: 0.19s; }
.hx-pipeline span:nth-child(4) { animation-delay: 0.26s; }
.hx-footer {
  padding: 2rem clamp(1.5rem, 4vw, 3.5rem) 3rem;
  color: var(--hx-muted);
  font-size: 0.9rem;
  border-top: 1px solid var(--hx-line);
}
.hx-footer strong {
  font-family: var(--font-display);
  color: var(--hx-ink);
}

.hx-app-shell h1, .hx-app-shell h2, .hx-app-shell h3 {
  font-family: var(--font-display) !important;
  letter-spacing: -0.03em;
}

/* Force light white controls (override dark OS / Streamlit theme) */
section.main .stTextInput input,
section.main .stNumberInput input,
section.main [data-testid="stTextInput"] input,
section.main [data-testid="stNumberInput"] input,
section.main [data-baseweb="input"],
section.main [data-baseweb="input"] > div,
section.main [data-baseweb="input"] input,
section.main div[data-baseweb="select"] > div,
section.main [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
.hx-search-controls input,
.hx-search-controls [data-baseweb="select"] > div {
  background-color: #ffffff !important;
  background: #ffffff !important;
  color: #071412 !important;
  caret-color: #071412 !important;
  border: 1px solid #c5d5d1 !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  -webkit-text-fill-color: #071412 !important;
}
section.main [data-baseweb="select"] span,
section.main .stSelectbox span {
  color: #071412 !important;
}
section.main .stSelectbox svg,
section.main div[data-baseweb="select"] svg {
  fill: #071412 !important;
  color: #071412 !important;
}

/* Light secondary buttons (Open) — no black slab */
section.main .stButton > button[kind="secondary"],
section.main .stButton > button {
  background: #ffffff !important;
  color: #071412 !important;
  border: 1px solid #c5d5d1 !important;
}
section.main .stButton > button:hover {
  background: #e7efed !important;
  border-color: #0f766e !important;
}

div[data-testid="stMetricValue"] {
  font-family: var(--font-display);
  color: #071412 !important;
}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
  color: #5a736e !important;
}
section.main [data-testid="stMetricValue"],
section.main [data-testid="stMetricLabel"] p {
  color: #071412 !important;
}
section.main [data-testid="stMetricLabel"] {
  color: #5a736e !important;
}

@keyframes hx-rise {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes hx-fade-down {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes hx-kenburns {
  from { transform: scale(1.06); }
  to { transform: scale(1.0); }
}

@media (max-width: 640px) {
  .hx-topbar-nav a:not(.hx-topbar-cta) { display: none; }
}

/* CTAs sit on the dark hero footer band */
.hx-home-cta {
  margin-top: -5.5rem;
  margin-bottom: 0;
  padding: 0 clamp(1.5rem, 4vw, 3.5rem) 3rem;
  position: relative;
  z-index: 5;
  max-width: 920px;
  background: transparent;
}
.hx-home-cta [data-testid="stHorizontalBlock"] {
  background: transparent !important;
}
.hx-home-cta [data-testid="stPageLink"] a {
  background: var(--hx-accent-bright) !important;
  color: var(--hx-ink) !important;
  border: none !important;
  border-radius: 4px !important;
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  padding: 0.85rem 1.2rem !important;
  text-decoration: none !important;
  transition: transform 0.2s ease, background 0.2s ease;
  justify-content: center;
}
.hx-home-cta [data-testid="stLinkButton"] button,
.hx-home-cta [data-testid="stLinkButton"] a {
  background: transparent !important;
  color: #e7efed !important;
  border: 1px solid rgba(231, 239, 237, 0.45) !important;
  border-radius: 4px !important;
  font-weight: 600 !important;
}
.hx-home-cta [data-testid="stPageLink"] a:hover {
  background: #2dd4bf !important;
  transform: translateY(-2px);
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def topbar(*, active: str = "home") -> None:
    search_cls = "hx-topbar-cta" if active != "search" else "hx-topbar-cta"
    st.markdown(
        f"""
        <div class="hx-topbar">
          <a class="hx-topbar-brand" href="/" target="_self">{BRAND}</a>
          <nav class="hx-topbar-nav">
            <a href="/" target="_self">Home</a>
            <a href="./Search" target="_self">Product</a>
            <a class="{search_cls}" href="./Search" target="_self">Open search</a>
          </nav>
        </div>
        """,
        unsafe_allow_html=True,
    )
