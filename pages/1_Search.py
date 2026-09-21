"""Local abstract/body search using the final AND, OR, PHRASE and BM25 rules."""

from __future__ import annotations

import sys
from collections import Counter
from html import escape
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import index_summary, load_docs, load_index
from src.preprocess import preprocess, tokenize
from src.search import build_spelling_vocabulary, highlight_text, search, suggest_query_correction
from src.stats import compute_stats
from ui.brand import (
    BRAND,
    chips,
    empty_state,
    footer,
    inject_css,
    intro,
    masthead,
    pubmed_abstract,
    reading_text,
    sidebar_nav,
)
from ui.ghost_search import ghost_search
from ui.i18n import dark_mode, t

DOCS_PATH = ROOT / "data" / "processed" / "docs.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"


@st.cache_resource
def load_resources(docs_version: int, index_version: int):
    docs = load_docs(DOCS_PATH)
    return docs, load_index(INDEX_PATH), build_spelling_vocabulary(docs)


def open_document(doc_id: str) -> None:
    st.session_state.reader_doc = doc_id


def close_document() -> None:
    st.session_state.pop("reader_doc", None)


def autocomplete_vocabulary(vocabulary: dict[str, int]) -> list[str]:
    """Most frequent English corpus terms are preferred for ghost completion."""
    return [word for word, _ in sorted(vocabulary.items(), key=lambda item: (-item[1], len(item[0]), item[0]))]


def render_document(document) -> None:
    st.button("Back to results", on_click=close_document)
    intro("Document analysis", document.title, "Abstract statistics follow the final course specification.")
    abstract, search_scope, statistics, sentences = st.tabs(
        ["Abstract", "Search scope", "Statistics", "Sentences and EOS"]
    )
    abstract_text = document.abstract
    stats = compute_stats(abstract_text)
    with abstract:
        if abstract_text:
            pubmed_abstract(abstract_text)
        else:
            st.info("This record has no abstract text.")
    with search_scope:
        st.caption("Search scope is abstract + keywords + body. The title is excluded.")
        reading_text(document.search_text)
    with statistics:
        st.caption("Statistics cover the abstract and its final Keywords line. Characters use normalized whitespace.")
        for column, label, value in zip(
            st.columns(5),
            ["Characters", "No-space chars", "Words", "Sentences", "Unique words"],
            [stats.num_characters, stats.characters_no_space, stats.num_words, stats.num_sentences, stats.vocabulary_size],
        ):
            column.metric(label, f"{value:,}")
        counts = Counter(tokenize(abstract_text)).most_common(20)
        if counts:
            st.dataframe(
                [{"Term": term, "Occurrences": count} for term, count in counts],
                hide_index=True,
                use_container_width=True,
            )
    with sentences:
        if stats.sentences:
            st.dataframe(
                [{"Sentence": number, "Text": text} for number, text in enumerate(stats.sentences, 1)],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No sentences are available.")


def main() -> None:
    st.set_page_config(page_title=f"{BRAND} | Local search", page_icon="🔬", layout="wide")
    inject_css()
    sidebar_nav()
    masthead(t("search_page_title"))

    if not DOCS_PATH.exists() or not INDEX_PATH.exists():
        intro(t("local_search"), t("no_saved"), t("lookup_page_body"))
        st.page_link("pages/2_PubMed.py", label=t("lookup_action"))
        footer()
        return

    docs, index, spelling_vocabulary = load_resources(DOCS_PATH.stat().st_mtime_ns, INDEX_PATH.stat().st_mtime_ns)
    if not docs:
        empty_state("No readable documents", "Rebuild the local index after saving XML files.")
        footer()
        return
    by_id = {document.doc_id: document for document in docs}
    summary = index_summary(index)

    with st.sidebar:
        st.divider()
        mode = st.radio(t("query_mode"), ["AND", "OR", "PHRASE"], help=t("search_hint"))
        st.metric(t("indexed_records"), summary["num_documents"])
        st.metric(t("vocabulary"), summary["vocabulary_size"])
        st.caption("Every candidate is ranked with BM25 (k1 = 1.2, b = 0.75).")

    selected = st.session_state.get("reader_doc")
    if selected in by_id:
        render_document(by_id[selected])
        footer()
        return

    intro(t("local_search"), t("search_heading"), t("search_body"))
    requested_suggestion = st.query_params.get("suggestion")
    if requested_suggestion:
        st.session_state.search_query = requested_suggestion
        st.session_state.run_local_search = True
        del st.query_params["suggestion"]
    if "home_query" in st.session_state:
        st.session_state.search_query = st.session_state.pop("home_query")
    elif "search_query" not in st.session_state:
        st.session_state.search_query = ""
    completion_event = ghost_search(
        value=st.session_state.search_query,
        vocabulary=autocomplete_vocabulary(spelling_vocabulary),
        label=t("search_terms"),
        placeholder="e.g. diabetes insulin treatment",
        submit_label=t("search_button"),
        dark=dark_mode(),
        key="local_search_ghost",
    )
    query = st.session_state.search_query
    submitted = st.session_state.pop("run_local_search", False)
    if isinstance(completion_event, dict) and completion_event.get("nonce") != st.session_state.get("last_search_nonce"):
        st.session_state.last_search_nonce = completion_event.get("nonce")
        query = str(completion_event.get("value", "")).strip()
        st.session_state.search_query = query
        submitted = True
    if not submitted:
        footer()
        return

    terms = preprocess(query, remove_stopwords=mode != "PHRASE", stem=index.stem)
    if not terms:
        empty_state("No valid search terms", "Try a biomedical term, or use PHRASE to keep stopwords.")
        footer()
        return
    started = perf_counter()
    hits = search(query, index, docs, method=mode, top_k=len(docs))
    elapsed = (perf_counter() - started) * 1000
    suggestion = suggest_query_correction(query, index, spelling_vocabulary)
    chips(list(dict.fromkeys(terms)))
    st.caption(f"{len(hits)} match(es) in {elapsed:.1f} ms. Ranking: BM25.")
    if suggestion:
        st.markdown(
            f'<p class="hx-suggestion">{escape(t("did_you_mean"))} '
            f'<a href="?suggestion={quote(suggestion)}" target="_self">{escape(suggestion)}</a>?</p>',
            unsafe_allow_html=True,
        )
    if not hits:
        empty_state("No matching articles", "Try OR, fewer terms, or different wording.")
        footer()
        return
    for rank, hit in enumerate(hits, 1):
        with st.container(key=f"result_{rank}"):
            left, right = st.columns([5, 1])
            with left:
                if hit.pmid:
                    st.markdown(
                        f'<h4 class="hx-result-title"><a href="https://pubmed.ncbi.nlm.nih.gov/{quote(hit.pmid)}/" '
                        f'target="_blank" rel="noopener noreferrer" title="Open this article on PubMed">{rank}. {escape(hit.title)}</a></h4>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"#### {rank}. {escape(hit.title)}")
                pmid = escape(hit.pmid or "—")
                pmcid = escape(f"PMC{hit.pmcid}" if hit.pmcid and not hit.pmcid.upper().startswith("PMC") else (hit.pmcid or "—"))
                st.markdown(
                    f'<p class="hx-result-identifiers">PMID: {pmid}<span>•</span>PMCID: {pmcid}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="hx-snippet">{highlight_text(hit.snippet, hit.matched_terms, index)}</p>',
                    unsafe_allow_html=True,
                )
                chips(hit.matched_terms)
            with right:
                st.metric("BM25", f"{hit.score:.3f}")
                st.button("Open", key=f"open_{hit.doc_id}", on_click=open_document, args=(hit.doc_id,), use_container_width=True)
            rows = [
                {"Term": term, "Abstract": counts["abstract"], "Body": counts["body"], "Full text": counts["full"]}
                for term, counts in hit.term_counts.items()
            ]
            if rows:
                with st.expander("Term occurrences by scope"):
                    st.dataframe(rows, hide_index=True, use_container_width=True)
    footer()


if __name__ == "__main__":
    main()
