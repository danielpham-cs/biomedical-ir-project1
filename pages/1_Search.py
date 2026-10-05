"""Local abstract/body search using the final AND, OR, PHRASE and BM25 rules."""

from __future__ import annotations

import sys
import hashlib
from collections import Counter
from html import escape
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import build_index, index_summary, load_docs, load_index
from src.parse_pmc import parse_xml_text
from src.library import hidden_article_ids
from src.tables import rows_csv
from src.matching import document_matches, highlight_spans, make_match_plan, searchable_text
from src.preprocess import tokenize
from src.search import build_spelling_vocabulary, search, suggest_query_correction
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


@st.cache_data(show_spinner=False, max_entries=32)
def run_search(query, method, match_mode, max_edits, docs_version, index_version):
    docs, index, _ = load_resources(docs_version, index_version)
    return search(query, index, docs, method=method, match_mode=match_mode,
                  max_edits=max_edits, top_k=len(docs))


def open_document(doc_id: str) -> None:
    st.session_state.reader_doc = doc_id


def close_document() -> None:
    st.session_state.pop("reader_doc", None)


def autocomplete_vocabulary(vocabulary: dict[str, int]) -> list[str]:
    """Most frequent English corpus terms are preferred for ghost completion."""
    return [word for word, _ in sorted(vocabulary.items(), key=lambda item: (-item[1], len(item[0]), item[0]))]


def move_occurrence(key, delta, total):
    st.session_state[key] = (st.session_state.get(key, 0) + delta) % total


def render_locations(document, index):
    active = st.session_state.get("active_search")
    if not active:
        st.info("Run a search and open a result to inspect its keyword locations.")
        return
    plan = make_match_plan(active["query"], stem=index.stem, method=active["method"],
                           match_mode=active["match_mode"], max_edits=active["max_edits"])
    matches = document_matches(document, plan)
    text = searchable_text(document)
    st.caption(f'Query: {active["query"]} · {plan.kind} · {plan.method} · {len(matches)} occurrence(s)')
    st.caption("Offsets are zero-based Unicode character positions [start, end) in the displayed searchable plaintext: "
               "abstract + keywords + body, excluding the title and markup. Whitespace is normalized within paragraphs; "
               "paragraphs are separated by two newlines. Sentence numbers start at 1. Download the text to verify offsets.")
    if not matches:
        st.info("This document has no occurrences for the saved query.")
        return
    signature = hashlib.sha256(repr((document.doc_id, plan)).encode()).hexdigest()[:16]
    key = f"occurrence_{signature}"
    position = st.selectbox("Go to occurrence", range(len(matches)), key=key,
                           format_func=lambda n: f'{n + 1}. {text[matches[n].start:matches[n].end]} · '
                           f'{matches[n].section} · sentence {matches[n].sentence} · '
                           f'chars [{matches[n].start}, {matches[n].end})')
    left, right, _ = st.columns([1, 1, 4])
    left.button("Previous match", on_click=move_occurrence, args=(key, -1, len(matches)))
    right.button("Next match", on_click=move_occurrence, args=(key, 1, len(matches)))
    chosen = matches[position]
    sentence = text[chosen.sentence_start:chosen.sentence_end]
    marked = highlight_spans(sentence, [(chosen.start - chosen.sentence_start, chosen.end - chosen.sentence_start)])
    st.markdown(f'<div class="hx-reading" style="white-space:pre-wrap">{marked}</div>', unsafe_allow_html=True)
    st.dataframe([match.row(text) for match in matches], hide_index=True, use_container_width=True, height=320)
    left, right = st.columns(2)
    left.download_button("Download keyword locations (CSV)", rows_csv([match.row(text) for match in matches]),
                         f"{document.doc_id}_matches.csv", "text/csv")
    right.download_button("Download searchable plaintext", text, f"{document.doc_id}_searchable.txt", "text/plain")


def render_document(document, index, *, from_corpus=False) -> None:
    if from_corpus:
        st.page_link("pages/11_HW2_Data.py", label="Back to experiment corpus")
    else:
        st.button("Back to results", on_click=close_document)
    intro("Document analysis", document.title, "Abstract statistics follow the final course specification.")
    if from_corpus:
        abstract, locations, search_scope, statistics, sentences = st.tabs(
            ["Abstract", "Keyword locations", "Search scope", "Statistics", "Sentences and EOS"]
        )
    else:
        locations, abstract, search_scope, statistics, sentences = st.tabs(
            ["Keyword locations", "Abstract", "Search scope", "Statistics", "Sentences and EOS"]
        )
    abstract_text = document.abstract
    stats = compute_stats(abstract_text)
    with locations:
        render_locations(document, index)
    with abstract:
        if abstract_text:
            pubmed_abstract(abstract_text)
        else:
            st.info("This record has no abstract text.")
    with search_scope:
        st.caption("Search scope is abstract + keywords + body. The title is excluded.")
        active = st.session_state.get("active_search")
        if active:
            plan = make_match_plan(active["query"], stem=index.stem, method=active["method"],
                                   match_mode=active["match_mode"], max_edits=active["max_edits"])
            text = searchable_text(document)
            marked = highlight_spans(text, [(m.start, m.end) for m in document_matches(document, plan)])
            st.markdown(f'<div class="hx-reading" style="white-space:pre-wrap">{marked}</div>', unsafe_allow_html=True)
        else:
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

    requested_pmid = st.query_params.get("pmid")
    if requested_pmid is not None:
        try:
            if not requested_pmid.isascii() or not requested_pmid.isdecimal():
                raise ValueError("Invalid PMID.")
            if f"PMID{requested_pmid}" in hidden_article_ids(ROOT / "data/library.sqlite3"):
                raise ValueError("This paper is archived (soft-deleted).")
            document = parse_xml_text((ROOT / "data/raw" / f"PMID{requested_pmid}.xml").read_bytes())
            if document is None or document.pmid != requested_pmid:
                raise ValueError("The saved paper does not match this PMID.")
        except (OSError, ValueError) as exc:
            st.warning(f"Could not open this paper in the local collection: {exc}")
            st.page_link("pages/11_HW2_Data.py", label="Back to experiment corpus")
        else:
            render_document(document, build_index([document]), from_corpus=True)
        footer()
        return

    if not DOCS_PATH.exists() or not INDEX_PATH.exists():
        intro(t("local_search"), t("no_saved"), t("lookup_page_body"))
        st.page_link("pages/2_PubMed.py", label=t("lookup_action"))
        footer()
        return

    versions = (DOCS_PATH.stat().st_mtime_ns, INDEX_PATH.stat().st_mtime_ns)
    docs, index, spelling_vocabulary = load_resources(*versions)
    if not docs:
        empty_state("No readable documents", "Rebuild the local index after saving XML files.")
        footer()
        return
    by_id = {document.doc_id: document for document in docs}
    summary = index_summary(index)

    with st.sidebar:
        st.divider()
        match_mode = st.selectbox("Word matching", ["STEM", "PARTIAL", "FUZZY"], key="word_matching",
                                 format_func=lambda value: {"STEM": "Standard tokens / stemming", "PARTIAL": "Partial / substring",
                                                            "FUZZY": "Fuzzy / edit distance"}[value])
        mode = st.radio(t("query_mode"), ["AND", "OR", "PHRASE"] if match_mode == "STEM" else ["AND", "OR"], help=t("search_hint"))
        max_edits = st.select_slider("Maximum edits", [1, 2], value=1) if match_mode == "FUZZY" else 1
        if match_mode == "PARTIAL":
            st.caption("Match inside words, ignoring case: diabe → diabetes. Each query token is a substring; stemming is off.")
        elif match_mode == "FUZZY":
            st.caption("Match whole words within the chosen edit distance, including adjacent transpositions. Stemming is off.")
        else:
            st.caption(f"Index stemming: {'on' if index.stem else 'off'}. PHRASE preserves word order and stopwords.")
        st.metric(t("indexed_records"), summary["num_documents"])
        st.metric(t("vocabulary"), summary["vocabulary_size"])
        st.caption("Every candidate is ranked with BM25 (k1 = 1.2, b = 0.75).")

    selected = st.session_state.get("reader_doc")
    if selected in by_id:
        render_document(by_id[selected], index)
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
        st.session_state.run_local_search = True
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
    if submitted:
        st.session_state.active_search = {"query": query, "method": mode, "match_mode": match_mode, "max_edits": max_edits}
    elif "active_search" in st.session_state:
        query = st.session_state.active_search["query"]
        st.session_state.active_search.update(method=mode, match_mode=match_mode, max_edits=max_edits)
    else:
        footer()
        return

    plan = make_match_plan(query, stem=index.stem, method=mode, match_mode=match_mode, max_edits=max_edits)
    terms = list(plan.terms)
    if not terms:
        empty_state("No valid search terms", "Try a biomedical term, or use PHRASE to keep stopwords.")
        footer()
        return
    started = perf_counter()
    hits = run_search(query, mode, match_mode, max_edits, *versions)
    elapsed = (perf_counter() - started) * 1000
    suggestion = suggest_query_correction(query, index, spelling_vocabulary)
    chips(list(dict.fromkeys(terms)))
    st.caption(f"{len(hits)} match(es) in {elapsed:.1f} ms · {match_mode} · {mode}. Ranking: BM25 over matched index terms.")
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
                    f'<p class="hx-snippet">{hit.snippet_html}</p>',
                    unsafe_allow_html=True,
                )
                chips(hit.matched_terms)
            with right:
                st.metric("BM25", f"{hit.score:.3f}")
                st.caption(f"{len(hit.occurrences)} occurrence(s)")
                st.button("Open & locate", key=f"open_{hit.doc_id}", on_click=open_document, args=(hit.doc_id,), use_container_width=True)
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
