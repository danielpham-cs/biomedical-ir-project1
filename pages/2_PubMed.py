"""Look up a PubMed abstract by PMID or PMCID and analyze it in place."""

from __future__ import annotations

import sys
import csv
import json
from hashlib import sha256
from collections import Counter
from html import escape
from io import StringIO
from pathlib import Path
from sqlite3 import Error as DatabaseError
from time import perf_counter

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import import_xml_document, parse_importable_xml
from src.index import rebuild_local_index
from src.library import save_article_record
from src.preprocess import tokenize
from src.pubmed import (
    PubMedArticle,
    PubMedError,
    fetch_pmc_fulltext,
    fetch_pubmed_article,
    load_saved_article,
    normalize_article_id,
    save_article,
    _write_atomic,
)
from src.stats import compute_stats
from ui.brand import (
    BRAND,
    chips,
    footer,
    inject_css,
    intro,
    masthead,
    pubmed_abstract,
    reading_text,
    sidebar_nav,
)
from ui.i18n import t

RAW_DIR = ROOT / "data" / "raw"
DATABASE_PATH = ROOT / "data" / "library.sqlite3"
PROCESSED_DIR = ROOT / "data" / "processed"
PARSER_CACHE_VERSION = "pubmed-network-v2"
MAX_UPLOADS = 20


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
def fetch_article(identifier: str, parser_version: str) -> PubMedArticle:
    """Cache the remote metadata lookup; no files are written here."""
    del parser_version  # Makes corrected XML typography a distinct cache entry.
    return fetch_pubmed_article(identifier)


@st.cache_data(ttl=3600, max_entries=30, show_spinner=False)
def fetch_fulltext(article: PubMedArticle) -> bytes:
    return fetch_pmc_fulltext(article)


def save_to_local_library(article: PubMedArticle, include_fulltext: bool, progress=None) -> dict:
    """Save exports only after the user explicitly selects the optional action."""
    fulltext = None
    fulltext_message = ""
    timings = {}
    if include_fulltext and article.pmcid:
        if progress:
            progress("Fetching the optional full text from PMC…")
        started = perf_counter()
        try:
            fulltext = fetch_fulltext(article)
        except PubMedError as exc:
            fulltext_message = str(exc)
        timings["PMC full text (or cache)"] = perf_counter() - started

    if progress:
        progress("Saving this paper to your library…")
    started = perf_counter()
    paths = save_article(article, RAW_DIR, fulltext)
    save_article_record(
        DATABASE_PATH,
        article,
        paths=paths,
        project_root=ROOT,
        has_full_text=bool(fulltext),
    )
    timings["Local file and catalog save"] = perf_counter() - started
    if progress:
        progress("Updating the local search index…")
    started = perf_counter()
    indexed_count = rebuild_local_index(RAW_DIR, PROCESSED_DIR)
    timings["Local index rebuild"] = perf_counter() - started
    return {
        "pmid": article.pmid,
        "paths": [str(path.relative_to(ROOT)) for path in paths],
        "indexed_count": indexed_count,
        "fulltext_message": fulltext_message,
        "has_fulltext": bool(fulltext),
        "timings": timings,
    }


def display_analysis(abstract: str, document_id: str, *, key_prefix: str) -> None:
    """Display homework statistics calculated solely from the current abstract."""
    abstract_text = abstract
    stats = compute_stats(abstract_text)
    overview, statistics, sentences = st.tabs([t("abstract"), t("statistics"), t("sentences_eos")])
    with overview:
        if abstract_text:
            pubmed_abstract(abstract_text)
        else:
            st.info("This record has no abstract text to analyze.")
    with statistics:
        st.caption(
            "These values are calculated from the abstract, including its final Keywords line."
        )
        for column, label, value in zip(
            st.columns(4),
            [t("characters"), t("words"), t("sentences"), t("unique_words")],
            [stats.num_characters, stats.num_words, stats.num_sentences, stats.vocabulary_size],
        ):
            column.metric(label, f"{value:,}")
        st.metric("Average words per sentence", f"{stats.avg_sentence_length:g}")
        csv_buffer = StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(stats.to_dict()))
        writer.writeheader()
        writer.writerow({key: value for key, value in stats.to_dict().items() if key != "sentences"})
        st.download_button(
            "Download abstract statistics CSV",
            csv_buffer.getvalue(),
            file_name=f"{document_id}_abstract_statistics.csv",
            mime="text/csv",
            key=f"{key_prefix}_statistics_csv",
            on_click="ignore",
        )
        counts = Counter(tokenize(abstract_text)).most_common(20)
        if counts:
            st.markdown("#### Most frequent abstract terms")
            st.dataframe(
                [{"Term": term, "Occurrences": count} for term, count in counts],
                hide_index=True,
                use_container_width=True,
            )
    with sentences:
        st.caption("Sentence boundaries use the abbreviation-aware EOS rules required by the homework.")
        if stats.sentences:
            st.dataframe(
                [{"Sentence": number, "Text": sentence} for number, sentence in enumerate(stats.sentences, 1)],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No abstract sentences are available.")


def display_article_title(title: str, pmid: str) -> None:
    """Show a safe PubMed link when the record has a stable PMID."""
    if pmid:
        st.markdown(
            f'<h3 class="hx-article-title"><a href="https://pubmed.ncbi.nlm.nih.gov/{escape(pmid)}/" '
            f'target="_blank" rel="noopener noreferrer" title="Open this article on PubMed">{escape(title)}</a></h3>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### {escape(title)}")


def display_batch_statistics_table(rows: list[dict]) -> None:
    """Render batch statistics with a readable PubMed link in the Title cell."""
    columns = list(rows[0])
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rendered_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = str(row[column])
            if column == "Title" and row["PMID"] != "—":
                cells.append(
                    f'<td class="hx-batch-title"><a href="https://pubmed.ncbi.nlm.nih.gov/{escape(str(row["PMID"]))}/" '
                    f'target="_blank" rel="noopener noreferrer" title="Open this article on PubMed">{escape(value)}</a></td>'
                )
            else:
                cells.append(f"<td>{escape(value)}</td>")
        rendered_rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="hx-batch-table"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rendered_rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def display_save_options(article: PubMedArticle) -> None:
    """Offer an XML download or a one-click import into the searchable library."""
    st.divider()
    st.markdown(f"### {t('save_options')}")
    st.caption(t("import_hint"))
    with st.expander(t("import_options")):
        include_fulltext = st.checkbox(
            "Also try to download PMC full-text XML",
            value=False,
            disabled=not bool(article.pmcid),
            help="Available only when the record has a linked PMCID.",
            key=f"import_fulltext_{article.pmid}",
        )
    left, right = st.columns(2)
    left.download_button(
        t("download_xml"),
        article.xml,
        file_name=f"PMID{article.pmid}.xml",
        mime="application/xml",
        on_click="ignore",
        use_container_width=True,
    )
    import_clicked = right.button(
        t("import_paper"),
        key=f"import_paper_{article.pmid}",
        type="primary",
        use_container_width=True,
    )
    if import_clicked:
        st.session_state.pop("save_result", None)
        try:
            with st.status(t("importing_paper"), expanded=True) as status:
                st.session_state.save_result = save_to_local_library(article, include_fulltext, progress=status.write)
                status.update(label=t("import_complete"), state="complete", expanded=False)
        except (PubMedError, OSError, RuntimeError, LookupError, DatabaseError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("save_result")
    if saved and saved.get("pmid") == article.pmid:
        st.success(t("import_success").format(pmid=article.pmid))
        if saved["has_fulltext"]:
            st.caption("PMC full text is included in your library.")
        elif saved["fulltext_message"]:
            st.caption(f"The abstract was imported; PMC full text was unavailable: {saved['fulltext_message']}")
        st.page_link("pages/1_Search.py", label=t("search_imported"))
        with st.expander(t("import_details")):
            st.code("\n".join(saved["paths"]), language=None)
            st.caption(f"Library: {saved['indexed_count']} indexed paper(s).")
            if saved.get("timings"):
                st.caption(" · ".join(f"{step}: {seconds:.2f} s" for step, seconds in saved["timings"].items()))


def uploaded_json(document) -> bytes:
    data = document.to_dict()
    data.update(source="Uploaded XML", has_full_text=bool(document.body))
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def save_uploaded_xml(content: bytes) -> dict:
    """Persist a reviewed upload and rebuild the shared local search index."""
    document, xml_path = import_xml_document(content, RAW_DIR)
    json_path = xml_path.with_suffix(".json")
    _write_atomic(json_path, uploaded_json(document))
    return {
        "paths": [str(xml_path.relative_to(ROOT)), str(json_path.relative_to(ROOT))],
        "indexed_count": rebuild_local_index(RAW_DIR, PROCESSED_DIR),
    }


def display_uploaded_save_options(document, content: bytes, filename: str, *, key_prefix: str) -> None:
    """Offer the same download/save options for a locally supplied XML file."""
    st.divider()
    st.markdown(f"### {t('save_options')}")
    st.caption("This upload is only being previewed. Save is optional and writes it to the server's local search collection.")
    left, right = st.columns(2)
    left.download_button("Download uploaded XML", content, file_name=filename, mime="application/xml", use_container_width=True, key=f"{key_prefix}_xml", on_click="ignore")
    right.download_button("Download parsed IR JSON", uploaded_json(document), file_name=f"{document.doc_id}_ir.json", mime="application/json", use_container_width=True, key=f"{key_prefix}_json", on_click="ignore")
    with st.expander("Save this uploaded XML to the local search library"):
        result_key = f"{key_prefix}_save_result"
        if st.button("Save XML and JSON locally", key=f"{key_prefix}_save", type="primary", use_container_width=True):
            try:
                with st.spinner("Saving files and updating the local search index..."):
                    st.session_state[result_key] = save_uploaded_xml(content)
            except (ValueError, OSError, RuntimeError, LookupError) as exc:
                st.error(str(exc))
        saved = st.session_state.get(result_key)
        if saved:
            st.success(f"Saved locally and indexed {saved['indexed_count']} document(s).")
            st.code("\n".join(saved["paths"]), language=None)
            st.page_link("pages/1_Search.py", label="Search the saved local library")


def main() -> None:
    st.set_page_config(
        page_title=f"{BRAND} | PubMed abstract lookup",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    sidebar_nav()
    masthead(t("lookup_page_title"))

    intro(
        "PMID or PMCID",
        t("lookup_page_heading"),
        t("lookup_page_body"),
    )
    with st.container(key="search_box"):
        with st.form("pubmed_lookup"):
            identifier = st.text_input(
                t("pmid_or_pmcid"),
                placeholder="30153250 or PMC13559554",
                help="Numbers are treated as PMIDs. PMCID must start with PMC.",
            )
            refresh = st.checkbox("Fetch latest from NCBI", value=False,
                                  help="Normally an existing local XML or the app's one-hour cache is reused. Select this to bypass both and request a fresh record.")
            submitted = st.form_submit_button(t("search_abstract"), type="primary", use_container_width=True)
        st.caption(t("lookup_hint"))

    st.caption(f"Or upload up to {MAX_UPLOADS} PMC or PubMed XML files to preview and analyze their abstracts without contacting PubMed.")
    uploaded_files = st.file_uploader(
        "Upload article XML files",
        type=["xml"],
        accept_multiple_files=True,
        key="abstract_xml_upload",
    )

    if submitted:
        st.session_state.pop("save_result", None)
        st.session_state.pop("lookup_info", None)
        started = perf_counter()
        try:
            normalized = normalize_article_id(identifier)
            article = None if refresh else load_saved_article(normalized, RAW_DIR)
            source = "Saved local XML (no network)"
            if article is None:
                with st.spinner("Fetching the PubMed record (up to 2 attempts per request)…"):
                    article = fetch_pubmed_article(normalized) if refresh else fetch_article(normalized, PARSER_CACHE_VERSION)
                source = "Live NCBI request" if refresh else "NCBI / one-hour app cache"
            st.session_state.lookup_article = article
            st.session_state.lookup_info = {"source": source, "seconds": perf_counter() - started}
        except (ValueError, PubMedError) as exc:
            st.session_state.pop("lookup_article", None)
            st.error(f"{exc} (Lookup elapsed: {perf_counter() - started:.1f} s)")

    uploaded_records = []
    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOADS:
            st.error(f"Choose no more than {MAX_UPLOADS} XML files at a time. You selected {len(uploaded_files)}.")
        else:
            for position, uploaded_file in enumerate(uploaded_files, 1):
                uploaded_content = uploaded_file.getvalue()
                signature = sha256(uploaded_content).hexdigest()[:16]
                try:
                    uploaded_records.append((position, uploaded_file.name, uploaded_content, signature, parse_importable_xml(uploaded_content)))
                except ValueError as exc:
                    st.error(f"{uploaded_file.name}: {exc}")

    if uploaded_records:
        batch_rows = []
        for position, filename, _, _, document in uploaded_records:
            stats = compute_stats(document.abstract)
            batch_rows.append({
                "#": position,
                "File": filename,
                "Title": document.title,
                "PMID": document.pmid or "—",
                "PMCID": f"PMC{document.pmcid}" if document.pmcid else "—",
                "Abstract sentences": stats.num_sentences,
                "Abstract words": stats.num_words,
                "Abstract characters": stats.num_characters,
                "No-space chars": stats.characters_no_space,
                "Unique words": stats.vocabulary_size,
                "Avg. words / sentence": stats.avg_sentence_length,
            })
        st.markdown(f"### Uploaded abstracts ({len(uploaded_records)})")
        file_count, sentences, words, characters = st.columns(4)
        file_count.metric("Readable XML files", len(batch_rows))
        sentences.metric("Total abstract sentences", f"{sum(row['Abstract sentences'] for row in batch_rows):,}")
        words.metric("Total abstract words", f"{sum(row['Abstract words'] for row in batch_rows):,}")
        characters.metric("Total abstract characters", f"{sum(row['Abstract characters'] for row in batch_rows):,}")
        display_batch_statistics_table(batch_rows)
        csv_buffer = StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
        st.download_button(
            "Download batch statistics CSV",
            csv_buffer.getvalue(),
            file_name="uploaded_abstract_statistics.csv",
            mime="text/csv",
            key="uploaded_batch_statistics_csv",
            on_click="ignore",
        )
        for position, filename, uploaded_content, signature, uploaded_document in uploaded_records:
            identifier = uploaded_document.pmid or (f"PMC{uploaded_document.pmcid}" if uploaded_document.pmcid else uploaded_document.doc_id)
            with st.expander(f"{position}. {uploaded_document.title}", expanded=len(uploaded_records) == 1):
                if uploaded_document.pmid:
                    st.caption("Click the title inside this section to open the source record on PubMed.")
                    display_article_title(uploaded_document.title, uploaded_document.pmid)
                identifiers = ([f"PMID {uploaded_document.pmid}"] if uploaded_document.pmid else []) + ([f"PMC{uploaded_document.pmcid}"] if uploaded_document.pmcid else [])
                if identifiers:
                    chips(identifiers)
                st.caption(" | ".join(filter(None, [uploaded_document.journal, uploaded_document.year, filename])))
                display_analysis(uploaded_document.abstract, identifier, key_prefix=f"upload_{signature}")
                display_uploaded_save_options(uploaded_document, uploaded_content, filename, key_prefix=f"upload_{signature}")
    else:
        article = st.session_state.get("lookup_article")
        if article:
            info = st.session_state.get("lookup_info")
            if info:
                st.caption(f"{info['source']} · Lookup time: {info['seconds']:.2f} s")
            display_article_title(article.title, article.pmid)
            chips([f"PMID {article.pmid}"] + ([article.pmcid] if article.pmcid else []))
            st.caption(" | ".join(filter(None, [article.journal, article.year, article.doi])))
            if article.authors:
                st.caption("; ".join(article.authors[:8]))
            display_analysis(article.abstract, f"PMID{article.pmid}", key_prefix=f"pmid_{article.pmid}")
            display_save_options(article)

    footer()


if __name__ == "__main__":
    main()
