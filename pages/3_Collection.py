"""Import article XML and inspect the actual corpus before indexing."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import document_rows, import_xml
from src.index import load_docs, load_index, rebuild_local_index
from src.parse_pmc import parse_directory
from src.library import list_articles, set_articles_deleted
from ui.brand import footer, inject_css, intro, masthead, sidebar_nav


@st.cache_data(show_spinner=False)
def inspect_raw(root: str, versions: tuple):
    return parse_directory(Path(root))


@st.cache_data(show_spinner=False)
def inspect_statistics(docs):
    return document_rows(docs)


def main() -> None:
    st.set_page_config(page_title="Helix · XML collection", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    sidebar_nav()
    masthead("01 / XML collection")
    intro("Prepare your dataset", "A collection you can inspect.",
          "Import PMC or PubMed XML, verify the text available for each paper, and build the local retrieval index.")
    raw = ROOT / "data/raw"
    processed = ROOT / "data/processed"
    files = sorted(raw.rglob("*.xml"))
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    database = ROOT / "data/library.sqlite3"
    if database.exists():
        signature += ((str(database), database.stat().st_mtime_ns, database.stat().st_size),)
    docs = inspect_raw(str(raw), signature)
    index_path = processed / "index.pkl"
    index = load_index(index_path) if index_path.exists() else None
    with st.container(key="overview_metrics"):
        for col, name, value in zip(st.columns(4), ["XML files (including archived)", "Active readable papers", "Full-text papers", "Indexed papers"],
                                    [len(files), len(docs), sum(bool(d.body) for d in docs), index.N if index else 0]):
            col.metric(name, value)

    with st.expander("Archived papers / restore"):
        archived = [row for row in list_articles(database, include_deleted=True) if row["Archived"]]
        st.caption(f"{len(archived)} archived papers. Source files remain intact. Restoring changes the active library; the Project 2 experiment manifest stays fixed.")
        if archived:
            st.dataframe(archived, hide_index=True, use_container_width=True)
            by_id = {row["PMID"]: row for row in archived}
            restore_ids = st.multiselect("Papers to restore", list(by_id), format_func=lambda pmid: f"{pmid} | {by_id[pmid]['Title']}")
            if st.button("Restore selected papers", disabled=not restore_ids):
                restored = set_articles_deleted(database, restore_ids, deleted=False)
                try:
                    rebuild_local_index(raw, processed, stem=True, remove_stopwords=True)
                    st.session_state.collection_restore_result = f"Restored {restored} papers and rebuilt the library index."
                    st.rerun()
                except (OSError, ValueError, RuntimeError, LookupError) as exc:
                    st.error(f"Restored {restored} catalog entries; index rebuild failed: {exc}. Use Build / update index to retry.")
        if st.session_state.get("collection_restore_result"):
            st.success(st.session_state.collection_restore_result)

    upload_tab, corpus_tab, build_tab = st.tabs(["1 · Import XML", "2 · Inspect documents", "3 · Build index"])
    with upload_tab:
        st.page_link("pages/2_PubMed.py", label="Retrieve XML by PMID / PMCID →")
        uploads = st.file_uploader("Or upload existing article XMLs", type=["xml"], accept_multiple_files=True,
                                   help="One article per file, up to 20 MB each. Both PMC and PubMed article exports are accepted.")
        if uploads and st.button("Import selected XML files", type="primary"):
            saved, errors = [], []
            for upload in uploads:
                try:
                    path = import_xml(upload.getvalue(), raw)
                    saved.append(path.name)
                except (ValueError, OSError) as exc:
                    errors.append(f"{upload.name}: {exc}")
            st.session_state.import_report = (saved, errors)
            st.rerun()
        if "import_report" in st.session_state:
            saved, errors = st.session_state.import_report
            if saved:
                st.success(f"Saved {len(saved)} XML file(s). Review the collection, then build the index.")
                with st.expander("Saved filenames"):
                    st.text("\n".join(saved))
            for error in errors:
                st.error(error)
        st.caption("Imports use article identifiers as filenames. Re-importing the same ID updates its XML; the index changes only when you rebuild it.")
        with st.expander("Download a larger sample from the command line"):
            st.code("python scripts/download_sample.py --query \"diabetes mellitus treatment\" --n 150", language="bash")
            st.caption("The sample size and topic are choices for your demo, not requirements imposed by the homework PDF.")

    with corpus_tab:
        if not docs:
            st.info("No readable documents yet. Import XML files or save a paper from the online lookup.")
        else:
            rows = inspect_statistics(docs)
            st.caption("Statistics cover the abstract and its final Keywords line. Duplicate PMID / PMCID records are merged, preferring the full-text body for search.")
            st.dataframe(rows, hide_index=True, use_container_width=True)
            chart = [{"Document": row["Document ID"], "Words": row["Words"]} for row in rows]
            st.markdown("#### Document lengths")
            st.bar_chart(chart, x="Document", y="Words", color="#087f78")
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            st.download_button("Download corpus statistics (CSV)", buffer.getvalue().encode("utf-8-sig"),
                               file_name="corpus_statistics.csv", mime="text/csv")
            if len(files) > len(docs):
                st.caption(f"{len(files)} XML files yielded {len(docs)} unique readable documents. Duplicate records or unreadable files are not counted twice.")

    with build_tab:
        st.markdown("#### Preprocessing settings")
        st.caption("Documents and queries use the same settings. Rebuilding updates the entire indexed collection.")
        stem = True
        stop = True
        st.caption("Final specification: Porter stemming is always enabled; AND/OR always remove English stopwords.")
        st.caption("Lowercasing and tokenization are always applied. Numeric tokens such as the 2 in “type 2” are retained.")
        if st.button("Build / update index", type="primary", disabled=not docs):
            try:
                with st.spinner("Parsing the complete articles and building the inverted index…"):
                    count = rebuild_local_index(raw, processed, stem=stem, remove_stopwords=stop)
                st.session_state.collection_build_result = count
                st.rerun()
            except (OSError, ValueError, RuntimeError, LookupError) as exc:
                st.error(f"Could not build the index: {exc}")
        if "collection_build_result" in st.session_state:
            st.success(f"Index built with {st.session_state.collection_build_result} documents.")
        if index:
            st.caption(f"Current index: {index.N:,} documents · {index.vocabulary_size():,} terms · stemming {'on' if index.stem else 'off'} · stopwords {'removed' if index.remove_stopwords else 'kept'}")
            st.page_link("pages/1_Search.py", label="Search & analyze the collection →")
        st.caption("Existing indexed documents are kept even if their original XML is no longer in data/raw.")
    footer()


if __name__ == "__main__":
    main()
