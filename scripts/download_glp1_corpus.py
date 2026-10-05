"""Download an English GLP-1 corpus, archive the old library, and rebuild search.

Downloads resume from data/staging/glp1. Activation happens only after the full
requested sample is validated. XML/JSON and old experiment models are retained.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_pubmed_abstracts import search_pmids, fetch_records
from src.index import load_docs, rebuild_local_index
from src.library import connect, count_articles, hidden_article_ids, save_article_record
from src.parse_pmc import parse_directory
from src.pubmed import PubMedArticle, _write_atomic, export_json, parse_pubmed_xml, save_article

QUERY = '("GLP-1"[Title/Abstract] OR "glucagon-like peptide-1"[Title/Abstract] OR "glucagon like peptide 1"[Title/Abstract]) AND hasabstract[text] AND english[lang]'


def write_json(path, value):
    _write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def valid_article(xml, expected=None):
    article = parse_pubmed_xml(xml)
    root = ET.fromstring(xml)
    languages = {node.text for node in root.findall(".//Language")}
    # Check the actual abstract, not the parser's appended keywords.
    text = " ".join("".join(node.itertext()) for node in root.findall(".//Abstract/AbstractText"))
    if "eng" not in languages or not text.strip() or (expected and article.pmid != expected):
        raise ValueError(f"Invalid English abstract / PMID: {article.pmid}")
    return article


def restore(archive: Path):
    previous = json.loads((archive / "previous_pmids.json").read_text(encoding="utf-8"))
    added = json.loads((archive / "new_pmids.json").read_text(encoding="utf-8"))
    with closing(connect(ROOT / "data/library.sqlite3")) as db:
        with db:
            db.executemany("UPDATE articles SET deleted_at=? WHERE pmid=?",
                           [(datetime.now(timezone.utc).isoformat(), pmid) for pmid in added])
            db.executemany("UPDATE articles SET deleted_at=NULL WHERE pmid=?", [(pmid,) for pmid in previous])
    _write_atomic(ROOT / "data/raw/pubmed_abstract_meta.json", (archive / "pubmed_abstract_meta.json").read_bytes())
    count = rebuild_local_index(ROOT / "data/raw", ROOT / "data/processed")
    print(f"Restored {count} active papers; downloaded GLP-1 files are retained.")


def run(count: int):
    raw, processed = ROOT / "data/raw", ROOT / "data/processed"
    database = ROOT / "data/library.sqlite3"
    manifest_path = raw / "pubmed_abstract_meta.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("query") == QUERY and len(current.get("saved", [])) == count:
            with closing(connect(database)) as db:
                active = {row[0] for row in db.execute("SELECT pmid FROM articles WHERE deleted_at IS NULL")}
            if active == {row["pmid"] for row in current["saved"]}:
                for pmid in active:
                    valid_article((raw / f"PMID{pmid}.xml").read_bytes(), pmid)
                indexed = rebuild_local_index(raw, processed)
                print(f"Already active: {len(active)} GLP-1 papers; index verified ({indexed}).")
                return

    previous_docs = {doc.doc_id: doc for doc in parse_directory(raw)}
    if (processed / "docs.jsonl").exists():
        previous_docs.update({doc.doc_id: doc for doc in load_docs(processed / "docs.jsonl")})
    if any(not doc.pmid for doc in previous_docs.values()):
        raise ValueError("A legacy paper has no PMID; catalog its identity before migration.")
    previous_ids = {doc.pmid for doc in previous_docs.values()}
    if database.exists():
        with closing(connect(database)) as db:
            previous_ids.update(row[0] for row in db.execute("SELECT pmid FROM articles WHERE deleted_at IS NULL"))
    excluded = previous_ids | {value[4:] for value in hidden_article_ids(database) if value.startswith("PMID")}
    stage = ROOT / "data/staging/glp1"
    selection = stage / f"selection_{count}.json"
    if selection.exists():
        saved_selection = json.loads(selection.read_text(encoding="utf-8"))
        if saved_selection["query"] != QUERY:
            raise ValueError("Staged query differs; use a separate staging directory.")
        candidates = [pmid for pmid in saved_selection["candidates"] if pmid not in excluded]
    else:
        candidates = [pmid for pmid in dict.fromkeys(search_pmids(QUERY, count + len(excluded) + 200)) if pmid not in excluded]
        write_json(selection, {"query": QUERY, "requested": count, "candidates": candidates,
                               "excluded_pmids": sorted(excluded), "created_at": datetime.now(timezone.utc).isoformat()})
    articles = {}
    for offset in range(0, len(candidates), 50):
        batch = candidates[offset:offset + 50]
        missing = []
        for pmid in batch:
            path = stage / f"PMID{pmid}.xml"
            try:
                articles[pmid] = valid_article(path.read_bytes(), pmid)
            except (OSError, ValueError, RuntimeError, ET.ParseError):
                missing.append(pmid)
        if missing:
            for xml in fetch_records(missing):
                try:
                    article = valid_article(xml)
                    if article.pmid not in missing:
                        raise ValueError("EFetch returned an unrequested PMID")
                    _write_atomic(stage / f"PMID{article.pmid}.xml", xml)
                    articles[article.pmid] = article
                except (ValueError, RuntimeError, ET.ParseError) as exc:
                    print(f"Skipped record: {exc}", flush=True)
        print(f"Validated {len(articles)}/{count} English abstracts", flush=True)
        if len(articles) >= count:
            break
    selected = [articles[pmid] for pmid in candidates if pmid in articles][:count]
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)}/{count} valid records. Active collection was not replaced.")

    # Snapshot the current catalog, corpus selection and index before activation.
    archive = ROOT / "data/archives" / datetime.now(timezone.utc).strftime("before_glp1_%Y%m%dT%H%M%S%fZ")
    archive.mkdir(parents=True)
    if database.exists():
        with closing(sqlite3.connect(database)) as source, closing(sqlite3.connect(archive / "library.sqlite3")) as target:
            source.backup(target)
    for path in [manifest_path, processed / "docs.jsonl", processed / "index.pkl"]:
        if path.exists():
            shutil.copy2(path, archive / path.name)
    write_json(archive / "previous_pmids.json", sorted(previous_ids))
    write_json(archive / "new_pmids.json", [article.pmid for article in selected])

    # Backfill legacy XML/index papers into SQLite so every old PMID has a tombstone.
    for doc in previous_docs.values():
        xml = raw / f"PMID{doc.pmid}.xml"
        article = parse_pubmed_xml(xml.read_bytes()) if xml.exists() else PubMedArticle(
            pmid=doc.pmid, title=doc.title, abstract=doc.abstract, journal=doc.journal,
            year=doc.year, pmcid=f"PMC{doc.pmcid.removeprefix('PMC')}" if doc.pmcid else "")
        json_path = raw / f"PMID{doc.pmid}.json"
        if not json_path.exists():
            _write_atomic(json_path, export_json(article).encode("utf-8"))
        paths = [xml, json_path]
        if article.pmcid and (raw / f"{article.pmcid}.xml").exists():
            paths.append(raw / f"{article.pmcid}.xml")
        save_article_record(database, article, paths=paths, project_root=ROOT, has_full_text=bool(doc.body))
    for article in selected:
        paths = save_article(article, raw)
        save_article_record(database, article, paths=paths, project_root=ROOT, has_full_text=False)
    with closing(connect(database)) as db:
        with db:
            db.executemany("UPDATE articles SET deleted_at=? WHERE pmid=?",
                           [(datetime.now(timezone.utc).isoformat(), pmid) for pmid in sorted(previous_ids)])
    manifest = {"query": QUERY, "requested": count, "default_count": count,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "language": "eng",
                "archive": str(archive.relative_to(ROOT)),
                "saved": [{"pmid": a.pmid, "title": a.title, "path": f"data/raw/PMID{a.pmid}.xml"} for a in selected]}
    write_json(manifest_path, manifest)
    indexed = rebuild_local_index(raw, processed)
    if indexed != count or count_articles(database) != count:
        raise RuntimeError(f"Activation counts differ: indexed={indexed}, catalog={count_articles(database)}. Snapshot: {archive}")
    print(f"DONE: {count} active GLP-1 papers; {len(previous_ids)} old papers soft-deleted.", flush=True)
    print(f"Snapshot: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--restore", type=Path, help="Restore previous active papers using a migration snapshot")
    args = parser.parse_args()
    if not 10 <= args.n <= 1000:
        parser.error("--n must be between 10 and 1000")
    try:
        restore(args.restore) if args.restore else run(args.n)
    except (OSError, RuntimeError, ValueError, sqlite3.Error, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
