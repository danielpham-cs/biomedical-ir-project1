"""Small SQLite catalog for locally downloaded PubMed and PMC records."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .pubmed import PubMedArticle

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY CHECK (pmid GLOB '[1-9]*'),
    pmcid TEXT UNIQUE,
    title TEXT NOT NULL,
    journal TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL CHECK (source IN ('PubMed', 'PMC')),
    has_full_text INTEGER NOT NULL CHECK (has_full_text IN (0, 1)),
    pubmed_xml_path TEXT NOT NULL,
    pmc_xml_path TEXT NOT NULL DEFAULT '',
    json_path TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the local catalog and make its schema available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(articles)")}
    if "deleted_at" not in columns:
        connection.execute("ALTER TABLE articles ADD COLUMN deleted_at TEXT")
        connection.commit()
    return connection


def hidden_article_ids(database_path: Path | str) -> set[str]:
    """Read tombstones without creating or changing the database."""
    path = Path(database_path)
    if not path.exists():
        return set()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if "deleted_at" not in {row[1] for row in connection.execute("PRAGMA table_info(articles)")}:
            return set()
        rows = connection.execute("SELECT pmid, pmcid FROM articles WHERE deleted_at IS NOT NULL").fetchall()
    return {f"PMID{pmid}" for pmid, _ in rows} | {f"PMC{str(pmcid).removeprefix('PMC')}" for _, pmcid in rows if pmcid}


def active_documents(documents, database_path: Path | str):
    hidden = hidden_article_ids(database_path)
    return [doc for doc in documents if f"PMID{doc.pmid}" not in hidden
            and (not doc.pmcid or f"PMC{doc.pmcid.removeprefix('PMC')}" not in hidden)]


def set_articles_deleted(database_path: Path | str, pmids: list[str], *, deleted: bool = True) -> int:
    """Soft-delete or explicitly restore catalog records; source files stay intact."""
    timestamp = datetime.now(timezone.utc).isoformat() if deleted else None
    with closing(connect(database_path)) as connection:
        with connection:
            cursor = connection.executemany("UPDATE articles SET deleted_at=? WHERE pmid=?", [(timestamp, pmid) for pmid in pmids])
        return cursor.rowcount


def save_article_record(
    database_path: Path | str,
    article: PubMedArticle,
    *,
    paths: list[Path],
    project_root: Path,
    has_full_text: bool,
) -> None:
    """Upsert one article's metadata after its local exports were written."""
    filenames = {path.name: path for path in paths}
    pubmed_xml = filenames.get(f"PMID{article.pmid}.xml")
    json_file = filenames.get(f"PMID{article.pmid}.json")
    pmc_xml = filenames.get(f"{article.pmcid}.xml") if article.pmcid else None
    if pubmed_xml is None or json_file is None:
        raise ValueError("PubMed XML and JSON exports are required before cataloging an article.")

    root = project_root.resolve()

    def relative(path: Path) -> str:
        return str(path.resolve().relative_to(root))

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with closing(connect(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO articles (
                pmid, pmcid, title, journal, year, authors, doi, abstract, source,
                has_full_text, pubmed_xml_path, pmc_xml_path, json_path, saved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
                pmcid=excluded.pmcid, title=excluded.title, journal=excluded.journal,
                year=excluded.year, authors=excluded.authors, doi=excluded.doi,
                abstract=excluded.abstract, source=excluded.source,
                has_full_text=excluded.has_full_text,
                pubmed_xml_path=excluded.pubmed_xml_path,
                pmc_xml_path=excluded.pmc_xml_path, json_path=excluded.json_path,
                saved_at=excluded.saved_at
            """,
            (
                article.pmid,
                article.pmcid or None,
                article.title,
                article.journal,
                article.year,
                "; ".join(article.authors),
                article.doi,
                article.abstract,
                "PMC" if has_full_text else "PubMed",
                int(has_full_text),
                relative(pubmed_xml),
                relative(pmc_xml) if pmc_xml else "",
                relative(json_file),
                timestamp,
            ),
        )
        connection.commit()


def list_articles(database_path: Path | str, *, include_deleted: bool = False) -> list[dict]:
    """Return saved records in a table-friendly form."""
    path = Path(database_path)
    if not path.exists():
        return []
    with closing(connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT pmid, pmcid, title, journal, year, has_full_text, source,
                   pubmed_xml_path, pmc_xml_path, json_path, saved_at, deleted_at
            FROM articles WHERE (? OR deleted_at IS NULL) ORDER BY saved_at DESC, pmid DESC
            """, (include_deleted,)
        ).fetchall()
    return [
        {
            "PMID": row["pmid"],
            "PMCID": row["pmcid"] or "-",
            "Title": row["title"],
            "Journal": row["journal"],
            "Year": row["year"],
            "Text": "Full text" if row["has_full_text"] else "Title / abstract",
            "Saved": row["saved_at"].replace("+00:00", "Z"),
            **({"Archived": row["deleted_at"] or ""} if include_deleted else {}),
        }
        for row in rows
    ]


def count_articles(database_path: Path | str, *, include_deleted: bool = False) -> int:
    """Return the number of records in the local catalog."""
    path = Path(database_path)
    if not path.exists():
        return 0
    with closing(connect(path)) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM articles WHERE (? OR deleted_at IS NULL)", (include_deleted,)).fetchone()[0])
