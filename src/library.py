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
    return connection


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
                article.pmcid,
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


def list_articles(database_path: Path | str) -> list[dict]:
    """Return saved records in a table-friendly form."""
    path = Path(database_path)
    if not path.exists():
        return []
    with closing(connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT pmid, pmcid, title, journal, year, has_full_text, source,
                   pubmed_xml_path, pmc_xml_path, json_path, saved_at
            FROM articles ORDER BY saved_at DESC, pmid DESC
            """
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
        }
        for row in rows
    ]


def count_articles(database_path: Path | str) -> int:
    """Return the number of records in the local catalog."""
    path = Path(database_path)
    if not path.exists():
        return 0
    with closing(connect(path)) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
