#!/usr/bin/env python3
"""Download about 100 PubMed abstract XML records and build the local BM25 index.

This intentionally uses PubMed, not PMC: it never requests or saves article
full text. Each saved XML file contains one PubMed record with bibliographic
metadata, abstract sections, and PubMed keywords when provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import rebuild_local_index
from src.pubmed import EFETCH, _request_ncbi, _write_atomic, parse_pubmed_xml

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DEFAULT_QUERY = "diabetes mellitus[Title/Abstract] AND hasabstract[text]"
DEFAULT_EMAIL = "biomedical-ir-homework@example.com"


def search_pmids(query: str, count: int) -> list[str]:
    """Return PMID values ordered by PubMed's default relevance/date policy."""
    payload = _request_ncbi(
        ESEARCH,
        {"db": "pubmed", "term": query, "retmax": str(count), "retmode": "xml"},
    )
    root = ET.fromstring(payload)
    return [node.text.strip() for node in root.findall("IdList/Id") if node.text and node.text.strip()]


def fetch_records(pmids: list[str]) -> list[bytes]:
    """Fetch a batch and return individual PubmedArticle XML documents."""
    if not pmids:
        return []
    payload = _request_ncbi(EFETCH, {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    root = ET.fromstring(payload)
    return [ET.tostring(record, encoding="utf-8", xml_declaration=True) for record in root if record.tag.endswith("PubmedArticle") or record.tag.endswith("PubmedBookArticle")]


def download_abstract_corpus(
    *,
    query: str,
    count: int,
    raw_dir: Path,
    processed_dir: Path,
    batch_size: int = 50,
    build_index: bool = True,
) -> tuple[int, int]:
    """Download individual abstract-only XMLs; return saved files and indexed docs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pmids = search_pmids(query, count)
    if not pmids:
        raise RuntimeError("PubMed did not return any PMIDs for this query.")

    saved: list[dict[str, str]] = []
    for offset in range(0, len(pmids), batch_size):
        batch = pmids[offset : offset + batch_size]
        print(f"Fetching {offset + 1}-{offset + len(batch)} of {len(pmids)} PubMed records...")
        for xml in fetch_records(batch):
            try:
                article = parse_pubmed_xml(xml)
            except Exception as exc:  # A malformed individual record should not stop 100 downloads.
                print(f"Skipping an unreadable PubMed record: {exc}", file=sys.stderr)
                continue
            if not article.abstract:
                continue
            path = raw_dir / f"PMID{article.pmid}.xml"
            _write_atomic(path, xml)
            saved.append({"pmid": article.pmid, "title": article.title, "path": str(path)})

    manifest = raw_dir / "pubmed_abstract_meta.json"
    manifest.write_text(
        json.dumps({"query": query, "requested": count, "saved": saved}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    indexed = rebuild_local_index(raw_dir, processed_dir) if build_index and saved else 0
    return len(saved), indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed query; hasabstract[text] is recommended")
    parser.add_argument("--n", type=int, default=100, help="Number of PubMed records to request")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--batch-size", type=int, default=50, help="EFetch batch size (1-200)")
    parser.add_argument("--email", default=DEFAULT_EMAIL, help="Contact email sent to NCBI")
    parser.add_argument("--no-index", action="store_true", help="Download XML files without rebuilding the local index")
    args = parser.parse_args()
    if args.n < 1 or args.batch_size < 1 or args.batch_size > 200:
        parser.error("--n must be positive and --batch-size must be between 1 and 200.")
    os.environ["NCBI_EMAIL"] = args.email
    try:
        saved, indexed = download_abstract_corpus(
            query=args.query,
            count=args.n,
            raw_dir=args.raw,
            processed_dir=args.processed,
            batch_size=args.batch_size,
            build_index=not args.no_index,
        )
    except (OSError, RuntimeError, ET.ParseError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {saved} PubMed abstract XML file(s) in {args.raw}")
    if not args.no_index:
        print(f"Built local BM25 index with {indexed} document(s) in {args.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
