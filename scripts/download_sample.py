#!/usr/bin/env python3
"""Download a sample PMC Open Access corpus via NCBI E-utilities / OA API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw"

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OA_SVC = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Contact email for NCBI polite use (override with --email)
DEFAULT_EMAIL = "biomedical-ir-homework@example.com"


def esearch_pmcids(term: str, retmax: int, email: str) -> list[str]:
    params = {
        "db": "pmc",
        "term": f"{term} AND open access[filter]",
        "retmax": retmax,
        "retmode": "json",
        "email": email,
        "tool": "biomedical_ir_project1",
    }
    r = requests.get(ESEARCH, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pmc_xml(pmcid: str, email: str) -> str | None:
    """Fetch full-text XML for a PMC article (numeric id without PMC prefix)."""
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
        "email": email,
        "tool": "biomedical_ir_project1",
    }
    r = requests.get(EFETCH, params=params, timeout=90)
    if r.status_code != 200:
        return None
    text = r.text.strip()
    if not text or "<error" in text.lower()[:200]:
        return None
    # Basic sanity: should look like XML article
    if "<article" not in text and "<pmc-articleset" not in text:
        return None
    return text


def unwrap_articleset(xml_text: str) -> list[str]:
    """If efetch wraps multiple articles, split into individual article XMLs."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [xml_text]
    tag = root.tag.split("}")[-1]
    if tag == "pmc-articleset":
        out = []
        for child in list(root):
            out.append(ET.tostring(child, encoding="unicode"))
        return out or [xml_text]
    return [xml_text]


def download_corpus(
    query: str,
    n: int,
    out_dir: Path,
    email: str,
    sleep_s: float = 0.34,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Searching PMC for: {query!r} (retmax={n})")
    ids = esearch_pmcids(query, n, email)
    print(f"Found {len(ids)} PMC IDs")
    saved: list[Path] = []
    meta = []

    for i, pmcid in enumerate(ids, 1):
        path = out_dir / f"PMC{pmcid}.xml"
        if path.exists() and path.stat().st_size > 100:
            print(f"[{i}/{len(ids)}] skip existing PMC{pmcid}")
            saved.append(path)
            meta.append({"pmcid": pmcid, "path": str(path)})
            continue
        try:
            xml_text = fetch_pmc_xml(pmcid, email)
        except requests.RequestException as e:
            print(f"[{i}/{len(ids)}] fail PMC{pmcid}: {e}")
            time.sleep(sleep_s)
            continue
        if not xml_text:
            print(f"[{i}/{len(ids)}] empty PMC{pmcid}")
            time.sleep(sleep_s)
            continue
        articles = unwrap_articleset(xml_text)
        # Save first article under this PMC id
        path.write_text(articles[0], encoding="utf-8")
        saved.append(path)
        meta.append({"pmcid": pmcid, "path": str(path)})
        print(f"[{i}/{len(ids)}] saved {path.name} ({path.stat().st_size} bytes)")
        time.sleep(sleep_s)

    meta_path = out_dir / "download_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. {len(saved)} files in {out_dir}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="diabetes mellitus treatment",
        help="PubMed/PMC search query",
    )
    parser.add_argument("--n", type=int, default=150, help="Number of articles")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    args = parser.parse_args()
    try:
        download_corpus(args.query, args.n, args.out, args.email)
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
