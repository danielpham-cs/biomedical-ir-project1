#!/usr/bin/env python3
"""Parse PMC XML files and build the inverted index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import build_index, index_summary, save_docs, save_index
from src.parse_pmc import parse_directory
from src.preprocess import ensure_nltk_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--docs-out", type=Path, default=ROOT / "data" / "processed" / "docs.jsonl"
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=ROOT / "data" / "processed" / "index.pkl",
    )
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--keep-stopwords", action="store_true")
    args = parser.parse_args()

    ensure_nltk_data()
    print(f"Parsing XML from {args.raw} …")
    docs = parse_directory(args.raw)
    if not docs:
        print("No documents found. Run scripts/download_sample.py first.", file=sys.stderr)
        return 1
    print(f"Parsed {len(docs)} documents")
    save_docs(docs, args.docs_out)
    print(f"Wrote {args.docs_out}")

    index = build_index(
        docs,
        stem=not args.no_stem,
        remove_stopwords=not args.keep_stopwords,
    )
    save_index(index, args.index_out)
    summary = index_summary(index)
    print(f"Wrote {args.index_out}")
    print("Index summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
