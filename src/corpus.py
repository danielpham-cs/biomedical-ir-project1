"""Validated XML imports and corpus statistics for the course workspace."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from .parse_pmc import Document, parse_xml_text
from .pubmed import _write_atomic
from .stats import compute_stats


def parse_importable_xml(content: bytes) -> Document:
    """Validate one uploaded PMC/PubMed XML file without writing it."""
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("XML files must be no larger than 20 MB each.")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("This file is not valid XML.") from exc
    tag = root.tag.split("}")[-1]
    if tag not in {"article", "pmc-articleset", "PubmedArticleSet", "PubmedArticle", "PubmedBookArticle"}:
        raise ValueError("Use a PMC article XML or a PubMed article XML export.")
    if tag in {"pmc-articleset", "PubmedArticleSet"}:
        records = [e for e in root if e.tag.split("}")[-1] in {"article", "PubmedArticle", "PubmedBookArticle"}]
        if len(records) != 1:
            raise ValueError("Upload one article per XML file; split multi-article exports first.")
    digest = sha256(content).hexdigest()[:16]
    doc = parse_xml_text(content, fallback_id=digest)
    if doc is None:
        raise ValueError("No readable article title or abstract was found in this XML.")
    return doc


def import_xml_document(content: bytes, raw_dir: Path) -> tuple[Document, Path]:
    """Validate and save one article XML under a safe, identifier-based name."""
    doc = parse_importable_xml(content)
    root = ET.fromstring(content)
    tag = root.tag.split("}")[-1]
    digest = sha256(content).hexdigest()[:16]
    # IDs in uploaded XML are untrusted too; hash fallback avoids path traversal.
    if tag.startswith("Pubmed") and re.fullmatch(r"[1-9][0-9]*", doc.pmid):
        name = f"PMID{doc.pmid}.xml"
    elif re.fullmatch(r"[1-9][0-9]*", doc.pmcid):
        name = f"PMC{doc.pmcid}.xml"
    else:
        name = f"article_{digest}.xml"
    path = Path(raw_dir) / name
    _write_atomic(path, content)
    return doc, path


def import_xml(content: bytes, raw_dir: Path) -> Path:
    """Backward-compatible import helper returning only the saved path."""
    _, path = import_xml_document(content, raw_dir)
    return path


def document_rows(docs: list[Document]) -> list[dict]:
    rows = []
    for doc in docs:
        stats = compute_stats(doc.abstract)
        rows.append({
            "Document ID": doc.doc_id, "Title": doc.title, "Journal": doc.journal,
            "Year": doc.year, "Text available": "Full text" if doc.body else "Title / abstract",
            "Characters": stats.num_characters, "Words": stats.num_words,
            "Characters (no spaces)": stats.characters_no_space,
            "Sentences (EOS)": stats.num_sentences, "Unique words": stats.vocabulary_size,
        })
    return rows
