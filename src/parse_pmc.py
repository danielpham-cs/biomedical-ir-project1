"""Parse PMC / PubMed XML into document records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET


@dataclass
class Document:
    doc_id: str
    pmid: str
    pmcid: str
    title: str
    abstract: str
    body: str
    journal: str
    year: str
    full_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        parts = [p for p in (self.title, self.abstract, self.body) if p]
        self.full_text = "\n\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            pmid=data.get("pmid", ""),
            pmcid=data.get("pmcid", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            body=data.get("body", ""),
            journal=data.get("journal", ""),
            year=data.get("year", ""),
        )


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _text_of(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    parts: list[str] = []
    for node in elem.iter():
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def _find_first(root: ET.Element, names: set[str]) -> ET.Element | None:
    for elem in root.iter():
        if _local(elem.tag) in names:
            return elem
    return None


def _find_all_text(root: ET.Element, names: set[str]) -> str:
    chunks: list[str] = []
    for elem in root.iter():
        if _local(elem.tag) in names:
            t = _text_of(elem)
            if t:
                chunks.append(t)
    return " ".join(chunks)


def parse_pmc_xml(path: Path | str) -> Document | None:
    """Parse a single PMC XML file into a Document."""
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    article_meta = _find_first(root, {"article-meta"})
    journal_meta = _find_first(root, {"journal-meta"})

    pmid = ""
    pmcid = ""
    search_root = article_meta if article_meta is not None else root
    for elem in search_root.iter():
        if _local(elem.tag) == "article-id":
            id_type = elem.attrib.get("pub-id-type", "")
            val = (elem.text or "").strip()
            if id_type == "pmid":
                pmid = val
            elif id_type in ("pmc", "pmcid"):
                pmcid = val.replace("PMC", "")

    title_elem = _find_first(search_root, {"article-title"})
    title = _text_of(title_elem)

    abstract = ""
    abs_elem = _find_first(search_root, {"abstract"})
    if abs_elem is not None:
        abstract = _text_of(abs_elem)

    body = ""
    body_elem = _find_first(root, {"body"})
    if body_elem is not None:
        # Prefer paragraph text to avoid captions/noise explosion
        paras = _find_all_text(body_elem, {"p"})
        body = paras if paras else _text_of(body_elem)
        # Cap body length for demo corpus
        if len(body) > 20000:
            body = body[:20000] + "…"

    journal = ""
    if journal_meta is not None:
        jtitle = _find_first(journal_meta, {"journal-title"})
        journal = _text_of(jtitle)

    year = ""
    for elem in search_root.iter():
        if _local(elem.tag) == "year" and (elem.text or "").strip():
            year = elem.text.strip()
            break

    if not title and not abstract:
        return None

    doc_id = pmid or pmcid or path.stem
    return Document(
        doc_id=str(doc_id),
        pmid=pmid,
        pmcid=pmcid,
        title=title or "(untitled)",
        abstract=abstract,
        body=body,
        journal=journal,
        year=year,
    )


def parse_directory(raw_dir: Path | str) -> list[Document]:
    """Parse all XML files under a directory."""
    raw_dir = Path(raw_dir)
    docs: list[Document] = []
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("**/*.xml")):
        doc = parse_pmc_xml(path)
        if doc is None or doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        docs.append(doc)
    return docs


def iter_docs(docs: list[Document]) -> Iterator[Document]:
    yield from docs
