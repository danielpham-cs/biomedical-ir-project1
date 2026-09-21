"""Parse PMC / PubMed XML into document records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from .engine import compose_abstract


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

    @property
    def search_text(self) -> str:
        """Final-specification retrieval scope: abstract + keywords + body."""
        return "\n\n".join(part for part in (self.abstract, self.body) if part)

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
    # Preserve mixed-content order while separating adjacent block elements.
    parts: list[str] = []
    blocks = {"p", "sec", "title", "list-item", "table", "tr", "td", "break"}

    def visit(node: ET.Element) -> None:
        tag = _local(node.tag).lower()
        if tag in {"sup", "sub"}:
            parts.append(f"<{tag}>")
        if node.text:
            parts.append(node.text)
        for child in node:
            block = _local(child.tag) in blocks
            if block:
                parts.append(" ")
            visit(child)
            if block:
                parts.append(" ")
            if child.tail:
                parts.append(child.tail)
        if tag in {"sup", "sub"}:
            parts.append(f"</{tag}>")

    visit(elem)
    text = "".join(parts)
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


def _main_pmc_abstract(article_meta: ET.Element) -> tuple[list[str], list[str]]:
    """Read only the primary JATS abstract and article-level keywords."""
    abstracts = [child for child in article_meta if _local(child.tag) == "abstract"]
    normal = [
        item for item in abstracts
        if item.get("abstract-type", "").strip().lower() in {"", "normal"}
    ]
    chosen = normal[0] if normal else None
    paragraphs: list[str] = []
    if chosen is not None:
        paragraphs = [_text_of(item) for item in chosen.iter() if _local(item.tag) == "p"]
        if not paragraphs:
            paragraphs = [_text_of(chosen)]
    keywords: list[str] = []
    for group in article_meta.iter():
        if _local(group.tag) != "kwd-group":
            continue
        keywords.extend(_text_of(item) for item in group.iter() if _local(item.tag) == "kwd")
    return paragraphs, keywords


def parse_pmc_xml(path: Path | str) -> Document | None:
    """Parse a single PMC XML file into a Document."""
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    return _parse_root(tree.getroot(), path.stem)


def parse_xml_text(xml: str | bytes, fallback_id: str = "") -> Document | None:
    """Parse downloaded XML without requiring an intermediate disk file."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    return _parse_root(root, fallback_id)


def _parse_root(root: ET.Element, fallback_id: str) -> Document | None:
    if _local(root.tag) in {"PubmedArticleSet", "PubmedArticle", "PubmedBookArticle"}:
        from .pubmed import PubMedError, parse_pubmed_xml

        try:
            return parse_pubmed_xml(ET.tostring(root)).to_document()
        except PubMedError:
            return None

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
    if article_meta is not None:
        paragraphs, keywords = _main_pmc_abstract(article_meta)
        abstract = compose_abstract(paragraphs, keywords)

    body = ""
    body_elem = _find_first(root, {"body"})
    if body_elem is not None:
        # Search all body text, including section headings and table/caption text.
        body = _text_of(body_elem)

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

    doc_id = pmid or pmcid or fallback_id
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
    parsed: list[Document] = []
    for path in sorted(raw_dir.glob("**/*.xml")):
        doc = parse_pmc_xml(path)
        if doc is not None:
            parsed.append(doc)
    # PubMed can supply a PMID omitted by its corresponding PMC export.
    pmids = {doc.pmcid: doc.pmid for doc in parsed if doc.pmcid and doc.pmid}
    docs: dict[str, Document] = {}
    for doc in parsed:
        if not doc.pmid and doc.pmcid in pmids:
            doc.doc_id = doc.pmid = pmids[doc.pmcid]
        previous = docs.get(doc.doc_id)
        # Prefer full text when both PubMed metadata and PMC XML were saved.
        if previous is None or len(doc.body) > len(previous.body):
            docs[doc.doc_id] = doc
    return list(docs.values())


def iter_docs(docs: list[Document]) -> Iterator[Document]:
    yield from docs
