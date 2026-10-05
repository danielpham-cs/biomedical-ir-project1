"""PMID lookup, PubMed XML parsing, and local IR exports."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET

import requests

from .parse_pmc import Document
from .engine import compose_abstract
from .markup import xml_text

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ID_CONVERTER = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_MAX_ATTEMPTS = 2
_HTTP_STATE = threading.local()


class PubMedError(RuntimeError):
    """A lookup/download error that can be shown to the user."""


def normalize_pmid(value: str) -> str:
    """Accept one PMID, a PMID: prefix, or an official PubMed article URL."""
    value = value.strip()
    url = re.fullmatch(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/([0-9]+)/?(?:\?[^#\s]*)?(?:#[^\s]*)?", value, re.I)
    if url:
        value = url.group(1)
    value = re.sub(r"^PMID\s*:\s*", "", value, flags=re.I)
    if not re.fullmatch(r"[1-9][0-9]{0,9}", value):
        raise ValueError("Enter one numeric PMID, for example 30153250, or paste its PubMed URL.")
    return value


def normalize_article_id(value: str) -> str:
    """Keep PMCID prefixes: bare numbers are PMIDs, never inferred PMC IDs."""
    value = value.strip()
    url = re.fullmatch(
        r"https?://(?:pmc\.ncbi\.nlm\.nih\.gov/articles/|(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/)"
        r"(PMC[1-9][0-9]{0,9})/?(?:\?[^#\s]*)?(?:#[^\s]*)?", value, re.I,
    )
    if url:
        return url.group(1).upper()
    pmcid = re.fullmatch(r"(?:PMCID\s*:\s*)?(PMC[1-9][0-9]{0,9})", value, re.I)
    if pmcid:
        return pmcid.group(1).upper()
    try:
        return normalize_pmid(value)
    except ValueError as exc:
        raise ValueError("Enter a PMID (for example 30153250), a PMCID (for example PMC13559554), or a PubMed/PMC article URL.") from exc


def _text(element: ET.Element | None) -> str:
    return xml_text(element)


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    pmcid: str = ""
    xml: bytes = field(default=b"", repr=False)

    def to_document(self) -> Document:
        return Document(self.pmid, self.pmid, self.pmcid.removeprefix("PMC"),
                        self.title, self.abstract, "", self.journal, self.year)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


def parse_pubmed_xml(xml: bytes | str, expected_pmid: str | None = None) -> PubMedArticle:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PubMedError("PubMed returned invalid XML. Please try again.") from exc
    error = root.find(".//ERROR")
    if root.tag.upper() == "ERROR" or error is not None:
        raise PubMedError("PubMed could not return this record. Check the PMID and try again.")
    records = [root] if root.tag in {"PubmedArticle", "PubmedBookArticle"} else list(root)
    for record in records:
        citation = record.find("MedlineCitation")
        is_book = citation is None
        if is_book:
            citation = record.find("BookDocument")
        if citation is None:
            continue
        pmid = _text(citation.find("PMID"))
        if not pmid or (expected_pmid is not None and pmid != expected_pmid):
            continue
        article = citation if is_book else citation.find("Article")
        if article is None:
            continue
        sections = []
        for section in article.findall("Abstract/AbstractText"):
            content = _text(section)
            if content:
                # AbstractText labels (METHODS, RESULTS, etc.) are headings,
                # not abstract words or characters under the final spec.
                sections.append(content)
        keywords = [
            _text(keyword)
            for keyword in citation.findall("KeywordList/Keyword")
            if _text(keyword)
        ]
        authors = []
        for author in article.findall("AuthorList/Author"):
            name = _text(author.find("CollectiveName")) or " ".join(filter(None, [
                _text(author.find("ForeName")) or _text(author.find("Initials")),
                _text(author.find("LastName")), _text(author.find("Suffix")),
            ]))
            if name:
                authors.append(name)
        data_tag = "PubmedBookData" if is_book else "PubmedData"
        identifiers = {e.get("IdType"): _text(e) for e in record.findall(f"{data_tag}/ArticleIdList/ArticleId")}
        if is_book:
            identifiers.update({e.get("IdType"): _text(e) for e in citation.findall("ArticleIdList/ArticleId")})
        date = article.find("Book/PubDate" if is_book else "Journal/JournalIssue/PubDate")
        year = ""
        if date is not None:
            year = _text(date.find("Year"))
            if not year:
                match = re.search(r"\b\d{4}\b", _text(date.find("MedlineDate")))
                year = match.group() if match else ""
        year = year or _text(article.find("ArticleDate/Year"))
        pmcid = identifiers.get("pmc", "").upper()
        if not re.fullmatch(r"PMC[1-9][0-9]*", pmcid):
            pmcid = ""
        return PubMedArticle(
            pmid=pmid, title=_text(article.find("ArticleTitle")) or "(untitled)",
            abstract=compose_abstract(sections, keywords), authors=authors,
            journal=_text(article.find("Book/BookTitle" if is_book else "Journal/Title")),
            year=year, doi=identifiers.get("doi", ""), pmcid=pmcid,
            xml=xml.encode("utf-8") if isinstance(xml, str) else xml,
        )
    raise PubMedError(f"No PubMed record found for PMID {expected_pmid or '(unknown)'}.")


def _http_session() -> requests.Session:
    """Reuse TLS connections without sharing mutable sessions between threads."""
    if not hasattr(_HTTP_STATE, "session"):
        _HTTP_STATE.session = requests.Session()
    return _HTTP_STATE.session


def _retry_delay(value: str | None, fallback: float) -> float:
    """Interpret Retry-After seconds or HTTP date without exposing response text."""
    if not value:
        return fallback
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return fallback


def _request_ncbi(url: str, params: dict[str, str]) -> bytes:
    """Rate-limit requests; retry only transient failures, with short waits.

    A single abstract/ID conversion gets a 12s read timeout. PMC full text and
    batch exports retain 30s for larger responses. These are socket inactivity
    limits, not a promise about total elapsed time (DNS/multiple IPs can add time).
    """
    global _LAST_REQUEST
    params = {**params, "tool": "helix_biomedical_ir"}
    email = os.environ.get("NCBI_EMAIL", "local@example.invalid")
    params["email"] = email
    if url.startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/") and os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    headers = {
        "User-Agent": f"helix-biomedical-ir/1.0 ({email})",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
    }
    read_timeout = 30 if params.get("db") == "pmc" or "," in params.get("id", "") else 12
    started = time.monotonic()
    last_error: requests.RequestException | None = None
    endpoint = "PMC ID conversion" if url == ID_CONVERTER else ("PMC full text" if params.get("db") == "pmc" else "PubMed")
    for attempt in range(_MAX_ATTEMPTS):
        # Keep consecutive requests within NCBI's unauthenticated rate limit.
        with _REQUEST_LOCK:
            time.sleep(max(0, 0.35 - (time.monotonic() - _LAST_REQUEST)))
            _LAST_REQUEST = time.monotonic()
        response = None
        retry_delay = 0.7 * (attempt + 1)
        try:
            response = _http_session().get(url, params=params, headers=headers, timeout=(5, read_timeout))
            status = response.status_code
            if status == 429 or status in {500, 502, 503, 504}:
                retry_delay = _retry_delay(response.headers.get("Retry-After"), retry_delay)
                reason = "too many requests (HTTP 429)" if status == 429 else f"a temporary server error (HTTP {status})"
                if attempt == _MAX_ATTEMPTS - 1 or retry_delay > 5:
                    wait = f" Wait at least {retry_delay:.0f} seconds before trying again." if retry_delay > 5 else " Try again shortly."
                    raise PubMedError(f"{endpoint}: NCBI returned {reason}.{wait}")
            elif status >= 400:
                # Retrying an invalid ID/request or an access denial adds delay
                # without changing the response. Report the actual HTTP class.
                raise PubMedError(f"{endpoint}: NCBI returned HTTP {status}. Check the article ID or access to this record; the request was not retried.")
            else:
                response.raise_for_status()
                return response.content
        except requests.Timeout as exc:
            last_error = exc
        except requests.exceptions.SSLError as exc:
            raise PubMedError(f"{endpoint}: secure connection verification failed. Check the server's certificate/proxy configuration.") from exc
        except requests.ConnectionError as exc:
            last_error = exc
        except requests.RequestException as exc:
            raise PubMedError(f"{endpoint}: the HTTP request failed ({type(exc).__name__}).") from exc
        finally:
            if response is not None:
                response.close()
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(retry_delay)

    elapsed = time.monotonic() - started
    if isinstance(last_error, requests.Timeout):
        raise PubMedError(f"{endpoint}: NCBI timed out after {_MAX_ATTEMPTS} attempts ({elapsed:.1f} s). Please try again, or use an already saved XML file.") from last_error
    raise PubMedError(
        f"{endpoint}: could not connect to NCBI after {_MAX_ATTEMPTS} attempts ({elapsed:.1f} s). Check the app server's internet, DNS, VPN or proxy."
    ) from last_error


def load_saved_article(value: str, raw_dir: Path) -> PubMedArticle | None:
    """Read an existing, identity-checked PubMed XML without any network call."""
    identifier = normalize_article_id(value)
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob("PMID*.xml")) if identifier.startswith("PMC") else [raw_dir / f"PMID{identifier}.xml"]
    for path in paths:
        try:
            expected_pmid = normalize_pmid(path.stem.removeprefix("PMID"))
            article = parse_pubmed_xml(path.read_bytes(), expected_pmid=expected_pmid)
        except (OSError, PubMedError, ValueError):
            continue
        if (identifier.startswith("PMC") and article.pmcid == identifier) or article.pmid == identifier:
            return article
    return None


def _fetch_xml(database: str, identifier: str) -> bytes:
    return _request_ncbi(EFETCH, {"db": database, "id": identifier, "retmode": "xml"})


def pmcid_to_pmid(pmcid: str) -> str:
    pmcid = normalize_article_id(pmcid)
    if not pmcid.startswith("PMC"):
        raise ValueError("A PMCID must include the PMC prefix.")
    xml = _request_ncbi(ID_CONVERTER, {"ids": pmcid, "idtype": "pmcid", "format": "xml"})
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PubMedError("NCBI returned an invalid ID conversion response. Please try again.") from exc
    for record in root.findall("record"):
        if record.get("pmcid", "").upper() != pmcid:
            continue
        pmid = record.get("pmid", "")
        if re.fullmatch(r"[1-9][0-9]{0,9}", pmid):
            return pmid
        raise PubMedError(f"{pmcid} exists in PMC but has no linked PubMed PMID. A PubMed record cannot be downloaded for it yet.")
    raise PubMedError(f"NCBI could not find a PubMed mapping for {pmcid}. Check the PMCID and try again.")


def fetch_pubmed_article(value: str) -> PubMedArticle:
    identifier = normalize_article_id(value)
    pmcid = identifier if identifier.startswith("PMC") else ""
    pmid = pmcid_to_pmid(pmcid) if pmcid else identifier
    article = parse_pubmed_xml(_fetch_xml("pubmed", pmid), expected_pmid=pmid)
    if pmcid:
        if article.pmcid and article.pmcid != pmcid:
            raise PubMedError("NCBI returned inconsistent article identifiers. Please try again later.")
        article.pmcid = pmcid
    return article


def fetch_pmc_fulltext(article: PubMedArticle) -> bytes:
    """Only return an article that matches the requested identifiers and has a body."""
    if not re.fullmatch(r"PMC[1-9][0-9]*", article.pmcid):
        raise PubMedError("This PubMed record does not have a linked PMC full text.")
    xml = _fetch_xml("pmc", article.pmcid.removeprefix("PMC"))
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PubMedError("PMC returned invalid XML. Please try again.") from exc
    for node in root.iter("article"):
        ids = {e.get("pub-id-type"): _text(e) for e in node.findall("front/article-meta/article-id")}
        pmcid = ids.get("pmc", ids.get("pmcid", "")).removeprefix("PMC")
        if pmcid != article.pmcid.removeprefix("PMC"):
            continue
        if ids.get("pmid") and ids["pmid"] != article.pmid:
            continue
        if not _text(node.find("body")):
            break
        return ET.tostring(node, encoding="utf-8", xml_declaration=True)
    raise PubMedError("PMC does not provide downloadable full-text XML for this article. You can still download its PubMed record.")


def export_json(article: PubMedArticle, fulltext_xml: bytes | None = None) -> str:
    from .parse_pmc import parse_xml_text

    doc = parse_xml_text(fulltext_xml) if fulltext_xml else article.to_document()
    if doc is None:
        raise PubMedError("The full-text XML could not be converted to an IR document.")
    if fulltext_xml and (doc.pmcid != article.pmcid.removeprefix("PMC") or (doc.pmid and doc.pmid != article.pmid)):
        raise PubMedError("The full-text XML does not match this PubMed record.")
    # Some PMC exports omit the PMID even though PubMed supplies the mapping.
    doc.doc_id = doc.pmid = article.pmid
    data = doc.to_dict()
    data.update(authors=article.authors, doi=article.doi, pubmed_url=article.url,
                source="PMC" if fulltext_xml else "PubMed", has_full_text=bool(fulltext_xml))
    return json.dumps(data, ensure_ascii=False, indent=2)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_article(article: PubMedArticle, raw_dir: Path, fulltext_xml: bytes | None = None) -> list[Path]:
    """Save portable exports beside the corpus XMLs; filenames depend only on IDs."""
    pmid = normalize_pmid(article.pmid)
    raw_dir = Path(raw_dir)
    files = {f"PMID{pmid}.xml": article.xml,
             f"PMID{pmid}.json": export_json(article, fulltext_xml).encode("utf-8")}
    if fulltext_xml:
        if not re.fullmatch(r"PMC[1-9][0-9]*", article.pmcid):
            raise PubMedError("A valid PMCID is required to save the full text.")
        files[f"{article.pmcid}.xml"] = fulltext_xml
    paths = []
    for filename, content in files.items():
        path = raw_dir / filename
        _write_atomic(path, content)
        paths.append(path)
    return paths
