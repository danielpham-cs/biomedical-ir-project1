"""Abstract-only, reproducible A-D experiments for the detailed Project 2 brief."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .eos import split_sentences
from .library import hidden_article_ids
from .parse_pmc import Document, parse_xml_text
from .porter import porter_stem
from .preprocess import _stop_words

VERSION = "abstract-ad-v1"
CONDITIONS = {
    "A": "Whitespace tokenization + lowercase; punctuation retained",
    "B": "A + boundary punctuation removal; internal biomedical punctuation retained",
    "C": "B + English stopword removal",
    "D": "C + original Porter 1980 (ASCII alphabetic words)",
}
HYPHENS = str.maketrans({"‐": "-", "‑": "-", "–": "-", "’": "'"})


@dataclass
class AbstractCorpus:
    documents: list[Document]
    query: str
    fingerprint: str
    source_hashes: dict[str, str]

    def metadata(self):
        return {"document_count": len(self.documents), "query": self.query,
                "pmids": [doc.pmid for doc in self.documents], "fingerprint": self.fingerprint,
                "scope": "PubMed Abstract/AbstractText only; title, keywords and body excluded",
                "language": "eng", "pipeline": VERSION, "source_sha256": self.source_hashes}


def corpus_fingerprint(query, documents):
    payload = {"query": query, "scope": "abstract-only", "version": VERSION,
               "documents": [(d.pmid, d.abstract) for d in documents]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_snapshot(path: Path) -> AbstractCorpus:
    """Replay exported texts without network, XML, or changes to the active catalog."""
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = snapshot["metadata"]
    documents = [Document.from_dict(row) for row in snapshot["documents"]]
    if (meta.get("pipeline") != VERSION or meta.get("language") != "eng"
            or meta.get("document_count") != len(documents) or not documents
            or len({d.pmid for d in documents}) != len(documents)
            or any(not re.fullmatch(r"[1-9][0-9]*", d.pmid) or not d.abstract.strip() or d.body for d in documents)):
        raise ValueError("Snapshot must contain unique English abstract-only records for the current pipeline.")
    fingerprint = corpus_fingerprint(meta["query"], documents)
    if fingerprint != meta["fingerprint"]:
        raise ValueError("Snapshot fingerprint does not match its texts and query.")
    return AbstractCorpus(documents, meta["query"], fingerprint, meta["source_sha256"])


def load_corpus(manifest_path: Path, limit: int | None = None) -> AbstractCorpus:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [str(row["pmid"]) for row in manifest["saved"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate PMIDs in the experiment manifest.")
    limit = len(ids) if limit is None else limit
    if not 1 <= limit <= len(ids):
        raise ValueError(f"Choose 1 to {len(ids)} documents.")
    hidden = hidden_article_ids(manifest_path.parent.parent / "library.sqlite3")
    documents, hashes = [], {}
    for pmid in ids[:limit]:
        if not re.fullmatch(r"[1-9][0-9]*", pmid) or f"PMID{pmid}" in hidden:
            raise ValueError(f"Invalid or archived PMID in frozen corpus: {pmid}")
        xml = (manifest_path.parent / f"PMID{pmid}.xml").read_bytes()
        root = ET.fromstring(xml)
        languages = {node.text for node in root.findall(".//Language")}
        paragraphs = [" ".join("".join(node.itertext()).split()) for node in root.findall(".//Abstract/AbstractText")]
        text = "\n\n".join(part for part in paragraphs if part)
        doc = parse_xml_text(xml)
        if "eng" not in languages or not text or doc is None or doc.pmid != pmid:
            raise ValueError(f"Missing English abstract or mismatched identity in PMID{pmid}.xml")
        documents.append(replace(doc, abstract=text, body=""))
        hashes[pmid] = hashlib.sha256(xml).hexdigest()
    fingerprint = corpus_fingerprint(manifest["query"], documents)
    return AbstractCorpus(documents, manifest["query"], fingerprint, hashes)


def clean_boundary(token: str) -> str:
    token = token.translate(HYPHENS)
    while token and not token[0].isalnum():
        token = token[1:]
    while token and not token[-1].isalnum():
        if token[-1] == "%" and len(token) > 1 and token[-2].isdigit():
            break
        token = token[:-1]
    return token if any(c.isalnum() for c in token) else ""


def token_trace(text: str, stops=None) -> list[dict]:
    stops = _stop_words() if stops is None else stops
    rows = []
    for match in re.finditer(r"\S+", text):
        original = match.group()
        a = original.lower()
        b = clean_boundary(a)
        c = b if b not in stops else ""
        d = porter_stem(c) if c else ""
        actions = []
        if original != a:
            actions.append("lowercased")
        if a != b:
            actions.append("punctuation normalized" if b else "punctuation removed")
        if b and not c:
            actions.append("stopword removed")
        if c != d:
            actions.append("stemmed" if d else "empty stem removed")
        rows.append({"original": original, "A": a, "B": b, "C": c, "D": d,
                     "action": "; ".join(actions) or "unchanged", "start": match.start(), "end": match.end()})
    return rows


def tokens(text: str, condition: str, stops=None) -> list[str]:
    if condition not in CONDITIONS:
        raise ValueError("Condition must be A, B, C or D.")
    words = text.lower().split()
    if condition != "A":
        words = [word for token in words if (word := clean_boundary(token))]
    if condition in {"C", "D"}:
        stops = _stop_words() if stops is None else stops
        words = [word for word in words if word not in stops]
    if condition == "D":
        words = [stem for word in words if (stem := porter_stem(word))]
    return words


def fit_zipf(ranks, frequencies) -> dict:
    ranks, frequencies = np.asarray(ranks, dtype=float), np.asarray(frequencies, dtype=float)
    if ranks.shape != frequencies.shape or ranks.ndim != 1 or np.any(ranks <= 0) or np.any(frequencies <= 0):
        raise ValueError("Ranks and frequencies must be equally sized positive one-dimensional arrays.")
    if not np.isfinite(ranks).all() or not np.isfinite(frequencies).all():
        raise ValueError("Regression input must be finite.")
    result = {"points": len(ranks), "rank_start": int(ranks[0]) if len(ranks) else None,
              "rank_end": int(ranks[-1]) if len(ranks) else None, "slope": None, "intercept": None,
              "zipf_exponent": None, "r_squared": None, "rmse_log10": None, "note": ""}
    if len(ranks) < 2 or np.ptp(ranks) == 0:
        result["note"] = "At least two distinct ranks are needed."
        return result
    x, y = np.log10(ranks), np.log10(frequencies)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    constant = np.ptp(y) == 0
    result.update(slope=float(slope), intercept=float(intercept), zipf_exponent=float(-slope),
                  r_squared=None if constant else float(1 - np.sum(residual**2) / np.sum((y - y.mean())**2)),
                  rmse_log10=float(np.sqrt(np.mean(residual**2))),
                  note="Constant frequencies: R² is undefined; this is not evidence of a perfect Zipf fit." if constant else "Descriptive OLS in log10 space; not a power-law goodness-of-fit test.")
    return result


def region_fits(rows):
    n = len(rows)
    head, middle = math.ceil(n * 0.1), math.ceil(n * 0.8)
    return {name: fit_zipf([r["rank"] for r in subset], [r["CF"] for r in subset])
            for name, subset in (("All", rows), ("Head (0–10%)", rows[:head]),
                                 ("Middle (10–80%)", rows[head:middle]), ("Tail (80–100%)", rows[middle:]))}


def representative_terms(rows, count=20):
    """Ten most frequent, then biomedical terms, then concentrated terms; deterministic."""
    by_term = {row["term"]: row for row in rows}
    chosen = [r["term"] for r in rows[:10]]
    for word in ("glp-1", "receptor", "semaglutide", "tirzepatide", "agonists"):
        if word in by_term and word not in chosen:
            chosen.append(word)
    for row in sorted(rows, key=lambda r: (-r["CF_per_document"], -r["CF"], r["term"])):
        if len(chosen) >= count:
            break
        if row["CF"] >= 5 and row["term"] not in chosen:
            chosen.append(row["term"])
    for row in rows:
        if len(chosen) >= count:
            break
        if row["term"] not in chosen:
            chosen.append(row["term"])
    return [by_term[term] for term in chosen[:count]]


def analyze_corpus(corpus: AbstractCorpus, stops=None) -> dict:
    if not corpus.documents or len({d.pmid for d in corpus.documents}) != len(corpus.documents):
        raise ValueError("Use a nonempty corpus with unique PMIDs.")
    stops = _stop_words() if stops is None else stops
    settings = {"version": VERSION, "conditions": CONDITIONS,
                "stopwords": sorted(stops), "regression_log_base": 10, "idf_log_base": "e",
                "regions": "head ranks 0–10%, middle 10–80%, tail 80–100%; ceiling at boundaries",
                "punctuation": "Strip non-alphanumeric boundaries, keep numeric trailing %, internal punctuation; normalize Unicode hyphens/apostrophe.",
                "term_selection": "Top 10 CF, up to 5 predefined biomedical terms present in vocabulary, then descending CF/DF with CF>=5; deterministic fill."}
    run_key = hashlib.sha256(json.dumps({"corpus": corpus.fingerprint, "settings": settings}, sort_keys=True).encode()).hexdigest()
    conditions = {}
    n = len(corpus.documents)
    for condition in CONDITIONS:
        cf, df = Counter(), Counter()
        for doc in corpus.documents:
            counts = Counter(tokens(doc.abstract, condition, stops))
            cf.update(counts)
            df.update(counts.keys())
        total = sum(cf.values())
        rows = [{"rank": i, "term": term, "CF": frequency, "DF": df[term], "DF_fraction": df[term] / n,
                 "CF_per_document": frequency / df[term], "IDF": math.log(n / df[term]),
                 "relative_frequency": frequency / total}
                for i, (term, frequency) in enumerate(sorted(cf.items(), key=lambda x: (-x[1], x[0])), 1)]
        conditions[condition] = {"stats": {"documents": n, "tokens": total, "vocabulary": len(cf),
                                           "average_tokens_per_document": total / n, "hapax": sum(v == 1 for v in cf.values())},
                                 "rows": rows, "fits": region_fits(rows), "selected_terms": representative_terms(rows),
                                 "spectrum": [{"frequency": freq, "terms": count} for freq, count in sorted(Counter(cf.values()).items())]}
    groups = defaultdict(list)
    for row in conditions["C"]["rows"]:
        stem = porter_stem(row["term"])
        groups[stem].append((row["term"], row["CF"]))
    merged = [{"stem": stem, "forms": ", ".join(f"{term} ({freq})" for term, freq in sorted(forms)),
               "form_count": len(forms), "CF": sum(freq for _, freq in forms)}
              for stem, forms in groups.items() if stem and len(forms) > 1]
    return {"corpus": corpus.metadata(), "settings": settings, "run_key": run_key, "conditions": conditions,
            "merged_groups": sorted(merged, key=lambda x: (-x["CF"], x["stem"]))}


def term_documents(corpus, term, condition, stops=None):
    rows = []
    for doc in corpus.documents:
        count = tokens(doc.abstract, condition, stops).count(term)
        if count:
            rows.append({"PMID": doc.pmid, "Title": doc.title, "TF": count})
    df = len(rows)
    idf = math.log(len(corpus.documents) / df) if df else 0
    return [dict(row, IDF=idf, TF_IDF=row["TF"] * idf) for row in sorted(rows, key=lambda r: (-r["TF"], r["PMID"]))]


def term_spans(text, term, condition, stops=None):
    return [(r["start"], r["end"]) for r in token_trace(text, stops) if r[condition] == term]


def embedding_sentences(corpus, condition="C"):
    return [words for doc in corpus.documents for paragraph in doc.abstract.split("\n\n")
            for sentence in split_sentences(paragraph) if (words := tokens(sentence, condition))]


def embedding_preprocessing(condition="C"):
    return {"pipeline": VERSION, "condition": condition, "scope": "abstract-only", "remove_stopwords": condition in {"C", "D"}, "stem": condition == "D"}
