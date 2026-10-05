"""Reproducible, subject-specific corpus and Zipf/Porter experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .eos import split_sentences
from .markup import plain_text
from .parse_pmc import Document, parse_xml_text
from .porter import porter_stem
from .preprocess import preprocess

PIPELINE_VERSION = "hw2-porter1980-v1"


@dataclass
class ExperimentCorpus:
    documents: list[Document]
    query: str
    fingerprint: str

    def metadata(self) -> dict:
        return {"query": self.query, "document_count": len(self.documents),
                "pmids": [doc.pmid for doc in self.documents], "fingerprint": self.fingerprint,
                "scope": "abstract + keywords; titles and bodies excluded; Keywords label removed",
                "pipeline": PIPELINE_VERSION}


def abstract_text(document: Document) -> str:
    return re.sub(r"(?im)^\s*keywords\s*:\s*", "", plain_text(document.abstract))


def load_experiment_corpus(manifest_path: Path, limit: int = 100) -> ExperimentCorpus:
    """Read only PMIDs in the chosen download manifest, never the mixed index.

    Resolve XML from PMID inside the manifest directory rather than trusting its
    historical absolute paths. Missing records fail explicitly for reproducibility.
    """
    if not 10 <= limit <= 1000:
        raise ValueError("Select between 10 and 1000 documents for Homework 2.")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("query") or not isinstance(manifest.get("saved"), list):
        raise ValueError("A PubMed download manifest with query and saved records is required.")
    pmids = list(dict.fromkeys(str(record.get("pmid", "")) for record in manifest["saved"]))
    from .library import hidden_article_ids
    hidden = hidden_article_ids(manifest_path.parent.parent / "library.sqlite3")
    pmids = [pmid for pmid in pmids if f"PMID{pmid}" not in hidden]
    if len(pmids) < limit:
        raise ValueError(f"The manifest contains {len(pmids)} unique records; requested {limit}.")
    documents: list[Document] = []
    for pmid in pmids[:limit]:
        if not re.fullmatch(r"[1-9][0-9]*", pmid):
            raise ValueError(f"Invalid PMID in manifest: {pmid!r}")
        path = manifest_path.parent / f"PMID{pmid}.xml"
        document = parse_xml_text(path.read_bytes())
        if document is None or document.pmid != pmid or not abstract_text(document).strip():
            raise ValueError(f"Missing abstract or incorrect PMID in {path.name}.")
        documents.append(document)
    payload = {"query": manifest["query"], "pipeline": PIPELINE_VERSION,
               "documents": [(doc.pmid, abstract_text(doc)) for doc in documents]}
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return ExperimentCorpus(documents, str(manifest["query"]), fingerprint)


def experiment_tokens(text: str, *, remove_stopwords: bool, stem: bool = False) -> list[str]:
    tokens = preprocess(plain_text(text), remove_stopwords=remove_stopwords, stem=False)
    if stem:
        tokens = [porter_stem(token) for token in tokens]
    return [token for token in tokens if token]


def training_sentences(corpus: ExperimentCorpus, *, remove_stopwords: bool, stem: bool = False) -> list[list[str]]:
    """Keep context windows within sentences and documents."""
    sequences = []
    for document in corpus.documents:
        for paragraph in abstract_text(document).split("\n\n"):
            for sentence in split_sentences(paragraph):
                tokens = experiment_tokens(sentence, remove_stopwords=remove_stopwords, stem=stem)
                if tokens:
                    sequences.append(tokens)
    return sequences


def frequency_rows(counts: Counter) -> list[dict]:
    total = sum(counts.values())
    return [{"rank": rank, "term": term, "frequency": count, "relative_frequency": count / total}
            for rank, (term, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1)]


def frequency_spectrum(counts: Counter) -> list[dict]:
    return [{"frequency": count, "terms": terms} for count, terms in sorted(Counter(counts.values()).items())]


def _distribution_stats(counts: Counter) -> dict:
    frequencies = np.array(sorted(counts.values(), reverse=True), dtype=float)
    slope = r_squared = None
    if len(frequencies) > 1 and np.ptp(frequencies) > 0:
        x = np.log10(np.arange(1, len(frequencies) + 1))
        y = np.log10(frequencies)
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = float(1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"tokens": sum(counts.values()), "vocabulary": len(counts),
            "hapax": sum(value == 1 for value in counts.values()),
            "log_log_slope": None if slope is None else float(slope), "log_log_r_squared": r_squared}


def compare_stemming(corpus: ExperimentCorpus, *, remove_stopwords: bool = True) -> dict:
    before: Counter = Counter()
    for doc in corpus.documents:
        before.update(experiment_tokens(abstract_text(doc), remove_stopwords=remove_stopwords))
    if not before:
        raise ValueError("The corpus has no tokens after preprocessing.")
    after: Counter = Counter()
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    removed = 0
    for term, count in before.items():
        stem = porter_stem(term)
        if not stem:
            removed += count
            continue
        after[stem] += count
        groups[stem].append((term, count))
    merged = [{"stem": stem, "forms": ", ".join(f"{term} ({count})" for term, count in sorted(forms)),
               "form_count": len(forms), "frequency": sum(count for _, count in forms)}
              for stem, forms in groups.items() if len(forms) > 1]
    merged.sort(key=lambda row: (-row["frequency"], row["stem"]))
    return {"corpus": corpus.metadata(), "remove_stopwords": remove_stopwords,
            "stemmer": "local original Porter 1980 (ASCII alphabetic tokens)",
            "before": frequency_rows(before), "after": frequency_rows(after),
            "spectrum_before": frequency_spectrum(before), "spectrum_after": frequency_spectrum(after),
            "stats_before": _distribution_stats(before), "stats_after": _distribution_stats(after),
            "vocabulary_reduction_percent": 100 * (1 - len(after) / len(before)),
            "empty_stems_removed": removed, "merged_groups": merged}


def rows_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()
