"""Persist additional PubMed experiment corpora without replacing the course corpus."""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from scripts.download_pubmed_abstracts import search_pmids, fetch_records
from src.pubmed import _write_atomic
from src.zipf_experiment import load_corpus, load_snapshot


def _connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root/'registry.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS corpora (
        id TEXT PRIMARY KEY, query TEXT NOT NULL, document_count INTEGER NOT NULL,
        requested INTEGER NOT NULL, created_at TEXT NOT NULL, fingerprint TEXT NOT NULL,
        deleted_at TEXT)''')
    return db


def list_corpora(root, include_deleted=False):
    if not (Path(root)/'registry.sqlite3').exists():
        return []
    with closing(_connect(root)) as db:
        return [dict(row) for row in db.execute('SELECT * FROM corpora WHERE (? OR deleted_at IS NULL) ORDER BY created_at DESC', (include_deleted,))]


def corpus_directory(root, corpus_id):
    if not re.fullmatch(r'[a-f0-9]{20}', corpus_id):
        raise ValueError('Invalid experiment corpus ID.')
    return Path(root)/corpus_id


def set_archived(root, corpus_id, archived=True):
    corpus_directory(root, corpus_id)
    with closing(_connect(root)) as db, db:
        db.execute('UPDATE corpora SET deleted_at=? WHERE id=?', (datetime.now(timezone.utc).isoformat() if archived else None, corpus_id))


def fetch_corpus(root, query, count, *, progress=None, excluded_pmids=()):
    """Resume staged batches; expose a corpus only after a complete valid snapshot.

    A query with fewer available papers saves the actual nonzero count, explicitly
    recorded separately from the requested maximum. Transient download errors do
    not register an incomplete corpus, and an identical request reuses its snapshot.
    """
    query = query.strip()
    if not query or len(query) > 2000 or isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3000:
        raise ValueError('Enter a PubMed query and a document limit between 1 and 3,000.')
    full_query = f'({query}) AND hasabstract[text] AND english[lang]'
    corpus_id = hashlib.sha256(json.dumps([full_query, count]).encode()).hexdigest()[:20]
    directory = corpus_directory(root, corpus_id)
    previous = next((r for r in list_corpora(root, True) if r['id'] == corpus_id), None)
    if previous:
        load_snapshot(directory/'corpus_snapshot.json')
        if previous['deleted_at']:
            raise ValueError('This exact corpus is archived. Restore it from Archived corpora.')
        return previous
    raw = directory/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    selection = directory/'selection.json'
    if selection.exists():
        candidates = json.loads(selection.read_text(encoding='utf8'))['pmids']
    else:
        candidates = list(dict.fromkeys(search_pmids(full_query, count + 200 + len(excluded_pmids))))
        _write_atomic(selection, json.dumps({'query': full_query, 'pmids': candidates}).encode())
    candidates = [p for p in candidates if re.fullmatch(r'[1-9][0-9]*', p) and p not in set(excluded_pmids)]
    saved = []
    def valid(xml, pmid=None):
        tree = ET.fromstring(xml)
        actual = tree.findtext('.//MedlineCitation/PMID')
        if not actual or (pmid and actual != pmid):
            return None
        if 'eng' not in {e.text for e in tree.findall('.//Language')}:
            return None
        abstract = ' '.join(''.join(e.itertext()).strip() for e in tree.findall('.//Abstract/AbstractText')).strip()
        if not abstract:
            return None
        return {'pmid': actual, 'title': ''.join(tree.find('.//ArticleTitle').itertext()) if tree.find('.//ArticleTitle') is not None else ''}
    for offset in range(0, len(candidates), 50):
        batch = candidates[offset:offset+50]
        accepted, missing = {}, []
        for pmid in batch:
            path = raw/f'PMID{pmid}.xml'
            try:
                record = valid(path.read_bytes(), pmid)
            except (OSError, ET.ParseError):
                record = None
            if record: accepted[pmid] = record
            else: missing.append(pmid)
        if missing:
            for xml in fetch_records(missing):
                try: record = valid(xml)
                except ET.ParseError: continue
                if record and record['pmid'] in missing:
                    _write_atomic(raw/f"PMID{record['pmid']}.xml", xml)
                    accepted[record['pmid']] = record
        saved.extend(accepted[p] for p in batch if p in accepted)
        if progress: progress(min(len(saved), count), count)
        if len(saved) >= count: break
    saved = saved[:count]
    if not saved:
        raise ValueError('No valid English abstracts matched. Try a broader PubMed query.')
    manifest = raw/'pubmed_abstract_meta.json'
    _write_atomic(manifest, json.dumps({'query': full_query, 'requested': count, 'saved': saved}, ensure_ascii=False).encode('utf8'))
    corpus = load_corpus(manifest)
    _write_atomic(directory/'corpus_snapshot.json', json.dumps({'metadata': corpus.metadata(), 'documents': [d.to_dict() for d in corpus.documents]}, ensure_ascii=False).encode('utf8'))
    row = dict(id=corpus_id, query=query, document_count=len(corpus.documents), requested=count,
               created_at=datetime.now(timezone.utc).isoformat(), fingerprint=corpus.fingerprint, deleted_at=None)
    with closing(_connect(root)) as db, db:
        db.execute('INSERT INTO corpora VALUES (:id,:query,:document_count,:requested,:created_at,:fingerprint,:deleted_at)', row)
    return row
