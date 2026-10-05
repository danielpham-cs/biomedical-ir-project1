"""Background Word2Vec training: the page stays usable while a model trains in a worker thread.

Jobs live in a process-wide registry (one per model key), so a rerun or another
browser tab sees the same progress, and the same model is never trained twice at once.
The worker thread never calls Streamlit; the page polls the job state instead.
"""
from __future__ import annotations

import threading
from pathlib import Path

import streamlit as st

from src.word2vec import train_skipgram


@st.cache_resource
def jobs() -> dict:
    return {}


_lock = threading.Lock()


def start(key, sentences, config, corpus_metadata, preprocessing, path: Path) -> dict:
    """Start training unless the same model is already running; return its job."""
    with _lock:
        job = jobs().get(key)
        if job and job['status'] == 'running':
            return job
        job = {'status': 'running', 'epoch': 0, 'total': config.epochs, 'loss': None, 'error': None,
               'path': str(path), 'architecture': config.architecture}
        jobs()[key] = job

    def progress(epoch, total, loss):
        job.update(epoch=epoch, total=total, loss=loss)

    def work():
        try:
            model = train_skipgram(sentences, config, corpus_metadata=corpus_metadata,
                                   preprocessing=preprocessing, progress=progress)
            model.save(path)
            job['status'] = 'done'
        except Exception as exc:  # reported on the page; the thread must not crash silently
            job.update(status='failed', error=str(exc))

    threading.Thread(target=work, name=f'word2vec-{key[:8]}', daemon=True).start()
    return job


def running() -> dict:
    return {key: job for key, job in jobs().items() if job['status'] == 'running'}
