"""Backward-compatible EOS entrypoint backed by the shared project engine."""

from .engine import split_sentences

__all__ = ["split_sentences"]
