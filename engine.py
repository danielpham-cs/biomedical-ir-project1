"""Public entrypoint for the unified Biomedical IR Project 1 engine.

Application modules use :mod:`src.engine`; this file lets course tooling import
the same authoritative implementation as ``engine`` from the project root.
"""

from src.engine import compose_abstract, normalize_whitespace, split_sentences, tokenize_words

__all__ = ["compose_abstract", "normalize_whitespace", "split_sentences", "tokenize_words"]
