"""Homework 2 · Skip-gram word embeddings (Brief)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import word2vec
from ui.hw2.shell import run_page

run_page('word2vec', word2vec.render)
