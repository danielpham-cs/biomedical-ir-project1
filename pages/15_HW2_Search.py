"""Homework 2 · DP edit distance, spelling correction and match locations (Brief)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import search
from ui.hw2.shell import run_page

run_page('search', search.render)
