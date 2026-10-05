"""Homework 2 · DF, CF vs DF and IDF (Spec Parts IV, VII, VIII)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import terms
from ui.hw2.shell import run_page

run_page('terms', terms.render)
