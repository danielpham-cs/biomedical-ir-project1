"""Homework 2 · dataset, vocabulary, preprocessing A–D and own Porter (Spec Dataset, I, II, VI; Brief Porter)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import data
from ui.hw2.shell import run_page

run_page('data', data.render)
