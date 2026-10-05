"""Homework 2 · CF Top 50, rank-frequency, log-log regression, two domains (Spec III, V, optional; Brief Zipf)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import zipf
from ui.hw2.shell import run_page

run_page('zipf', zipf.render)
