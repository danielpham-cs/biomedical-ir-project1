"""Homework 2 · overview, executive summary, requirement map and report answers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.hw2 import overview
from ui.hw2.shell import run_page

run_page('overview', overview.render)
