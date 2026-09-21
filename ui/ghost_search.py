"""A small Streamlit component for inline search-term completion."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "ghost_search",
    path=Path(__file__).with_name("ghost_search_component"),
)


def ghost_search(
    *,
    value: str,
    vocabulary: list[str],
    label: str,
    placeholder: str,
    submit_label: str,
    dark: bool,
    key: str,
) -> dict | None:
    """Render a keyboard-accessible ghost-text input.

    The browser computes prefix completions locally, so typing does not trigger
    Streamlit reruns. Only Enter or the search button sends a query to Python.
    """
    return _COMPONENT(
        value=value,
        vocabulary=vocabulary,
        label=label,
        placeholder=placeholder,
        submit_label=submit_label,
        dark=dark,
        default=None,
        key=key,
    )
