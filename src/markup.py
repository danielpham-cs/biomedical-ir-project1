"""Preserve the small subset of XML typography needed in biomedical text."""

from __future__ import annotations

import re
from html import escape
from xml.etree import ElementTree as ET

_INLINE_TAGS = frozenset({"sup", "sub"})
_MARKUP = re.compile(r"</?(?:sup|sub)>", re.IGNORECASE)


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def xml_text(element: ET.Element | None) -> str:
    """Read XML text while retaining only safe superscript/subscript tags."""
    if element is None:
        return ""
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        tag = _local(node.tag).lower()
        formatted = tag in _INLINE_TAGS
        if formatted:
            parts.append(f"<{tag}>")
        if node.text:
            parts.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                parts.append(child.tail)
        if formatted:
            parts.append(f"</{tag}>")

    visit(element)
    return " ".join("".join(parts).split())


def plain_text(text: str) -> str:
    """Remove the preserved presentation tags for search and statistics."""
    return _MARKUP.sub("", text or "")


def render_inline_markup(text: str) -> str:
    """Escape article text, then restore the two safe tags emitted by ``xml_text``."""
    safe = escape(text or "")
    safe = re.sub(r"&lt;(sup|sub)&gt;", r"<\1>", safe, flags=re.IGNORECASE)
    return re.sub(r"&lt;/(sup|sub)&gt;", r"</\1>", safe, flags=re.IGNORECASE)
