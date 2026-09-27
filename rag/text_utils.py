from __future__ import annotations

import re

_SENT_RE = re.compile(r"(?<=[.!?>])\s+(?=[\"'(\[<#]?[A-Z0-9<])")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

STOPWORDS = frozenset(
    """a an the of to in on for and or is are was were be been by with as at from that this these
    those which what how why who whom when where does do did it its into than then there their about
    can could should would will may might according knowledge document documents""".split()
)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (good enough for technical prose, no model download)."""
    out: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if block:
            out.extend(s.strip() for s in _SENT_RE.split(block) if s.strip())
    return out


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens (keeps numbers like 28.4 and terms like l2-normalized)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_control(text: str) -> str:
    return _CTRL_RE.sub("", text)
