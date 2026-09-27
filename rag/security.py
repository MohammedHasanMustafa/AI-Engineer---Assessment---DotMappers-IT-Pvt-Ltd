from __future__ import annotations

import re
import secrets
import unicodedata

from .text_utils import normalize_ws, split_sentences, strip_control

_FLAGS = re.IGNORECASE | re.MULTILINE

INJECTION_PATTERNS: dict[str, re.Pattern] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,60}?\b(instructions?|prompts?|rules|guidelines|directives|(user'?s? )?question)\b", _FLAGS),
    "reveal_system_prompt": re.compile(
        r"\b(reveal|print|show|output|repeat|display|leak|disclose)\b[^.\n]{0,40}?\b(system|hidden|developer|initial|secret)\s+(prompt|instructions?|message|rules)\b", _FLAGS),
    "role_hijack": re.compile(
        r"\b(you are now|from now on,? you|pretend (to be|you are)|(developer|dan|god|jailbreak) mode|act as an? (unrestricted|jailbroken|different|new|evil))\b", _FLAGS),
    "forced_output": re.compile(
        r"\b(respond|reply|answer|output|say)\b[^.\n]{0,20}?\b(only|exactly|just)\b[^.\n]{0,15}?(with\b|[\"“'])", _FLAGS),
    "append_payload": re.compile(
        r"\b(append|add|include|insert)\b[^.\n]{0,60}?\b(to|in|at the end of)\b[^.\n]{0,20}?\b(every|all|each|your)\b[^.\n]{0,10}?\b(answers?|responses?|replies|outputs?)\b", _FLAGS),
    "fake_role_tag": re.compile(
        r"</?\s*(system|assistant|user|instructions?)\s*>|\[\s*(system|inst)\s*\]|^\s*#{2,}\s*(system|instruction)", _FLAGS),
    "authority_claim": re.compile(
        r"\b(system override|admin override|note to (ai|llm|assistant|language model)s?|attention (ai|llm|assistant)s?)\b", _FLAGS),
    "exfiltration": re.compile(
        r"\b(send|post|upload|exfiltrate|forward)\b[^.\n]{0,40}?(https?://|\bapi key\b|\bpassword\b|\bcredentials\b)", _FLAGS),
}


_PAYLOAD_FLAGS = {"forced_output", "append_payload"}
_QUOTED_RE = re.compile(r"[\"“']([^\"”']{3,40})[\"”']")

NEUTRALIZED_MARK = "[REMOVED: suspected embedded instruction ({flags})]"


class QueryRejected(ValueError):
    """Raised for questions that cannot be processed (empty, absurdly long, ...)."""


def scan_for_injection(text: str) -> list[str]:
    return sorted(name for name, pat in INJECTION_PATTERNS.items() if pat.search(text))


def neutralize(text: str) -> tuple[str, int]:
    """Replace instruction-like sentences with an inert marker. Returns (text, n_replaced)."""
    out, n = [], 0
    for sent in split_sentences(text):
        flags = scan_for_injection(sent)
        if flags:
            out.append(NEUTRALIZED_MARK.format(flags=", ".join(flags)))
            n += 1
        else:
            out.append(sent)
    return " ".join(out), n


def extract_payload_markers(text: str) -> set[str]:
    """Quoted strings that injected instructions try to force into the output (e.g. "PWNED")."""
    markers: set[str] = set()
    for sent in split_sentences(text):
        if set(scan_for_injection(sent)) & _PAYLOAD_FLAGS:
            for m in _QUOTED_RE.findall(sent):
                m = m.strip()
                if len(m) >= 4:
                    markers.add(m)
    return markers


def make_canary() -> str:
    return f"CANARY-{secrets.token_hex(6)}"


def escape_for_prompt(text: str) -> str:
    """Stop passages from closing or spoofing the delimiter tags used in the prompt."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def validate_query(query: str | None, max_chars: int, hard_max_chars: int) -> tuple[str, list[str]]:
    """Normalise a user question; truncate long ones, reject empty or absurdly long ones."""
    warnings: list[str] = []
    if query is None:
        raise QueryRejected("The question is empty. Type a question about the indexed documents.")
    q = normalize_ws(strip_control(unicodedata.normalize("NFKC", query)))
    if not q or not re.search(r"\w", q):
        raise QueryRejected("The question is empty. Type a question about the indexed documents.")
    if len(q) > hard_max_chars:
        raise QueryRejected(
            f"The question is {len(q):,} characters long; the limit is {hard_max_chars:,}. "
            "Shorten it to the core question and try again.")
    if len(q) > max_chars:
        cut = q[:max_chars]
        cut = cut[: cut.rfind(" ")] if " " in cut else cut
        warnings.append(f"Question was {len(q):,} characters; only the first {len(cut):,} were used.")
        q = cut
    if scan_for_injection(q):
        warnings.append("The question itself contains instruction-like text; it is answered only from documents.")
    return q, warnings


def check_output(text: str, canary: str, system_prompt: str) -> list[str]:
    """Hard failures that force the whole response to be withheld."""
    events = []
    if canary and canary in text:
        events.append("system prompt leak detected (canary token in output)")
    for line in system_prompt.splitlines():
        line = line.strip()
        if len(line) > 50 and "{" not in line and line in text:
            events.append("system prompt leak detected (verbatim instruction text in output)")
            break
    return events


def contains_marker(text: str, markers: set[str]) -> list[str]:
    low = text.lower()
    return sorted(m for m in markers if m.lower() in low)
