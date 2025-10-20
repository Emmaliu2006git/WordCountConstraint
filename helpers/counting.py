# helpers/counting.py
import re

WORD_RE   = re.compile(r"\b\w+\b", re.UNICODE)
BULLET_RE = re.compile(r"^\s*([\-*\u2022])\s+.+", re.MULTILINE)

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))

def paragraph_count(text: str) -> int:
    if not text:
        return 0
    bullets = BULLET_RE.findall(text)
    if bullets:
        return len(bullets)
    parts = [p for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return len(parts)

def line_count(text: str) -> int:
    if not text:
        return 0
    # Count non-empty lines
    return sum(1 for ln in text.splitlines() if ln.strip())
