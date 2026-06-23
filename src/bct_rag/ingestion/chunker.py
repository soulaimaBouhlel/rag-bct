import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from transformers import AutoTokenizer

# ─────────────────────────────
# TOKENIZER
# ─────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_TOKENS = 360
OVERLAP = 60


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def split_tokens(text: str):
    tokens = _tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        part = _tokenizer.decode(tokens[start:end])
        chunks.append(part)

        if end == len(tokens):
            break

        start = end - OVERLAP

    return chunks


# ─────────────────────────────
# DATA MODEL
# ─────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    circular_ref: str
    circular_type: str

    title: str
    objet: str
    page_hint: int

    chunk_type: str
    chapter: Optional[str]
    article_number: Optional[int]
    article_label: str

    token_count: int


# ─────────────────────────────
# ARTICLE DETECTION (FIXED)
# ─────────────────────────────

_ARTICLE = re.compile(r"article\s*(\d+|premier)", re.IGNORECASE)

def detect_article(line: str):
    clean = re.sub(r"^#+\s*", "", line.strip().lower())
    clean = re.sub(r"\*\*", "", clean)

    m = _ARTICLE.search(clean)
    if not m:
        return None

    if m.group(1) == "premier":
        return 1, "premier"

    return int(m.group(1)), m.group(1)


# ─────────────────────────────
# SIGNATURE DETECTION (ROBUST)
# ─────────────────────────────

def is_signature(lines: List[str]) -> bool:
    text = "\n".join(lines).strip()
    if count_tokens(text) > 120:
        return False

    lower = text.lower()

    # structural heuristics
    score = 0

    if len(lines) <= 5:
        score += 1

    if any("gouverneur" in l.lower() for l in lines):
        score += 2

    if any(len(l.split()) <= 6 for l in lines):
        score += 1

    return score >= 2


# ─────────────────────────────
# MAIN CHUNKER
# ─────────────────────────────

def chunk_markdown(
    markdown: str,
    circular_ref: str,
    circular_type: str,
    source_file: str,
    title: str = "",
    objet: str = "",
):

    lines = markdown.splitlines()

    chunks = []
    buffer = []

    state = "doc"

    cur_article = None
    cur_label = ""

    def flush(chunk_type="doc", art_num=None, art_label=""):

        nonlocal buffer

        if not buffer:
            return

        text = "\n".join(buffer).strip()
        buffer = []

        if not text:
            return

        # TOKEN SAFETY (critical)
        if count_tokens(text) > MAX_TOKENS:
            parts = split_tokens(text)
        else:
            parts = [text]

        for part in parts:
            chunks.append(Chunk(
                chunk_id=f"{Path(source_file).stem}-{chunk_type}",
                text=part,
                source_file=source_file,
                circular_ref=circular_ref,
                circular_type=circular_type,
                title=title,
                objet=objet,
                page_hint=0,
                chunk_type=chunk_type,
                chapter=None,
                article_number=art_num,
                article_label=art_label,
                token_count=count_tokens(part),
            ))

    # ─────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────

    for i, line in enumerate(lines):

        clean = line.strip()

        # ARTICLE DETECTION (FIXED)
        art = detect_article(clean)
        if art:
            flush(state, cur_article, cur_label)
            state = "article"
            buffer = [line]
            cur_article, cur_label = art
            continue

        # SIGNATURE DETECTION (end heuristic)
        if i > len(lines) * 0.8:
            if is_signature(lines[i:]):
                flush(state, cur_article, cur_label)
                state = "signature"
                buffer = lines[i:]
                break

        buffer.append(line)

    flush(state, cur_article, cur_label)

    return chunks


def parse_circular_ref(filename: str):
    stem = Path(filename).stem
    return stem, "banques"