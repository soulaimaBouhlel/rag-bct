"""
Production-quality regulation chunker.
Signature detection via post-processing:
- Parse normally into articles, annexes, preambles
- After ingestion, check if articles end with signature pattern
- Split article chunks that contain trailing signatures
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from transformers import AutoTokenizer
from src.bct_rag.graph.reference_extractor import extract_references
# ─────────────────────────────
# TOKENIZER
# ─────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-base"
)

MAX_TOKENS = 360
OVERLAP = 60


def count_tokens(text: str) -> int:
    """Count tokens using embedding model tokenizer."""
    return len(_tokenizer.encode(text, add_special_tokens=False))


def split_tokens(text: str) -> list[str]:
    """Split text into chunks with overlap if > MAX_TOKENS."""
    tokens = _tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= MAX_TOKENS:
        return [text]

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
def strip_self_reference(refs: dict, own_circular_ref: str) -> dict:
    """
    Remove a chunk's citation of its own circular (e.g. a header that
    reads "CIRCULAIRE N° 2026-01" matching as if it referenced circular
    2026-01 — it's naming itself, not citing another document).
    """
    return {
        "laws": refs["laws"],
        "circulars": [c for c in refs["circulars"] if c != own_circular_ref],
        "articles": refs["articles"],
        "annex_number": refs["annex_number"],

    }

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

    parent_chunk: Optional[str]
    chunk_index: int
    num_chunks: int

    references: dict
    annex_number: Optional[str]

# ─────────────────────────────
# SIGNATURE POST-PROCESSING
# ─────────────────────────────

def looks_like_name(text: str) -> bool:
    """
    Check if text looks like a person's name.
    Simple heuristic: 2-3 capitalized words, no punctuation except spaces.
    Examples: "Fethi Zouhaier NOURI", "Jean Dupont", "AHMED BEN SALAH"
    """
    text = text.strip()

    # Remove leading markdown
    text = re.sub(r"^#+\s*", "", text)

    # Should be 2-3 words
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False

    # Each word should be mostly letters (allow one apostrophe/hyphen in names)
    for word in words:
        # Strip markdown bold
        word_clean = word.strip("*").strip()
        # Check if it looks like a name (letters, hyphens, apostrophes only)
        if not re.match(r"^[A-Za-zÀ-ÿ\-']+$", word_clean):
            return False

    return True


def extract_signature_from_text(text: str) -> Optional[tuple[str, str]]:
    """
    Extract signature from end of text with STRICT pattern matching.

    Signature pattern:
    - Max 3 lines
    - Max 100 tokens (real signatures are tiny)
    - First line: ONLY role keyword (Gouverneur, Directeur, Président, etc.)
    - Last line: ONLY person name (2-3 capitalized words)
    - No bullet points, no article markers, no tables

    Returns: (article_body, signature_text) or None if no signature found
    """
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None

    # Check only last 3 lines for signature
    for split_point in range(max(0, len(lines) - 3), len(lines)):
        potential_sig_lines = lines[split_point:]
        potential_sig_text = "\n".join(potential_sig_lines).strip()

        # STRICT: Max 100 tokens (signatures are very short)
        if count_tokens(potential_sig_text) > 100:
            continue

        # STRICT: Max 3 lines
        if len(potential_sig_lines) > 3:
            continue

        # STRICT: Must NOT contain bullet points, tables, or article markers
        lower = potential_sig_text.lower()
        if "-" in potential_sig_text or "|" in lower or "article " in lower or "annexe" in lower:
            continue

        # STRICT: First non-empty line must be ONLY role keyword (with optional comma)
        first_line = potential_sig_lines[0].strip()
        first_clean = re.sub(r"^#+\s*", "", first_line).lower()

        roles = {"gouverneur", "directeur", "président", "vice-gouverneur", "secrétaire", "chef", "délégué"}
        has_role = any(role in first_clean for role in roles)

        if not has_role:
            continue

        # First line should be ONLY role (+ optional comma, markdown)
        # Allow: "## Le Gouverneur,", "Le Gouverneur,", "LE GOUVERNEUR"
        first_normalized = re.sub(r"^#+\s*", "", first_line)
        first_normalized = re.sub(r"^le\s+|^la\s+", "", first_normalized, flags=re.IGNORECASE)
        first_normalized = first_normalized.rstrip(",").strip()

        # Should be just role name
        if not any(role in first_normalized.lower() for role in roles):
            continue

        # STRICT: Last non-empty line must be ONLY a name
        last_line = potential_sig_lines[-1].strip()
        if not looks_like_name(last_line):
            continue

        # Found a real signature! Return split point
        article_body = "\n".join(lines[:split_point]).strip()
        if article_body:  # Only if there's actual article body before signature
            return article_body, potential_sig_text

    return None


def post_process_signatures(chunks: List[Chunk]) -> List[Chunk]:
    """
    Post-process chunks to extract signatures from articles.

    If an article chunk ends with a signature pattern,
    split it into two chunks:
    1. Article body
    2. Signature
    """
    result = []
    stem = None

    for chunk in chunks:
        # Only process article chunks
        if chunk.chunk_type != "article":
            result.append(chunk)
            continue

        # Try to extract signature from this article
        split = extract_signature_from_text(chunk.text)

        if split is None:
            # No signature found, keep as-is
            result.append(chunk)
            continue

        # Signature found! Split the chunk
        article_body, sig_text = split

        stem = Path(chunk.source_file).stem

        # Create article chunk (body only).
        # article_body still starts with the "Article N" heading line
        # (extract_signature_from_text only trims the trailing signature),
        # so strip it before extracting references — same self-reference
        # issue as in flush().
        if chunk.chunk_index == 1:
            refs_source = "\n".join(article_body.split("\n")[1:])
        else:
            refs_source = article_body

        body_refs = extract_references(refs_source)
        body_refs = strip_self_reference(body_refs, chunk.circular_ref)
        article_chunk = Chunk(
            chunk_id=chunk.chunk_id,
            text=article_body,
            source_file=chunk.source_file,
            circular_ref=chunk.circular_ref,
            circular_type=chunk.circular_type,
            title=chunk.title,
            objet=chunk.objet,
            page_hint=chunk.page_hint,
            chunk_type="article",
            chapter=chunk.chapter,
            article_number=chunk.article_number,
            article_label=chunk.article_label,
            token_count=count_tokens(article_body),
            parent_chunk=chunk.parent_chunk,
            chunk_index=chunk.chunk_index,
            num_chunks=chunk.num_chunks,
            references={
                "laws": body_refs["laws"],
                "circulars": body_refs["circulars"],
                "articles": body_refs["articles"],
            },
            annex_number=body_refs["annex_number"],
        )
        result.append(article_chunk)

        # Create signature chunk (signatures never carry regulatory references)
        sig_chunk = Chunk(
            chunk_id=f"{stem}-signature",
            text=sig_text,
            source_file=chunk.source_file,
            circular_ref=chunk.circular_ref,
            circular_type=chunk.circular_type,
            title=chunk.title,
            objet=chunk.objet,
            page_hint=chunk.page_hint,
            chunk_type="signature",
            chapter=None,
            article_number=None,
            article_label="",
            token_count=count_tokens(sig_text),
            parent_chunk=chunk.parent_chunk,
            chunk_index=1,
            num_chunks=1,
            references={"laws": [], "circulars": [], "articles": []},
            annex_number=None,
        )
        result.append(sig_chunk)

    return result


# ─────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────

def extract_title_and_objet(lines: list[str]) -> tuple[str, str]:
    """Extract title and objet."""
    title = ""
    objet = ""
    in_objet = False

    for line in lines[:80]:
        clean = line.strip()

        if not title and "circulaire" in clean.lower():
            title = clean

        if clean.lower().startswith("objet"):
            in_objet = True
            if ":" in clean:
                objet = clean.split(":", 1)[1].strip()
            continue

        if in_objet and clean and not any(
            clean.lower().startswith(x) for x in ["le gouverneur", "vu ", "décide"]
        ):
            objet = clean
            break

    return title, objet


# ─────────────────────────────
# CIRCULAR REFERENCE
# ─────────────────────────────

def parse_circular_ref(filename: str) -> tuple[str, str]:
    """Parse circular reference from filename."""
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 3:
        year = parts[1]
        number = parts[2].lstrip("0") or "0"
        circular_ref = f"{year}-{number.zfill(2)}"
    else:
        circular_ref = stem

    circular_type = "note" if stem.lower().startswith("note") else "banques"

    return circular_ref, circular_type


# ─────────────────────────────
# PATTERN DETECTION
# ─────────────────────────────

def _normalize_line(line: str) -> str:
    """Remove markdown."""
    s = re.sub(r"^#+\s*", "", line.strip())
    s = re.sub(r"\*\*", "", s)
    return s.lower()


def detect_article(line: str) -> Optional[tuple[int, str]]:
    """Detect article heading."""
    clean = _normalize_line(line)
    m = re.match(r"^article\s+(\d+|premier)", clean)
    if not m:
        return None

    label = m.group(1)
    if label == "premier":
        return 1, "premier"

    return int(label), label


def detect_chapter(line: str) -> Optional[str]:
    """Detect chapter heading."""
    clean = _normalize_line(line)
    if re.match(r"^chapitre\s+", clean):
        return line.strip()
    return None


def is_annex(line: str) -> bool:
    """Detect annex heading."""
    return _normalize_line(line).startswith("annexe")


# ─────────────────────────────
# MAIN CHUNKER (simplified - no signature detection)
# ─────────────────────────────

def chunk_markdown(
    markdown: str,
    circular_ref: str,
    circular_type: str,
    source_file: str,
) -> list[Chunk]:
    """
    Chunk markdown (no signature detection).
    Signatures will be extracted via post-processing.
    """
    lines = markdown.splitlines()
    stem = Path(source_file).stem

    title, objet = extract_title_and_objet(lines)

    chunks: list[Chunk] = []
    buffer: list[str] = []

    state = "header"
    current_chapter: Optional[str] = None
    cur_article: Optional[int] = None
    cur_label: str = ""
    annex_counter: int = 0

    def flush(
        chunk_type: str,
        art_num: Optional[int] = None,
        art_label: str = "",
        annex_num: Optional[int] = None,
    ) -> None:
        """Flush buffered lines."""
        nonlocal buffer

        if not buffer:
            return

        text = "\n".join(buffer).strip()
        buffer = []

        if not text:
            return

        # Split if necessary
        if count_tokens(text) > MAX_TOKENS:
            parts = split_tokens(text)
        else:
            parts = [text]

        num_parts = len(parts)

        # Generate parent ID
        if chunk_type == "article" and art_num is not None:
            parent_id = f"{stem}-article-{art_num}"
        elif chunk_type == "annex":
            parent_id = f"{stem}-annex-{annex_num}" if annex_num else f"{stem}-annex"
        else:
            parent_id = f"{stem}-{chunk_type}"

        # Create chunks
        for split_idx, part in enumerate(parts, start=1):
            if num_parts > 1:
                cid = f"{parent_id}-{split_idx}"
            else:
                cid = parent_id

            if chunk_type == "article" and split_idx == 1:
                # Exclude the "Article N" heading line itself, so the
                # chunk doesn't register a reference to its own article
                # number (e.g. "Article 1" heading matching as if the
                # body referenced "article 1" elsewhere).
                body_for_refs = "\n".join(part.split("\n")[1:])
            else:
                body_for_refs = part

            refs = extract_references(body_for_refs)
            refs = strip_self_reference(refs, circular_ref)

            # Annex numbers come from document order (annex_num, set by
            # the caller), not from parsing the heading text: OCR'd annex
            # headings are frequently garbled or missing a number
            # entirely, so a structural counter is far more reliable.
            if chunk_type == "annex":
                resolved_annex_number = str(annex_num) if annex_num is not None else None
            else:
                resolved_annex_number = refs["annex_number"]

            chunks.append(Chunk(
                chunk_id=cid,
                text=part,
                source_file=source_file,
                circular_ref=circular_ref,
                circular_type=circular_type,
                title=title,
                objet=objet,
                page_hint=0,
                chunk_type=chunk_type,
                chapter=current_chapter if chunk_type == "article" else None,
                article_number=art_num,
                article_label=art_label,
                token_count=count_tokens(part),
                parent_chunk=parent_id,
                chunk_index=split_idx,
                num_chunks=num_parts,
                references={
                    "laws": refs["laws"],
                    "circulars": refs["circulars"],
                    "articles": refs["articles"],
                },
                annex_number=resolved_annex_number,
            ))

    # ─────────────────────────────
    # MAIN LOOP (simplified)
    # ─────────────────────────────

    for i, line in enumerate(lines):
        clean = line.strip()

        if not clean:
            buffer.append(line)
            continue

        # CHAPTER DETECTION
        ch = detect_chapter(line)
        if ch:
            current_chapter = ch
            buffer.append(line)
            continue

        # ARTICLE DETECTION
        art = detect_article(line)
        if art:
            flush(state, cur_article, cur_label, annex_counter)
            state = "article"
            buffer = [line]
            cur_article, cur_label = art
            continue

        # ANNEX DETECTION
        if is_annex(line):
            flush(state, cur_article, cur_label, annex_counter)
            state = "annex"
            annex_counter += 1
            buffer = [line]
            cur_article = None
            cur_label = ""
            continue
        # PREAMBLE DETECTION
        if state == "header":
            if clean.lower().startswith("vu ") or clean.lower().startswith("décide"):
                flush("header")
                state = "preamble"
                buffer = [line]
                continue

        # Accumulate (no signature detection in main loop)
        buffer.append(line)

    # Flush remainder
    flush(state, cur_article, cur_label, annex_counter)

    # ─────────────────────────────
    # POST-PROCESSING: Extract signatures
    # ─────────────────────────────
    chunks = post_process_signatures(chunks)

    return chunks