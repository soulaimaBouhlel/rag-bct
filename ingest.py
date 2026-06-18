"""
BCT Regulation Ingestion Pipeline
PDF -> Docling -> Structured Chunks with Metadata
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from docling.document_converter import DocumentConverter


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    circular_ref: str   # e.g. "2026-01"
    circular_type: str  # "banques" | "etablissements" | "unknown"
    chunk_type: str     # "header" | "preamble" | "article" | "signature"
    article_number: Optional[int]
    article_label: str  # "premier", "2", "3" … (raw label from text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_circular_ref(filename: str) -> tuple[str, str]:
    """
    Extract (circular_ref, circular_type) from filename.
    Examples:
      Cir_2026_01_fr.pdf  -> ("2026-01", "banques")
      Note_2024_03_fr.pdf -> ("2024-03", "note")
    """
    stem = Path(filename).stem          # e.g. "Cir_2026_01_fr"
    parts = stem.split("_")
    if len(parts) >= 3:
        year = parts[1]
        number = parts[2].lstrip("0") or "0"
        ref = f"{year}-{number.zfill(2)}"
    else:
        ref = stem

    name_lower = filename.lower()
    if "banque" in name_lower or name_lower.startswith("cir"):
        circ_type = "banques"
    elif "note" in name_lower:
        circ_type = "note"
    else:
        circ_type = "unknown"

    return ref, circ_type


# Article patterns that appear in BCT circulars (French)
ARTICLE_PATTERNS = [
    re.compile(r"^article\s+premier\b", re.IGNORECASE),
    re.compile(r"^article\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"^art\.\s*(\d+)\b", re.IGNORECASE),
]

def detect_article(line: str) -> Optional[tuple[int, str]]:
    """
    Returns (article_number, raw_label) if line starts an article, else None.
    'Article premier' -> (1, "premier")
    'Article 3'       -> (3, "3")
    """
    stripped = line.strip()
    m = re.match(r"^article\s+premier\b", stripped, re.IGNORECASE)
    if m:
        return 1, "premier"
    m = re.match(r"^article\s+(\d+)\b", stripped, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(1)
    m = re.match(r"^art\.\s*(\d+)\b", stripped, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(1)
    return None


SIGNATURE_MARKERS = ["le gouverneur", "le directeur", "le président"]

def is_signature_block(line: str) -> bool:
    low = line.strip().lower()
    return any(low.startswith(m) for m in SIGNATURE_MARKERS)


VU_PATTERN = re.compile(r"^vu\s+", re.IGNORECASE)
DECIDE_PATTERN = re.compile(r"^(décide|decide)\s*:?$", re.IGNORECASE)
OBJET_PATTERN = re.compile(r"^objet\s*:?", re.IGNORECASE)


# ── Core chunker ──────────────────────────────────────────────────────────────

def chunk_markdown(markdown: str, circular_ref: str, circular_type: str,
                   source_file: str) -> list[Chunk]:
    """
    Split a Docling-produced markdown into structured chunks.
    Strategy:
      1. Header block  : everything up to first "Vu …" or "Objet"
      2. Preamble      : all "Vu …" lines grouped together
      3. Articles      : one chunk per article
      4. Signature     : trailing "Le Gouverneur …" block
    """
    lines = markdown.splitlines()
    chunks: list[Chunk] = []

    # ── Pass 1: classify every line ──────────────────────────────────────────
    # States: header | preamble | article | signature
    state = "header"
    current_lines: list[str] = []
    current_article_num: Optional[int] = None
    current_article_label: str = ""
    chunk_counter = 0

    def flush(chunk_type: str, art_num: Optional[int], art_label: str):
        nonlocal chunk_counter
        text = "\n".join(current_lines).strip()
        if not text:
            return
        chunk_counter += 1
        cid = f"{Path(source_file).stem}-{chunk_type}"
        if art_num is not None:
            cid += f"-{art_num}"
        chunks.append(Chunk(
            chunk_id=cid,
            text=text,
            source_file=source_file,
            circular_ref=circular_ref,
            circular_type=circular_type,
            chunk_type=chunk_type,
            article_number=art_num,
            article_label=art_label,
        ))

    for line in lines:
        stripped = line.strip()

        # Detect signature block (always wins)
        if is_signature_block(stripped):
            if state == "article":
                flush("article", current_article_num, current_article_label)
                current_lines = []
            elif state != "signature":
                flush(state, None, "")
                current_lines = []
            state = "signature"
            current_lines.append(line)
            continue

        # Detect article start
        art = detect_article(stripped)
        if art:
            # flush whatever was accumulating
            if state == "article":
                flush("article", current_article_num, current_article_label)
            elif current_lines:
                flush(state, None, "")
            current_lines = [line]
            state = "article"
            current_article_num, current_article_label = art
            continue

        # Detect "Décide :" — marks end of preamble; fold into preamble
        if DECIDE_PATTERN.match(stripped):
            if state == "header":
                flush("header", None, "")
                current_lines = [line]
                state = "preamble"
            else:
                current_lines.append(line)
            continue

        # Detect "Vu …" lines — switch to preamble
        if VU_PATTERN.match(stripped):
            if state == "header":
                flush("header", None, "")
                current_lines = [line]
                state = "preamble"
            else:
                current_lines.append(line)
            continue

        # Default: accumulate into current state
        current_lines.append(line)

    # Flush whatever remains
    if current_lines:
        flush(state, current_article_num if state == "article" else None,
              current_article_label if state == "article" else "")

    return chunks


# ── Per-file processor ────────────────────────────────────────────────────────

converter = DocumentConverter()

def process_pdf(pdf_path: Path) -> list[Chunk]:
    print(f"  Converting {pdf_path.name} …")
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    circular_ref, circular_type = parse_circular_ref(pdf_path.name)
    chunks = chunk_markdown(markdown, circular_ref, circular_type, pdf_path.name)
    print(f"  → {len(chunks)} chunks produced")
    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_all(pdf_dir: str = "pdfs", output_file: str = "chunks.json") -> list[dict]:
    pdf_dir_path = Path(pdf_dir)
    pdfs = sorted(pdf_dir_path.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {pdf_dir}/")
        return []

    all_chunks: list[dict] = []
    for pdf in pdfs:
        chunks = process_pdf(pdf)
        all_chunks.extend([asdict(c) for c in chunks])

    # Persist to JSON for inspection
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Saved to {output_file}")
    return all_chunks


if __name__ == "__main__":
    ingest_all()