import json
from dataclasses import asdict
from pathlib import Path
from collections import Counter

from docling.document_converter import DocumentConverter

from src.bct_rag.config import PDF_DIR, CHUNKS_FILE
from src.bct_rag.ingestion.chunker import chunk_markdown, parse_circular_ref

converter = DocumentConverter()


def process_pdf(pdf_path: Path) -> list[dict]:
    print(f"  [{pdf_path.name}] converting …")
    result   = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    ref, circ_type = parse_circular_ref(pdf_path.name)
    chunks = chunk_markdown(markdown, ref, circ_type, pdf_path.name)

    # ✅ ADD THIS HERE (validation step)
    for c in chunks:
        if c.token_count > 380:
            raise ValueError(f"Chunk too large: {c.chunk_id} ({c.token_count})")

    from collections import Counter
    type_counts = Counter(c.chunk_type for c in chunks)
    summary = "  ".join(f"{t}:{n}" for t, n in sorted(type_counts.items()))
    max_tokens = max((c.token_count for c in chunks), default=0)

    print(f"  [{pdf_path.name}] → {len(chunks)} chunks  ({summary})  max_tokens={max_tokens}")

    return [asdict(c) for c in chunks]

def run():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    all_chunks = []

    for pdf in pdfs:
        all_chunks.extend(process_pdf(pdf))

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(all_chunks)} chunks saved → {CHUNKS_FILE}")


if __name__ == "__main__":
    run()