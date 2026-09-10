import json
from dataclasses import asdict
from pathlib import Path
from collections import Counter

from docling.document_converter import DocumentConverter

from src.bct_rag.config import PDF_DIR, CHUNKS_FILE
from src.bct_rag.ingestion.chunker import chunk_markdown, parse_circular_ref

converter = DocumentConverter()

import re

LAW_DIFF_PATTERN = re.compile(
    r"(?:base|fonde).{0,40}loi\s+n?°?\s*(\d{4}-\d{2,4}).{0,40}(?:mais\s+pas|sans|non)\s+.{0,40}loi\s+n?°?\s*(\d{4}-\d{2,4})",
    re.IGNORECASE,
)


def _try_law_diff_query(question: str):
    """
    Detect 'which circulars cite law A but not law B' style questions
    and answer them directly from the graph — this question class
    depends on negation/set-difference logic across the whole corpus,
    which chunk-retrieval + generation cannot reliably answer (the
    correct source document may not even be in the vector top-k, and
    small local models don't reliably follow an abstention instruction
    when surface-level co-occurrence looks plausible).
    """

    match = LAW_DIFF_PATTERN.search(question)

    if not match:
        return None

    law_a, law_b = match.group(1), match.group(2)

    from src.bct_rag.retrieval.graph_retriever import GraphRetriever

    gr = GraphRetriever()
    rows = gr.find_circulars_citing_but_not(law_a, law_b)
    gr.close()

    if not rows:
        return (
            f"Aucune circulaire ne se base sur la loi n°{law_a} sans se baser également sur la loi n°{law_b}."
        )

    circulars = ", ".join(row["circular_reference"] for row in rows)

    return f"Circulaire(s) se basant sur la loi n°{law_a} mais pas sur la loi n°{law_b} : {circulars}"
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