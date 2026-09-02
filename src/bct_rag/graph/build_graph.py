import json

from src.bct_rag.graph.client import GraphClient
from src.bct_rag.graph.builder import GraphBuilder


CHUNKS_PATH = "data/chunks.json"


def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def infer_year(reference: str) -> int:
    """
    Pull the year out of a canonical reference like "2016-35" -> 2016.
    """

    return int(reference.split("-")[0])


def build_pass_1_nodes(chunks: list[dict], builder: GraphBuilder) -> dict:
    """
    Create every Circular, Law, Article, and Annex node.

    Circulars are created first and unconditionally, before any
    Article/Annex — create_article()/create_annex() silently create
    nothing if their parent Circular doesn't exist yet (see builder.py),
    so ordering here is load-bearing, not cosmetic.
    """

    stats = {
        "circulars": 0,
        "laws": 0,
        "articles": 0,
        "annexes": 0,
    }

    # 1a. All circulars first.
    seen_circulars = set()

    for chunk in chunks:

        circular_ref = chunk.get("circular_ref")

        if not circular_ref or circular_ref in seen_circulars:
            continue

        builder.create_circular(
            circular_ref,
            title=chunk.get("title"),
            year=infer_year(circular_ref),
        )

        seen_circulars.add(circular_ref)
        stats["circulars"] += 1

    # 1b. All laws referenced anywhere.
    seen_laws = set()

    for chunk in chunks:

        for law_ref in chunk.get("references", {}).get("laws", []):

            if law_ref in seen_laws:
                continue

            builder.create_law(
                law_ref,
                year=infer_year(law_ref),
            )

            seen_laws.add(law_ref)
            stats["laws"] += 1

    # 1c. All articles.
    for chunk in chunks:

        if chunk.get("chunk_type") != "article":
            continue

        circular_ref = chunk.get("circular_ref")
        article_number = chunk.get("article_number")

        if not circular_ref or article_number is None:
            continue

        result = builder.create_article(circular_ref, article_number)

        if result:
            stats["articles"] += 1

    # 1d. All annexes.
    for chunk in chunks:

        if chunk.get("chunk_type") != "annex":
            continue

        circular_ref = chunk.get("circular_ref")
        annex_number = chunk.get("annex_number")

        if not circular_ref or annex_number is None:
            continue

        result = builder.create_annex(circular_ref, annex_number)

        if result:
            stats["annexes"] += 1

    return stats


def build_pass_2_relationships(chunks: list[dict], builder: GraphBuilder) -> dict:
    """
    Create all Circular -> Law and Circular -> Circular REFERENCES edges.

    Every node type was created in pass 1, so any skip here reflects a
    genuine data issue (e.g. a referenced circular that was never
    ingested) rather than an ordering problem.
    """

    stats = {
        "circular_to_law_ok": 0,
        "circular_to_law_skipped": 0,
        "circular_to_circular_ok": 0,
        "circular_to_circular_skipped": 0,
    }

    seen_law_edges = set()
    seen_circular_edges = set()

    for chunk in chunks:

        circular_ref = chunk.get("circular_ref")

        if not circular_ref:
            continue

        refs = chunk.get("references", {})

        for law_ref in refs.get("laws", []):

            edge = (circular_ref, law_ref)

            if edge in seen_law_edges:
                continue

            seen_law_edges.add(edge)

            result = builder.link_circular_to_law(circular_ref, law_ref)
            linked = result[0]["linked"] if result else 0

            if linked:
                stats["circular_to_law_ok"] += 1
            else:
                stats["circular_to_law_skipped"] += 1
                print(f"  ⚠ skipped: {circular_ref} -> law {law_ref} (law not found)")

        for other_circular in refs.get("circulars", []):

            edge = (circular_ref, other_circular)

            if edge in seen_circular_edges:
                continue

            seen_circular_edges.add(edge)

            result = builder.link_circular_to_circular(circular_ref, other_circular)
            linked = result[0]["linked"] if result else 0

            if linked:
                stats["circular_to_circular_ok"] += 1
            else:
                stats["circular_to_circular_skipped"] += 1
                print(f"  ⚠ skipped: {circular_ref} -> circular {other_circular} (circular not found — likely an older/unindexed document)")

    return stats

def build_pass_3_chunk_links(chunks: list[dict], builder: GraphBuilder) -> dict:
    """
    Link chunks to their graph entity via REPRESENTED_BY.

    - article/annex chunks -> their Article/Annex node
    - header/preamble chunks -> the Circular itself (this is where
      most law/circular citations actually live, so the Circular
      needs a direct link to them for graph expansion to find them)

    Must run after pass 1 (nodes must exist).
    """

    stats = {
        "article_chunks_ok": 0,
        "article_chunks_skipped": 0,
        "annex_chunks_ok": 0,
        "annex_chunks_skipped": 0,
        "circular_chunks_ok": 0,
        "circular_chunks_skipped": 0,
    }

    for chunk in chunks:

        chunk_type = chunk.get("chunk_type")
        circular_ref = chunk.get("circular_ref")
        chunk_id = chunk.get("chunk_id")

        if not circular_ref or not chunk_id:
            continue

        if chunk_type == "article":

            article_number = chunk.get("article_number")

            if article_number is None:
                continue

            result = builder.link_article_chunk(circular_ref, article_number, chunk_id)
            linked = result[0]["linked"] if result else 0

            if linked:
                stats["article_chunks_ok"] += 1
            else:
                stats["article_chunks_skipped"] += 1
                print(f"  ⚠ skipped: chunk {chunk_id} -> article {article_number} of {circular_ref} (article node not found)")

        elif chunk_type == "annex":

            annex_number = chunk.get("annex_number")

            if annex_number is None:
                continue

            result = builder.link_annex_chunk(circular_ref, annex_number, chunk_id)
            linked = result[0]["linked"] if result else 0

            if linked:
                stats["annex_chunks_ok"] += 1
            else:
                stats["annex_chunks_skipped"] += 1
                print(f"  ⚠ skipped: chunk {chunk_id} -> annex {annex_number} of {circular_ref} (annex node not found)")

        elif chunk_type in ("header", "preamble"):

            result = builder.link_circular_chunk(circular_ref, chunk_id)
            linked = result[0]["linked"] if result else 0

            if linked:
                stats["circular_chunks_ok"] += 1
            else:
                stats["circular_chunks_skipped"] += 1
                print(f"  ⚠ skipped: chunk {chunk_id} -> circular {circular_ref} (circular node not found)")

    return stats

def main():

    chunks = load_chunks()

    client = GraphClient()
    builder = GraphBuilder(client)

    print(f"\nLoaded {len(chunks)} chunks from {CHUNKS_PATH}\n")

    print("Pass 1: creating nodes...")
    node_stats = build_pass_1_nodes(chunks, builder)
    print(f"  circulars: {node_stats['circulars']}")
    print(f"  laws:      {node_stats['laws']}")
    print(f"  articles:  {node_stats['articles']}")
    print(f"  annexes:   {node_stats['annexes']}")

    print("\nPass 2: creating relationships...")
    rel_stats = build_pass_2_relationships(chunks, builder)
    print(f"  circular->law:      {rel_stats['circular_to_law_ok']} ok, {rel_stats['circular_to_law_skipped']} skipped")
    print(f"  circular->circular: {rel_stats['circular_to_circular_ok']} ok, {rel_stats['circular_to_circular_skipped']} skipped")

    print("\nPass 3: linking chunks to graph entities...")
    chunk_stats = build_pass_3_chunk_links(chunks, builder)
    print(f"  article chunks:   {chunk_stats['article_chunks_ok']} ok, {chunk_stats['article_chunks_skipped']} skipped")
    print(f"  annex chunks:     {chunk_stats['annex_chunks_ok']} ok, {chunk_stats['annex_chunks_skipped']} skipped")
    print(f"  circular chunks:  {chunk_stats['circular_chunks_ok']} ok, {chunk_stats['circular_chunks_skipped']} skipped")
    client.close()

    print(f"\n✓ Graph built from {len(chunks)} chunks.")


if __name__ == "__main__":
    main()