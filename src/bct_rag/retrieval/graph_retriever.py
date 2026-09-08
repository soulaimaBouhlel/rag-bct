from src.bct_rag.graph.client import GraphClient


class GraphRetriever:

    def __init__(self):

        self.client = GraphClient()

    def expand_from_chunk(
        self,
        chunk_id: str,
    ):
        """
        Given a Qdrant chunk_id, find its graph entity (Article/Annex)
        and expand outward 1 hop to related entities (laws it cites,
        circulars it references/is referenced by, sibling articles in
        the same circular).

        Kept shallow (1 hop) deliberately — see Step 27's note on why
        unbounded traversal is dangerous here.
        """

        query = """
        MATCH (c:Chunk {id: $chunk_id})

        OPTIONAL MATCH (entity)-[:REPRESENTED_BY]->(c)

        OPTIONAL MATCH (entity)-[rel:REFERENCES|AMENDS|CONTAINS]-(related)

        RETURN
            entity,
            labels(entity)[0] AS entity_type,
            related,
            labels(related)[0] AS related_type,
            type(rel) AS relation_type
        """
        return self.client.execute(
            query,
            {
                "chunk_id": chunk_id,
            },
        )

    def close(self):

        self.client.close()

def enrich_hits(hits: list[dict]) -> list[dict]:
        """
        Attach graph context to a list of vector-search hits.

        Each hit is expected to look like {"payload": {...}, "score": float}
        (the shape returned by Retriever.search() / retrieve()).
        """

        graph = GraphRetriever()

        enriched = []

        for hit in hits:
            chunk_id = hit["payload"]["chunk_id"]

            graph_data = graph.expand_from_chunk(chunk_id)

            enriched.append(
                {
                    "vector_result": hit,
                    "graph_context": graph_data,
                }
            )

        graph.close()

        return enriched
def format_graph_context(enriched_hits: list[dict]) -> str:
    """
    Turn raw graph rows from enrich_hits() into a short, deduplicated,
    correctly-directed list of cross-references.

    Direction is inferred from node labels (CONTAINS is always
    Circular -> Article/Annex; REFERENCES to a Law is always
    Circular -> Law) rather than trusting which node the undirected
    Cypher match happened to bind as "entity" — that binding varies
    depending on which chunk anchored the traversal and does not
    reflect the true relationship direction.
    """

    lines = set()

    for item in enriched_hits:

        for row in item.get("graph_context", []):

            line = _format_edge(
                row.get("entity_type"),
                row.get("entity"),
                row.get("related_type"),
                row.get("related"),
                row.get("relation_type"),
            )

            if line:
                lines.add(line)

    return "\n".join(f"- {line}" for line in sorted(lines))


def _format_edge(entity_type, entity, related_type, related, relation_type) -> str | None:

    if not entity or not related or not relation_type:
        return None

    nodes = {entity_type: entity, related_type: related}

    if relation_type == "CONTAINS":

        circular = nodes.get("Circular")
        child_type = "Article" if "Article" in nodes else ("Annex" if "Annex" in nodes else None)

        if not circular or not child_type:
            return None

        child = nodes[child_type]
        return f"Circular {circular.get('reference')} contains {child_type} {child.get('number')} of {child.get('circular_reference')}"

    if relation_type == "REFERENCES":

        if "Law" in nodes:
            circular = nodes.get("Circular")
            law = nodes["Law"]

            if not circular:
                return None

            return f"Circular {circular.get('reference')} references Law {law.get('reference')}"

        # Circular -> Circular REFERENCES: both sides share the same
        # label, so direction can't be inferred from labels alone with
        # the current undirected query. No such edges exist yet in this
        # corpus (all 7 attempts pointed at unindexed older circulars),
        # so this is skipped for now rather than risk showing it backwards.
        return None

    # AMENDS: same ambiguity as circular-to-circular REFERENCES, and no
    # AMENDS edges exist yet (link_amends is never called from
    # build_graph.py). Skipped until that's addressed.
    return None