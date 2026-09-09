from src.bct_rag.graph.client import GraphClient


class GraphRetriever:

    def __init__(self):

        self.client = GraphClient()

    def expand_from_chunk(
            self,
            chunk_id: str,
    ):
        """
        Given a Qdrant chunk_id, find its graph entity and expand
        outward 1 hop, in both directions, to related entities.

        Direction is queried explicitly (outgoing vs incoming) rather
        than inferred from node labels afterward — label-based
        inference breaks for Circular -> Circular edges, since both
        sides share the same label and can't be told apart after the
        fact. Two directed queries, anchored on the entity's elementId,
        avoid that ambiguity entirely.
        """

        anchor_query = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (entity)-[:REPRESENTED_BY]->(c)
        RETURN entity, labels(entity)[0] AS entity_type, elementId(entity) AS entity_id
        """

        anchor = self.client.execute(anchor_query, {"chunk_id": chunk_id})

        if not anchor or anchor[0]["entity"] is None:
            return []

        entity = anchor[0]["entity"]
        entity_type = anchor[0]["entity_type"]
        entity_id = anchor[0]["entity_id"]

        outgoing_query = """
        MATCH (entity) WHERE elementId(entity) = $eid
        MATCH (entity)-[rel:REFERENCES|AMENDS|CONTAINS]->(target)
        RETURN type(rel) AS relation_type, target, labels(target)[0] AS target_type
        """

        incoming_query = """
        MATCH (entity) WHERE elementId(entity) = $eid
        MATCH (source)-[rel:REFERENCES|AMENDS|CONTAINS]->(entity)
        RETURN type(rel) AS relation_type, source, labels(source)[0] AS source_type
        """

        outgoing = self.client.execute(outgoing_query, {"eid": entity_id})
        incoming = self.client.execute(incoming_query, {"eid": entity_id})

        results = []

        for row in outgoing:
            results.append({
                "direction": "outgoing",
                "entity": entity,
                "entity_type": entity_type,
                "related": row["target"],
                "related_type": row["target_type"],
                "relation_type": row["relation_type"],
            })

        for row in incoming:
            results.append({
                "direction": "incoming",
                "entity": entity,
                "entity_type": entity_type,
                "related": row["source"],
                "related_type": row["source_type"],
                "relation_type": row["relation_type"],
            })

        return results
    def find_circulars_sharing_laws(self):
        """
        Find pairs of circulars that cite at least one common law,
        grouped by the shared law.
        """

        query = """
        MATCH (c1:Circular)-[:REFERENCES]->(l:Law)<-[:REFERENCES]-(c2:Circular)
        WHERE c1.reference < c2.reference
        RETURN
            l.reference AS shared_law,
            collect(DISTINCT c1.reference + ' & ' + c2.reference) AS circular_pairs
        ORDER BY shared_law
        """

        return self.client.execute(query)

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
    Turn raw graph rows from enrich_hits() into a short, deduplicated
    list of correctly-directed cross-references.
    """

    lines = set()

    for item in enriched_hits:

        for row in item.get("graph_context", []):

            line = _format_edge(row)

            if line:
                lines.add(line)

    return "\n".join(f"- {line}" for line in sorted(lines))


def _describe_node(node_type: str, node: dict) -> str:

    if node_type == "Circular":
        return f"Circular {node.get('reference')}"

    if node_type == "Law":
        return f"Law {node.get('reference')}"

    if node_type == "Article":
        return f"Article {node.get('number')} of {node.get('circular_reference')}"

    if node_type == "Annex":
        return f"Annex {node.get('number')} of {node.get('circular_reference')}"

    return f"{node_type or 'Entity'} {dict(node)}"


def _format_edge(row: dict) -> str | None:
    """
    Direction comes pre-resolved from expand_from_chunk() ("outgoing"
    means entity -> related; "incoming" means related -> entity), so
    this only needs to normalize into (source, target) and render —
    no more guessing direction from node labels.
    """

    entity = row.get("entity")
    related = row.get("related")
    relation_type = row.get("relation_type")
    direction = row.get("direction")

    if not entity or not related or not relation_type:
        return None

    if direction == "outgoing":
        source, source_type = entity, row.get("entity_type")
        target, target_type = related, row.get("related_type")
    else:
        source, source_type = related, row.get("related_type")
        target, target_type = entity, row.get("entity_type")

    verb = {
        "CONTAINS": "contains",
        "REFERENCES": "references",
        "AMENDS": "amends",
    }.get(relation_type, relation_type.lower())

    return f"{_describe_node(source_type, source)} {verb} {_describe_node(target_type, target)}"
