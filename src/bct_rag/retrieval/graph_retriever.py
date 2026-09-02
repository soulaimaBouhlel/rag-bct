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
            related,
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