from src.bct_rag.graph.client import GraphClient


class GraphBuilder:

    def __init__(self, client: GraphClient):

        self.client = client

    def create_circular(
        self,
        circular_ref: str,
        title: str | None = None,
        year: int | None = None,
    ):
        """
        Create (or update) a Circular node, keyed by its canonical
        reference (e.g. "2026-01").
        """

        query = """
        MERGE (c:Circular {reference: $reference})
        SET
            c.title = $title,
            c.year = $year

        RETURN c
        """

        return self.client.execute(
            query,
            {
                "reference": circular_ref,
                "title": title,
                "year": year,
            },
        )

    def create_law(
        self,
        reference: str,
        year: int,
    ):
        """
        Create (or update) a Law node, keyed by its canonical
        reference (e.g. "2016-35").
        """

        query = """
        MERGE (l:Law {reference: $reference})
        SET l.year = $year

        RETURN l
        """

        return self.client.execute(
            query,
            {
                "reference": reference,
                "year": year,
            },
        )

    def create_article(
        self,
        circular_ref: str,
        article_number: int,
    ):
        """
        Create (or update) an Article node scoped to its parent Circular,
        and link Circular -[:CONTAINS]-> Article.

        Returns [] if the Circular doesn't exist yet — caller must
        create all Circulars before any Article/Annex (see build_graph.py).
        """

        query = """
        MATCH (c:Circular {reference: $circular})

        MERGE (
            a:Article {
                circular_reference: $circular,
                number: $number
            }
        )

        MERGE (c)-[:CONTAINS]->(a)

        RETURN a
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "number": article_number,
            },
        )

    def create_annex(
        self,
        circular_ref: str,
        annex_number: str,
    ):
        """
        Create (or update) an Annex node scoped to its parent Circular,
        and link Circular -[:CONTAINS]-> Annex.

        Returns [] if the Circular doesn't exist yet — same ordering
        requirement as create_article.
        """

        query = """
        MATCH (c:Circular {reference: $circular})

        MERGE (
            a:Annex {
                circular_reference: $circular,
                number: $number
            }
        )

        MERGE (c)-[:CONTAINS]->(a)

        RETURN a
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "number": annex_number,
            },
        )

    def link_circular_to_law(
        self,
        circular_ref: str,
        law_ref: str,
    ):
        """
        Circular -[:REFERENCES]-> Law.

        Returns [] (not an error) if either node doesn't exist yet —
        both must already be created first.
        """

        query = """
        MATCH (c:Circular {reference: $circular})
        MATCH (l:Law {reference: $law})

        MERGE (c)-[:REFERENCES]->(l)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "law": law_ref,
            },
        )

    def link_circular_to_circular(
        self,
        source: str,
        target: str,
    ):
        """
        Circular -[:REFERENCES]-> Circular (source cites target).

        Returns [] (not an error) if either circular doesn't exist yet.
        """

        query = """
        MATCH (source:Circular {reference: $source})
        MATCH (target:Circular {reference: $target})

        MERGE (source)-[:REFERENCES]->(target)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "source": source,
                "target": target,
            },
        )

    def link_amends(
        self,
        source: str,
        target: str,
    ):
        """
        Circular -[:AMENDS]-> Circular (source amends target).

        Coarse version: circular-level only. A more precise version
        (Circular -[:AMENDS]-> Article -[:PART_OF]-> Circular) can be
        added later once we're confident in the basic graph.

        Returns [] (not an error) if either circular doesn't exist yet.
        """

        query = """
        MATCH (source:Circular {reference: $source})
        MATCH (target:Circular {reference: $target})

        MERGE (source)-[:AMENDS]->(target)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "source": source,
                "target": target,
            },
        )
    def link_article_chunk(
        self,
        circular_ref: str,
        article_number: int,
        chunk_id: str,
    ):
        """
        Article -[:REPRESENTED_BY]-> Chunk.

        Creates the Chunk node (keyed by its Qdrant chunk_id) if it
        doesn't exist yet. Returns [] / linked=0 if the Article node
        doesn't exist — should not happen if pass 1 ran first.
        """

        query = """
        MATCH (a:Article {
            circular_reference: $circular,
            number: $number
        })

        MERGE (c:Chunk {id: $chunk_id})

        MERGE (a)-[:REPRESENTED_BY]->(c)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "number": article_number,
                "chunk_id": chunk_id,
            },
        )

    def link_annex_chunk(
        self,
        circular_ref: str,
        annex_number: str,
        chunk_id: str,
    ):
        """
        Annex -[:REPRESENTED_BY]-> Chunk.

        Same pattern as link_article_chunk, for Annex nodes.
        """

        query = """
        MATCH (a:Annex {
            circular_reference: $circular,
            number: $number
        })

        MERGE (c:Chunk {id: $chunk_id})

        MERGE (a)-[:REPRESENTED_BY]->(c)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "number": annex_number,
                "chunk_id": chunk_id,
            },
        )
    def link_circular_chunk(
        self,
        circular_ref: str,
        chunk_id: str,
    ):
        """
        Circular -[:REPRESENTED_BY]-> Chunk.

        For header/preamble chunks — these aren't Article/Annex
        entities, but they're where most law/circular citations
        actually live, so the Circular itself needs a direct link to
        them for expand_from_chunk() to surface that context.
        """

        query = """
        MATCH (c:Circular {reference: $circular})

        MERGE (chunk:Chunk {id: $chunk_id})

        MERGE (c)-[:REPRESENTED_BY]->(chunk)

        RETURN count(*) AS linked
        """

        return self.client.execute(
            query,
            {
                "circular": circular_ref,
                "chunk_id": chunk_id,
            },
        )