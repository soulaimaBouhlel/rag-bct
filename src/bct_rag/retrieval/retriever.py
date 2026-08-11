from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
import os

from src.bct_rag.embedding.embedder import embed

from src.bct_rag.retrieval.parent_child import (
    get_parent_id,
    group_hits_by_parent,
    merge_children,
)

COLLECTION = "regulations"


class Retriever:

    def __init__(self):

        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    def search(
            self,
            question: str,
            k: int = 5,
            score_threshold: float = 0.82,
    ):
        vector = embed(question)

        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=k,
            with_payload=True,
        ).points

        return [
            hit
            for hit in hits
            if hit.score >= score_threshold
        ]

    def get_parent_chunks(
        self,
        parent_id: str,
    ) -> list[dict]:
        """
        Retrieve all child chunks belonging to a parent.

        For unsplit documents, parent_id is the chunk_id,
        so we retrieve the chunk directly.
        """

        # First try the parent_chunk relationship.
        result, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_chunk",
                        match=MatchValue(
                            value=parent_id
                        ),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        children = [
            {
                "payload": point.payload,
                "score": 0.0,
            }
            for point in result
        ]

        # If no explicit parent relationship exists,
        # retrieve the chunk by chunk_id.
        if not children:

            result, _ = self.client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(
                                value=parent_id
                            ),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            children = [
                {
                    "payload": point.payload,
                    "score": 0.0,
                }
                for point in result
            ]

        return children
    def retrieve_parents(
        self,
        hits: list[dict],
    ) -> list[dict]:
        """
        Convert retrieved child chunks into complete parent documents.
        """

        grouped = group_hits_by_parent(hits)

        parents = []

        for parent_id, matched_children in grouped.items():

            all_children = self.get_parent_chunks(
                parent_id
            )

            if not all_children:
                # Safety fallback:
                # keep the originally retrieved children.
                all_children = matched_children

            text = merge_children(all_children)

            best_score = max(
                hit["score"]
                for hit in matched_children
            )

            first_payload = matched_children[0]["payload"]

            first_payload = matched_children[0]["payload"]

            parent_payload = dict(first_payload)

            parent_payload["text"] = text
            parent_payload["parent_chunk"] = parent_id
            parent_payload["num_chunks"] = len(all_children)

            parents.append(
                {
                    "payload": parent_payload,
                    "score": best_score,
                }
            )

        parents.sort(
            key=lambda parent: parent["score"],
            reverse=True,
        )

        return parents

def retrieve(
            question: str,
            k: int = 5,
            score_threshold: float = 0.82,
    ) -> list[dict]:

        retriever = Retriever()

        hits = retriever.search(
            question,
            k=k,
            score_threshold=score_threshold,
        )

        child_hits = [
            {
                "payload": hit.payload,
                "score": hit.score,
            }
            for hit in hits
        ]

        return retriever.retrieve_parents(
            child_hits
        )
def print_results(results):

    for i, hit in enumerate(results, 1):

        p = hit.payload

        print("=" * 70)
        print(f"{i}")
        print(f"Score      : {hit.score:.4f}")
        print(f"Chunk ID   : {p['chunk_id']}")
        print(f"Circular   : {p['circular_ref']}")
        print(f"Type       : {p['chunk_type']}")
        print(f"Article    : {p.get('article_number')}")
        print()

        print(p["text"][:500])