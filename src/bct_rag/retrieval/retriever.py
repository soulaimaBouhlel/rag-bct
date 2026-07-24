from qdrant_client import QdrantClient
import os

from src.bct_rag.embedding.embedder import embed


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

    return [
        {
            "payload": hit.payload,
            "score": hit.score,
        }
        for hit in hits
    ]
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