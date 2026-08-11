from src.bct_rag.retrieval.retriever import Retriever


QUESTION = (
    "What are the dividend distribution conditions?"
)


retriever = Retriever()


hits = retriever.search(
    QUESTION,
    k=5,
    score_threshold=0.0,
)


print("=" * 70)
print("CHILD RETRIEVAL")
print("=" * 70)


for hit in hits:

    payload = hit.payload

    print(
        payload.get("chunk_id"),
        "score=",
        round(hit.score, 4),
        "parent=",
        payload.get("parent_chunk"),
        "tokens=",
        payload.get("token_count"),
    )


child_hits = [
    {
        "payload": hit.payload,
        "score": hit.score,
    }
    for hit in hits
]


parents = retriever.retrieve_parents(
    child_hits
)


print("\n" + "=" * 70)
print("PARENT RETRIEVAL")
print("=" * 70)


for parent in parents:

    payload = parent["payload"]

    print(
        payload.get("parent_chunk"),
        "score=",
        round(parent["score"], 4),
        "chunks=",
        payload.get("num_chunks"),
        "characters=",
        len(payload.get("text", "")),
    )