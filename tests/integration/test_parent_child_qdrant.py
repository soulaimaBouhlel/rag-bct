import pytest

from src.bct_rag.retrieval.retriever import Retriever


@pytest.fixture
def retriever():

    return Retriever()


def test_parent_retrieval_returns_complete_parent(
    retriever,
):

    question = (
        "What are the dividend distribution "
        "conditions?"
    )

    child_hits = retriever.search(
        question,
        k=5,
        score_threshold=0.0,
    )

    assert child_hits, (
        "Qdrant returned no chunks. "
        "Make sure Qdrant is running and "
        "the collection is indexed."
    )

    hits = [
        {
            "payload": hit.payload,
            "score": hit.score,
        }
        for hit in child_hits
    ]

    parents = retriever.retrieve_parents(
        hits
    )

    assert parents

    for result in parents:

        payload = result["payload"]

        assert "text" in payload
        assert payload["text"]

        assert "chunk_id" in payload
        assert "parent_chunk" in payload