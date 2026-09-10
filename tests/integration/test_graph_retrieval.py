"""
Integration tests for the Phase 2 GraphRAG pipeline.

These tests run against the REAL Qdrant + Neo4j instances (docker
compose up -d qdrant neo4j) and the real ingested chunks.json — they
are not isolated unit tests with mocks. Run with:

    PYTHONPATH=. pytest tests/integration/test_graph_retrieval.py -v

Graph structure tests assert against the specific counts verified
during development (9 circulars, 10 laws, etc.) — if you re-ingest
new PDFs, these counts will legitimately need updating.
"""

import pytest

from src.bct_rag.graph.client import GraphClient
from src.bct_rag.retrieval.graph_retriever import GraphRetriever
from src.bct_rag.retrieval.retriever import retrieve, graph_enhanced_retrieve
from src.bct_rag.pipeline import ask


# ─────────────────────────────
# FIXTURES
# ─────────────────────────────

@pytest.fixture(scope="module")
def graph_client():
    client = GraphClient()
    yield client
    client.close()


@pytest.fixture(scope="module")
def graph_retriever():
    gr = GraphRetriever()
    yield gr
    gr.close()


# ─────────────────────────────
# GRAPH STRUCTURE TESTS
# ─────────────────────────────

def test_node_counts(graph_client):
    """
    Confirms the graph was built with the expected node counts,
    verified manually during Phase 2B development.
    """

    counts = {
        row["type"]: row["count"]
        for row in graph_client.execute(
            "MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count"
        )
    }

    assert counts.get("Circular") == 9
    assert counts.get("Law") == 10
    assert counts.get("Article") == 37
    assert counts.get("Annex") == 4
    assert counts.get("Chunk") == 121


def test_relationship_counts(graph_client):

    counts = {
        row["rel"]: row["count"]
        for row in graph_client.execute(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count"
        )
    }

    assert counts.get("CONTAINS") == 41
    assert counts.get("REPRESENTED_BY") == 121
    assert counts.get("REFERENCES") == 19
    # AMENDS: confirmed zero on this corpus (no active amendment
    # clauses in the 8 ingested documents) — this asserts the
    # machinery is inert, not broken, on real data.
    assert counts.get("AMENDS") is None


def test_no_self_referencing_circular(graph_client):
    """
    Regression test: a circular must never REFERENCES itself (the
    header self-citation bug fixed early in Phase 2A).
    """

    result = graph_client.execute(
        "MATCH (c:Circular)-[:REFERENCES]->(c) RETURN c"
    )

    assert result == []


def test_law_2016_35_cited_by_all_circulars(graph_client):
    """
    Regression test for the shared-law finding: Law 2016-35 (the BCT
    statute) is cited by all 6 real circulars/notes in the corpus.
    """

    result = graph_client.execute(
        """
        MATCH (c:Circular)-[:REFERENCES]->(:Law {reference: "2016-35"})
        RETURN count(c) AS n
        """
    )

    assert result[0]["n"] == 6


def test_circular_2026_43_cites_2016_35_but_not_2016_48(graph_client):
    """
    Regression test for the law-diff finding that originally exposed
    the hallucination bug: 2026-43 (a Note, not a Circulaire) is the
    one document citing 2016-35 without also citing 2016-48.
    """

    result = graph_client.execute(
        """
        MATCH (c:Circular {reference: "2026-43"})-[:REFERENCES]->(:Law {reference: "2016-35"})
        WHERE NOT (c)-[:REFERENCES]->(:Law {reference: "2016-48"})
        RETURN c
        """
    )

    assert len(result) == 1


# ─────────────────────────────
# GRAPH EXPANSION TESTS
# ─────────────────────────────

def test_expand_from_article_chunk_finds_parent_circular(graph_retriever):

    results = graph_retriever.expand_from_chunk("Cir_2026_01_fr-article-1")

    assert any(
        row["relation_type"] == "CONTAINS" and row["related"]["reference"] == "2026-01"
        for row in results
    )


def test_expand_from_preamble_chunk_finds_laws(graph_retriever):
    """
    Regression test for the preamble-gap fix: preamble chunks must be
    linked (via REPRESENTED_BY on the Circular itself) so expansion
    surfaces the laws that circular cites.
    """

    results = graph_retriever.expand_from_chunk("Cir_2026_01_fr-preamble")

    referenced_laws = {
        row["related"]["reference"]
        for row in results
        if row["relation_type"] == "REFERENCES" and row["related_type"] == "Law"
    }

    assert referenced_laws == {"2016-35", "2016-48"}


def test_expand_from_unlinked_chunk_returns_empty(graph_retriever):
    """
    A chunk_id that doesn't exist should return an empty expansion,
    not raise an error.
    """

    results = graph_retriever.expand_from_chunk("does-not-exist")

    assert results == []


# ─────────────────────────────
# RETRIEVAL REGRESSION TESTS
# ─────────────────────────────

def test_vector_retrieval_still_works():
    """
    Core Phase 1 regression check: graph additions must never change
    plain vector retrieval behavior.
    """

    results = retrieve("Quelles sont les conditions de distribution des dividendes ?", k=5)

    assert results
    assert len(results) <= 5
    assert results[0]["payload"]["circular_ref"] == "2026-03"


def test_graph_enhanced_retrieve_matches_vector_hits():
    """
    graph_enhanced_retrieve() must return the same vector hits as
    retrieve(), just with graph_context attached — never different
    hits, never fewer/more.
    """

    plain = retrieve("Quelles sont les conditions de distribution des dividendes ?", k=5)
    enriched = graph_enhanced_retrieve("Quelles sont les conditions de distribution des dividendes ?", k=5)

    plain_ids = [hit["payload"]["chunk_id"] for hit in plain]
    enriched_ids = [item["vector_result"]["payload"]["chunk_id"] for item in enriched]

    assert plain_ids == enriched_ids

    for item in enriched:
        assert "graph_context" in item


# ─────────────────────────────
# END-TO-END PIPELINE TESTS (slow — hits Ollama)
# ─────────────────────────────

@pytest.mark.slow
def test_ask_phase1_and_phase2_agree_on_easy_question():
    """
    For a question with a clear, unambiguous answer, Phase 1 and
    Phase 2 should both cite the same circular/article.
    """

    q = "Quelles sont les conditions de distribution des dividendes ?"

    answer_1 = ask(q, use_graph=False)
    answer_2 = ask(q, use_graph=True)

    assert "2026-3" in answer_1 or "2026-03" in answer_1
    assert "2026-3" in answer_2 or "2026-03" in answer_2


@pytest.mark.slow
def test_law_diff_router_bypasses_generation_and_is_correct():
    """
    The specific hallucination fix: this question shape must be
    answered directly from the graph via the router, not generation.
    """

    answer = ask(
        "Quelles circulaires se basent sur la loi n°2016-35 mais pas sur la loi n°2016-48 ?",
        use_graph=True,
    )

    assert "2026-43" in answer


@pytest.mark.slow
def test_law_diff_router_does_not_false_positive():
    """
    Ordinary graph-enhanced questions must not be intercepted by the
    router.
    """

    answer = ask("Sur quelles lois se fonde la circulaire 2026-03 ?", use_graph=True)

    assert "2016-35" in answer or "2016-48" in answer