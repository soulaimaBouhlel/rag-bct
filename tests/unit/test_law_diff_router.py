"""
Unit tests for the Phase 2 law-diff router and the evaluation runner.

Neo4j is replaced by an in-memory graph built from data/chunks.json
(same references the real graph is built from), and the LLM / Qdrant
are never called.
"""

import json
from collections import defaultdict

import pytest

import src.bct_rag.pipeline as pipeline
from src.bct_rag.config import CHUNKS_FILE
from src.bct_rag.evaluation import run_eval

CHUNKS = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


class InMemoryGraphRetriever:
    """Implements the only GraphRetriever method the router uses."""

    def __init__(self):
        self.laws = defaultdict(set)
        for chunk in CHUNKS:
            for law in chunk.get("references", {}).get("laws", []):
                self.laws[chunk["circular_ref"]].add(law)

    def find_circulars_citing_but_not(self, law_ref, excluded_law_ref):
        return [
            {"circular_reference": c}
            for c in sorted(self.laws)
            if law_ref in self.laws[c] and excluded_law_ref not in self.laws[c]
        ]

    def close(self):
        pass


@pytest.fixture
def fake_graph(monkeypatch):
    monkeypatch.setattr(pipeline, "GraphRetriever", InMemoryGraphRetriever)


@pytest.mark.parametrize(
    "question",
    [
        "Quelles circulaires se basent sur la loi n°2016-35 mais pas sur la loi n°2016-48 ?",
        "Which circulars cite law 2016-35 but not law 2016-48?",
        "Circulaires fondées sur la loi 2016 35 excluant la loi 2016 48",
    ],
)
def test_router_answers_law_diff_from_graph(fake_graph, question):
    answer = pipeline._try_law_diff_query(question)

    assert answer is not None
    assert "2026-43" in answer
    assert "2026-03" not in answer


@pytest.mark.parametrize(
    "question",
    [
        "Sur quelles lois se fonde la circulaire 2026-03 ?",
        "Quelles sont les conditions de distribution des dividendes ?",
        "Quelles circulaires se basent à la fois sur la loi n°2016-35 et sur la loi n°2016-48 ?",
    ],
)
def test_router_does_not_intercept_other_questions(fake_graph, question):
    assert pipeline._try_law_diff_query(question) is None


def test_router_reports_empty_result(fake_graph):
    answer = pipeline._try_law_diff_query("circulaires citant la loi 2016-48 mais pas la loi 2016-35")
    assert answer.startswith("Aucune circulaire")


def test_ask_detailed_uses_router_without_retrieval(fake_graph, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("retrieval/generation must be bypassed")

    monkeypatch.setattr(pipeline, "graph_enhanced_retrieve", must_not_be_called)
    monkeypatch.setattr(pipeline, "generate", must_not_be_called)

    result = pipeline.ask_detailed(
        "Quelles circulaires se basent sur la loi n°2016-35 mais pas sur la loi n°2016-48 ?",
        use_graph=True,
    )

    assert result["route"] == "graph_router"
    assert "2026-43" in result["answer"]


def test_ask_detailed_phase1_never_uses_router(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q: [])
    monkeypatch.setattr(pipeline, "generate", lambda q, docs, graph_context=None: "I could not find this information in the available regulations.")

    result = pipeline.ask_detailed(
        "Quelles circulaires se basent sur la loi n°2016-35 mais pas sur la loi n°2016-48 ?",
        use_graph=False,
    )

    assert result["route"] == "vector"


def test_run_eval_end_to_end_with_fake_pipeline():
    """The runner scores both phases and builds a report (fake answers)."""

    items = [i for i in run_eval.load_golden_set() if i["id"] in {
        "dividend_conditions", "law_diff_2016_35_not_2016_48", "unanswerable_branch_capital",
    }]

    def fake_ask(question, use_graph=False):
        doc = {"payload": {"chunk_id": "Cir_2026_03_fr-article-1", "parent_chunk": "Cir_2026_03_fr-article-1"}}

        if "succursale" in question:
            return {"answer": "I could not find this information in the available regulations.",
                    "docs": [], "graph_context": None, "route": "vector"}

        if "mais pas" in question:
            if use_graph:
                return {"answer": "Circulaire(s) ... : 2026-43", "docs": [],
                        "graph_context": None, "route": "graph_router"}
            return {"answer": "- Circular 2026-03 (Article 4)", "docs": [doc],
                    "graph_context": None, "route": "vector"}

        return {"answer": "35% ... 2,5% ... 3,5% ... accord préalable (Circular 2026-3, Article 1)",
                "docs": [doc], "graph_context": "- x" if use_graph else None,
                "route": "graph_enhanced" if use_graph else "vector"}

    report = run_eval.run(items, ask_fn=fake_ask)

    p1 = report["summary"]["phase1_vector"]
    p2 = report["summary"]["phase2_graph"]

    assert p1["hallucination_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert p2["hallucination_rate"] == 0.0
    assert p2["accuracy"] == 1.0
    # Retrieval is reported once (from Phase 1), so the router question
    # can't silently inflate Phase 2's retrieval average.
    assert report["summary"]["retrieval"]["hit@k"] == pytest.approx(0.5)

    markdown = run_eval.to_markdown(report)
    assert "| Answer accuracy |" in markdown
    assert "## Retrieval (shared by both phases)" in markdown
