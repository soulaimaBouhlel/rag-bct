"""
Runs the full Phase 1 vs Phase 2 evaluation on the real stack
(Qdrant + Neo4j + Ollama) and checks Phase 2's guarantees:

- it never hallucinates more than Phase 1,
- it answers the law-diff question correctly (the original bug),
- its vector retrieval is identical to Phase 1's.

    PYTHONPATH=. pytest tests/integration/test_eval_phase_comparison.py -v -s
"""

import pytest

from src.bct_rag.evaluation.run_eval import load_golden_set, run, to_markdown


@pytest.fixture(scope="module")
def report():
    report = run(load_golden_set())
    print("\n" + to_markdown(report))
    return report


@pytest.mark.slow
def test_phase2_does_not_hallucinate_more(report):
    p1 = report["summary"]["phase1_vector"]
    p2 = report["summary"]["phase2_graph"]

    assert p2["hallucination_rate"] <= p1["hallucination_rate"]


@pytest.mark.slow
def test_phase2_answers_law_diff_question(report):
    row = next(r for r in report["phases"]["phase2_graph"] if r["id"] == "law_diff_2016_35_not_2016_48")

    assert row["route"] == "graph_router"
    assert row["correct"]
    assert not row["hallucination"]


@pytest.mark.slow
def test_phase2_keeps_phase1_retrieval(report):
    p1 = {r["id"]: r["retrieved"] for r in report["phases"]["phase1_vector"]}

    for row in report["phases"]["phase2_graph"]:
        if row["route"] == "graph_enhanced":
            assert row["retrieved"] == p1[row["id"]]
