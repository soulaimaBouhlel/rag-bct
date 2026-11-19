"""
Unit tests for the evaluation metrics (no Qdrant / Neo4j / Ollama).
"""

import pytest

from src.bct_rag.evaluation.metrics import (
    aggregate,
    citation_ok,
    fact_recall,
    forbidden_hits,
    hit_at_k,
    is_abstention,
    normalize,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_answer,
)


def doc(chunk_id, parent=None):
    return {"payload": {"chunk_id": chunk_id, "parent_chunk": parent or chunk_id}}


DOCS = [doc("a"), doc("b-1", parent="b"), doc("c")]


# ─── normalization ───

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2, 5 %", "2,5%"),
        ("2.5%", "2,5%"),
        ("Circulaire 2026-3", "circulaire 2026-03"),
        ("Circular n°2026-03", "circular n°2026-03"),
        ("loi 2016-35", "loi 2016-35"),
        ("Accord Préalable", "accord prealable"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


# ─── retrieval ───

def test_hit_at_k_matches_chunk_or_parent():
    assert hit_at_k(DOCS, ["b"]) == 1.0
    assert hit_at_k(DOCS, ["b-1"]) == 1.0
    assert hit_at_k(DOCS, ["zzz"]) == 0.0


def test_recall_precision_mrr():
    assert recall_at_k(DOCS, ["a", "c", "missing"]) == pytest.approx(2 / 3)
    assert precision_at_k(DOCS, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(DOCS, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(DOCS, ["a"]) == 1.0
    assert reciprocal_rank([], ["a"]) == 0.0


# ─── answers ───

def test_abstention_detection():
    assert is_abstention("I could not find this information in the available regulations.")
    assert not is_abstention("La circulaire 2026-03 se fonde sur la loi n°2016-35.")


def test_fact_recall_handles_alternatives_and_formatting():
    answer = "Dans la limite de 35 % ... de 2, 5 % et 3.5% au moins, quinze jours avant."
    assert fact_recall(answer, ["35%", "2,5%", "3,5%", "15|quinze"]) == 1.0
    assert fact_recall(answer, ["35%", "6%"]) == 0.5


def test_citation_ok_ignores_padding():
    assert citation_ok("(Circular 2026-3, Article 1)", ["2026-03"])
    assert not citation_ok("(Circular 2026-1)", ["2026-03"])


def test_forbidden_hits_detects_wrong_circular():
    assert forbidden_hits("- Circular 2026-03", ["2026-03", "2026-01"]) == ["2026-03"]


def test_score_answer_flags_the_original_phase2_hallucination():
    """The exact Phase 2 output that exposed the bug must be scored as a hallucination."""

    item = {
        "facts": ["2026-43"],
        "circulars": ["2026-43"],
        "forbidden": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
    }

    scored = score_answer(item, "- Circular 2026-03\n\nSources:\n- Circular 2026-03\n- Article 4")

    assert scored["hallucination"] is True
    assert scored["correct"] is False


def test_score_answer_router_answer_is_correct():
    item = {"facts": ["2026-43"], "circulars": ["2026-43"], "forbidden": ["2026-03"]}

    scored = score_answer(
        item,
        "Circulaire(s) se basant sur la loi n°2016-35 mais pas sur la loi n°2016-48 : 2026-43",
    )

    assert scored["correct"] is True
    assert scored["hallucination"] is False


def test_score_answer_unanswerable():
    item = {"should_abstain": True}

    assert score_answer(item, "I could not find this information in the available regulations.")["correct"]
    assert score_answer(item, "Le capital minimum est de 50 MD.")["hallucination"]


def test_false_abstention_counted():
    scored = score_answer({"facts": ["6%"]}, "I could not find this information in the available regulations.")
    assert scored["false_abstention"] is True
    assert scored["correct"] is False


def test_aggregate():
    rows = [
        {"should_abstain": False, "correct": True, "fact_recall": 1.0, "citation_ok": True,
         "hallucination": False, "false_abstention": False, "abstained": False,
         "hit": 1.0, "recall": 1.0, "precision": 0.2, "rr": 1.0, "latency_s": 2.0},
        {"should_abstain": False, "correct": False, "fact_recall": 0.0, "citation_ok": False,
         "hallucination": True, "false_abstention": False, "abstained": False,
         "hit": None, "recall": None, "precision": None, "rr": None, "latency_s": 4.0},
        {"should_abstain": True, "correct": True, "fact_recall": None, "citation_ok": None,
         "hallucination": False, "false_abstention": False, "abstained": True,
         "hit": None, "recall": None, "precision": None, "rr": None, "latency_s": 3.0},
    ]

    summary = aggregate(rows)

    assert summary["accuracy"] == pytest.approx(0.667, abs=1e-3)
    assert summary["hallucination_rate"] == pytest.approx(0.333, abs=1e-3)
    assert summary["fact_recall"] == 0.5
    assert summary["correct_abstention_rate"] == 1.0
    assert summary["hit@k"] == 1.0          # None rows are excluded
    assert summary["mean_latency_s"] == 3.0
