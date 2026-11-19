"""
Checks that the evaluation gold set is consistent with the real corpus
(data/chunks.json), so the measured scores mean something:

- every expected chunk exists,
- every expected fact really appears in the expected source text,
- the cross-document answers match the law references in the corpus.
"""

import json
from collections import defaultdict

import pytest

from src.bct_rag.config import CHUNKS_FILE
from src.bct_rag.evaluation.metrics import fact_matches, normalize
from src.bct_rag.evaluation.run_eval import load_golden_set

GOLDEN = load_golden_set()
CHUNKS = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def text_by_id():
    """chunk_id or parent_chunk -> concatenated text."""

    texts = defaultdict(str)

    for chunk in CHUNKS:
        texts[chunk["chunk_id"]] += " " + chunk["text"]
        if chunk.get("parent_chunk") and chunk["parent_chunk"] != chunk["chunk_id"]:
            texts[chunk["parent_chunk"]] += " " + chunk["text"]

    return texts


TEXTS = text_by_id()


def laws_by_circular():
    laws = defaultdict(set)

    for chunk in CHUNKS:
        for law in chunk.get("references", {}).get("laws", []):
            laws[chunk["circular_ref"]].add(law)

    return laws


def test_ids_are_unique():
    ids = [item["id"] for item in GOLDEN]
    assert len(ids) == len(set(ids))


def test_golden_set_covers_every_category():
    categories = {item["category"] for item in GOLDEN}
    assert {"single_doc", "cross_doc", "cross_doc_negation", "unanswerable"} <= categories


@pytest.mark.parametrize("item", GOLDEN, ids=lambda i: i["id"])
def test_expected_chunks_exist(item):
    for chunk_id in item.get("expected_chunks", []):
        assert chunk_id in TEXTS, f"unknown chunk {chunk_id}"


@pytest.mark.parametrize(
    "item",
    [i for i in GOLDEN if i["category"] == "single_doc"],
    ids=lambda i: i["id"],
)
def test_single_doc_facts_are_in_source_text(item):
    """Gold facts must come from the regulation text, not from an LLM answer."""

    source = " ".join(TEXTS[c] for c in item["expected_chunks"])

    for fact in item["facts"]:
        assert fact_matches(source, fact), f"{fact!r} not found in {item['expected_chunks']}"


def test_law_diff_answer_matches_corpus():
    laws = laws_by_circular()
    expected = {c for c, refs in laws.items() if "2016-35" in refs and "2016-48" not in refs}

    item = next(i for i in GOLDEN if i["id"] == "law_diff_2016_35_not_2016_48")

    assert expected == {normalize(f) for f in item["facts"]} == {"2026-43"}


def test_law_intersection_answer_matches_corpus():
    laws = laws_by_circular()
    expected = {c for c, refs in laws.items() if {"2016-35", "2016-48"} <= refs}

    item = next(i for i in GOLDEN if i["id"] == "law_intersection_2016_35_and_2016_48")

    assert expected == set(item["facts"])
