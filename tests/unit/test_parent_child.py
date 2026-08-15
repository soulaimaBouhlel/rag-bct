from src.bct_rag.retrieval.parent_child import (
    get_parent_id,
    group_hits_by_parent,
    sort_children,
    merge_children,
)


def make_hit(
    chunk_id,
    text,
    parent_chunk=None,
    chunk_index=None,
):
    return {
        "payload": {
            "chunk_id": chunk_id,
            "text": text,
            "parent_chunk": parent_chunk,
            "chunk_index": chunk_index,
        },
        "score": 0.90,
    }


def test_split_chunk_returns_parent_id():

    payload = {
        "chunk_id": "article-20-2",
        "parent_chunk": "article-20",
    }

    assert get_parent_id(payload) == "article-20"


def test_unsplit_chunk_uses_itself_as_parent():

    payload = {
        "chunk_id": "article-5",
        "parent_chunk": None,
    }

    assert get_parent_id(payload) == "article-5"


def test_group_hits_by_parent():

    hits = [
        make_hit(
            "article-20-2",
            "part two",
            "article-20",
            1,
        ),
        make_hit(
            "article-20-1",
            "part one",
            "article-20",
            0,
        ),
        make_hit(
            "article-5",
            "article five",
            None,
            0,
        ),
    ]

    groups = group_hits_by_parent(hits)

    assert set(groups.keys()) == {
        "article-20",
        "article-5",
    }

    assert len(groups["article-20"]) == 2
    assert len(groups["article-5"]) == 1


def test_sort_children():

    children = [
        make_hit(
            "article-20-3",
            "third",
            "article-20",
            2,
        ),
        make_hit(
            "article-20-1",
            "first",
            "article-20",
            0,
        ),
        make_hit(
            "article-20-2",
            "second",
            "article-20",
            1,
        ),
    ]

    ordered = sort_children(children)

    assert [
        hit["payload"]["chunk_id"]
        for hit in ordered
    ] == [
        "article-20-1",
        "article-20-2",
        "article-20-3",
    ]


def test_merge_children_in_correct_order():

    children = [
        make_hit(
            "article-20-2",
            "Second part.",
            "article-20",
            1,
        ),
        make_hit(
            "article-20-1",
            "First part.",
            "article-20",
            0,
        ),
        make_hit(
            "article-20-3",
            "Third part.",
            "article-20",
            2,
        ),
    ]

    result = merge_children(children)

    assert result == (
        "First part.\n\n"
        "Second part.\n\n"
        "Third part."
    )