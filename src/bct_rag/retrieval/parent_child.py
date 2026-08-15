from collections import defaultdict


def get_parent_id(payload: dict) -> str:
    """
    Return the logical parent ID for a retrieved chunk.

    For chunks produced by the current chunker:
    - split chunks have a `parent_chunk`
    - unsplit chunks may have `parent_chunk=None`

    If there is no parent_chunk, the chunk itself is
    treated as the parent.
    """

    parent_id = payload.get("parent_chunk")

    if parent_id:
        return parent_id

    return payload["chunk_id"]


def group_hits_by_parent(hits: list[dict]) -> dict[str, list[dict]]:
    """
    Group retrieved child chunks by their parent ID.

    Input:
        [
            {"payload": {...}, "score": 0.91},
            {"payload": {...}, "score": 0.88},
        ]

    Output:
        {
            "article-1": [...],
            "article-2": [...]
        }
    """

    groups = defaultdict(list)

    for hit in hits:
        payload = hit["payload"]

        parent_id = get_parent_id(payload)

        groups[parent_id].append(hit)

    return dict(groups)


def sort_children(children: list[dict]) -> list[dict]:
    """
    Sort child chunks according to their original chunk order.
    """

    return sorted(
        children,
        key=lambda hit: (
            hit["payload"].get("chunk_index") or 0
        ),
    )


def merge_children(children: list[dict]) -> str:
    """
    Merge child chunks into one parent document.

    Children are sorted using chunk_index before merging.
    """

    ordered = sort_children(children)

    return "\n\n".join(
        hit["payload"]["text"]
        for hit in ordered
    )