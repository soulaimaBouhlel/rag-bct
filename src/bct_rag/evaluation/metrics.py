"""
Metrics for comparing Phase 1 (vector-only) and Phase 2 (graph-enhanced).

All functions are pure (no Qdrant / Neo4j / Ollama), so they are unit
tested in tests/unit/test_eval_metrics.py.

Measures
--------
Retrieval (per question, on the documents given to the LLM):
    hit@k        1 if at least one expected chunk was retrieved
    recall@k     share of expected chunks that were retrieved
    precision@k  share of retrieved chunks that were expected
    MRR          1 / rank of the first expected chunk (0 if none)

Answer (per question):
    fact_recall      share of expected facts found in the answer
    correct          all expected facts found AND no forbidden content
                     (for unanswerable questions: the model abstained)
    citation_ok      every expected circular is cited in the answer
    hallucination    answer contains forbidden content, OR the model
                     answered a question it should have abstained on
    false_abstention model abstained although the answer is in the corpus
"""

import re
import unicodedata
from statistics import mean

ABSTENTION_PATTERNS = [
    r"could not find this information",
    r"je n'ai pas trouve",
    r"information (non|pas) (disponible|trouvee)",
    r"ne figure pas dans",
]

_CIRCULAR_REF = re.compile(r"\b(20\d{2})\s*-\s*0*(\d{1,3})\b")


# ─────────────────────────────
# NORMALIZATION
# ─────────────────────────────

def normalize(text: str) -> str:
    """
    Lowercase, strip accents, unify apostrophes / spaces, and make
    equivalent notations identical:

        "2, 5 %"  -> "2,5%"     (OCR spacing)
        "2.5%"    -> "2,5%"     (decimal point vs comma)
        "2026-3"  -> "2026-03"  (reference padding)
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace(" ", " ")

    text = re.sub(r"(\d)\s*[.,]\s*(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = _CIRCULAR_REF.sub(lambda m: f"{m.group(1)}-{int(m.group(2)):02d}", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_ref(ref: str) -> str:
    return normalize(ref)


# ─────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────

def doc_ids(doc: dict) -> set[str]:
    """
    A retrieved parent document can be matched by its own chunk_id or
    by its parent_chunk (after parent-child reconstruction both exist).
    """

    payload = doc.get("payload", {})

    return {
        value
        for value in (payload.get("chunk_id"), payload.get("parent_chunk"))
        if value
    }


def _ranked_ids(docs: list[dict]) -> list[set[str]]:
    return [doc_ids(doc) for doc in docs]


def hit_at_k(docs: list[dict], expected: list[str]) -> float:
    expected_set = set(expected)
    return float(any(ids & expected_set for ids in _ranked_ids(docs)))


def recall_at_k(docs: list[dict], expected: list[str]) -> float:
    if not expected:
        return 0.0

    found = set().union(*_ranked_ids(docs)) if docs else set()

    return sum(1 for e in expected if e in found) / len(expected)


def precision_at_k(docs: list[dict], expected: list[str]) -> float:
    if not docs:
        return 0.0

    expected_set = set(expected)

    return sum(1 for ids in _ranked_ids(docs) if ids & expected_set) / len(docs)


def reciprocal_rank(docs: list[dict], expected: list[str]) -> float:
    expected_set = set(expected)

    for rank, ids in enumerate(_ranked_ids(docs), start=1):
        if ids & expected_set:
            return 1.0 / rank

    return 0.0


# ─────────────────────────────
# ANSWER METRICS
# ─────────────────────────────

def is_abstention(answer: str) -> bool:
    norm = normalize(answer)
    return any(re.search(p, norm) for p in ABSTENTION_PATTERNS)


def fact_matches(answer: str, fact: str) -> bool:
    """
    A fact is a set of alternatives separated by "|", each compared
    after normalization: "15|quinze" matches "15 jours" or "quinze jours".
    """

    norm = normalize(answer)

    return any(normalize(alt) in norm for alt in fact.split("|"))


def fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0

    return sum(fact_matches(answer, f) for f in facts) / len(facts)


def forbidden_hits(answer: str, forbidden: list[str]) -> list[str]:
    return [f for f in forbidden if fact_matches(answer, f)]


def citation_ok(answer: str, expected_circulars: list[str]) -> bool:
    norm = normalize(answer)
    return all(normalize_ref(ref) in norm for ref in expected_circulars)


def score_answer(item: dict, answer: str) -> dict:
    """
    Score one answer against one gold-set item.
    """

    abstained = is_abstention(answer)
    should_abstain = item.get("should_abstain", False)

    if should_abstain:
        return {
            "abstained": abstained,
            "fact_recall": None,
            "correct": abstained,
            "citation_ok": None,
            "forbidden": [],
            "hallucination": not abstained,
            "false_abstention": False,
        }

    forbidden = forbidden_hits(answer, item.get("forbidden", []))
    recall = 0.0 if abstained else fact_recall(answer, item.get("facts", []))

    return {
        "abstained": abstained,
        "fact_recall": recall,
        "correct": (not abstained) and recall == 1.0 and not forbidden,
        "citation_ok": (not abstained) and citation_ok(answer, item.get("circulars", [])),
        "forbidden": forbidden,
        "hallucination": bool(forbidden),
        "false_abstention": abstained,
    }


def score_retrieval(item: dict, docs: list[dict]) -> dict:
    expected = item.get("expected_chunks", [])

    if not expected:
        return {"hit": None, "recall": None, "precision": None, "rr": None}

    return {
        "hit": hit_at_k(docs, expected),
        "recall": recall_at_k(docs, expected),
        "precision": precision_at_k(docs, expected),
        "rr": reciprocal_rank(docs, expected),
    }


# ─────────────────────────────
# AGGREGATION
# ─────────────────────────────

def _avg(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 3) if values else None


def aggregate(rows: list[dict]) -> dict:
    """
    rows: one dict per question for ONE phase, each holding the keys
    produced by score_answer() and score_retrieval() plus "latency_s"
    and "should_abstain".
    """

    answerable = [r for r in rows if not r["should_abstain"]]
    unanswerable = [r for r in rows if r["should_abstain"]]

    return {
        "questions": len(rows),
        "accuracy": _avg(r["correct"] for r in rows),
        "fact_recall": _avg(r["fact_recall"] for r in answerable),
        "citation_accuracy": _avg(r["citation_ok"] for r in answerable),
        "hallucination_rate": _avg(r["hallucination"] for r in rows),
        "false_abstention_rate": _avg(r["false_abstention"] for r in answerable),
        "correct_abstention_rate": _avg(r["abstained"] for r in unanswerable),
        "hit@k": _avg(r["hit"] for r in rows),
        "recall@k": _avg(r["recall"] for r in rows),
        "precision@k": _avg(r["precision"] for r in rows),
        "mrr": _avg(r["rr"] for r in rows),
        "mean_latency_s": _avg(r["latency_s"] for r in rows),
    }
