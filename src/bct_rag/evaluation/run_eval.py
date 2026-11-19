"""
Phase 1 (vector-only) vs Phase 2 (graph-enhanced) evaluation.

Runs every question of data/eval/golden_set.json through both phases,
scores retrieval and answers (see metrics.py), and writes:

    eval_results/eval_<timestamp>.json   every answer + per-question scores
    eval_results/eval_<timestamp>.md     summary table + per-question table

Needs the full stack: Qdrant + Neo4j (docker compose up -d qdrant neo4j),
an indexed collection and graph, and Ollama with the configured model.

Usage:
    PYTHONPATH=. python -m src.bct_rag.evaluation.run_eval
    PYTHONPATH=. python -m src.bct_rag.evaluation.run_eval --only law_diff_2016_35_not_2016_48
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.bct_rag.config import DATA_DIR, ROOT_DIR
from src.bct_rag.evaluation.metrics import aggregate, score_answer, score_retrieval

GOLDEN_SET = DATA_DIR / "eval" / "golden_set.json"
RESULTS_DIR = ROOT_DIR / "eval_results"

PHASES = {
    "phase1_vector": False,
    "phase2_graph": True,
}

SUMMARY_METRICS = [
    ("accuracy", "Answer accuracy"),
    ("fact_recall", "Fact recall"),
    ("citation_accuracy", "Citation accuracy"),
    ("hallucination_rate", "Hallucination rate (lower is better)"),
    ("false_abstention_rate", "False abstention rate (lower is better)"),
    ("correct_abstention_rate", "Correct abstention (unanswerable)"),
    ("hit@k", "Retrieval hit@k"),
    ("recall@k", "Retrieval recall@k"),
    ("mrr", "Retrieval MRR"),
    ("mean_latency_s", "Mean latency (s)"),
]


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_item(item: dict, use_graph: bool, ask_fn) -> dict:
    start = time.perf_counter()
    result = ask_fn(item["question"], use_graph=use_graph)
    latency = time.perf_counter() - start

    docs = result["docs"]

    row = {
        "id": item["id"],
        "category": item.get("category"),
        "should_abstain": item.get("should_abstain", False),
        "route": result["route"],
        "answer": result["answer"],
        "retrieved": [d["payload"].get("parent_chunk") or d["payload"].get("chunk_id") for d in docs],
        "graph_context_chars": len(result["graph_context"] or ""),
        "latency_s": round(latency, 2),
    }

    # The graph router answers without vector retrieval: retrieval
    # metrics don't apply to it, so they are left empty (None).
    if result["route"] == "graph_router":
        row.update({"hit": None, "recall": None, "precision": None, "rr": None})
    else:
        row.update(score_retrieval(item, docs))

    row.update(score_answer(item, result["answer"]))

    return row


def run(items: list[dict], ask_fn=None) -> dict:
    if ask_fn is None:
        from src.bct_rag.pipeline import ask_detailed as ask_fn

    report = {"phases": {}, "summary": {}}

    for phase, use_graph in PHASES.items():
        rows = []

        for item in items:
            print(f"[{phase}] {item['id']}")
            rows.append(evaluate_item(item, use_graph, ask_fn))

        report["phases"][phase] = rows
        report["summary"][phase] = aggregate(rows)

    return report


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✅" if value else "❌"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def to_markdown(report: dict) -> str:
    p1, p2 = report["summary"]["phase1_vector"], report["summary"]["phase2_graph"]

    lines = [
        "# Phase 1 (vector) vs Phase 2 (graph-enhanced)",
        "",
        f"Questions: {p1['questions']}",
        "",
        "| Metric | Phase 1 | Phase 2 |",
        "|---|---|---|",
    ]

    for key, label in SUMMARY_METRICS:
        lines.append(f"| {label} | {_fmt(p1[key])} | {_fmt(p2[key])} |")

    lines += [
        "",
        "## Per question",
        "",
        "| Question | Category | P1 correct | P2 correct | P1 halluc. | P2 halluc. | P2 route |",
        "|---|---|---|---|---|---|---|",
    ]

    for r1, r2 in zip(report["phases"]["phase1_vector"], report["phases"]["phase2_graph"]):
        lines.append(
            f"| {r1['id']} | {r1['category']} | {_fmt(r1['correct'])} | {_fmt(r2['correct'])} "
            f"| {_fmt(r1['hallucination'])} | {_fmt(r2['hallucination'])} | {r2['route']} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", nargs="*", help="run only these question ids")
    args = parser.parse_args()

    items = load_golden_set()
    if args.only:
        items = [i for i in items if i["id"] in set(args.only)]

    report = run(items)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    md_path = RESULTS_DIR / f"eval_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n" + to_markdown(report))
    print(f"Saved: {json_path}\n       {md_path}")


if __name__ == "__main__":
    main()
