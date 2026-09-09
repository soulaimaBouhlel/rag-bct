"""
Demo: Phase 1 (vector-only) vs Phase 2 (graph-enhanced) retrieval.

Run with:
    PYTHONPATH=. python3 demo_comparison.py
"""

from src.bct_rag.pipeline import ask

QUESTIONS = [
    "Sur quelles lois se fonde la circulaire 2026-03 ?",
    "Quelles sont les conditions de distribution des dividendes ?",
    "Quelles circulaires se basent sur la loi n°2016-35 mais pas sur la loi n°2016-48 ?",
]


def main():

    for question in QUESTIONS:

        print("=" * 90)
        print(f"QUESTION: {question}")
        print("=" * 90)

        print("\n--- Phase 1 (vector search only) ---\n")
        print(ask(question, use_graph=False))

        print("\n--- Phase 2 (graph-enhanced) ---\n")
        print(ask(question, use_graph=True))

        print("\n")


if __name__ == "__main__":
    main()