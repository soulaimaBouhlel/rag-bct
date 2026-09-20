import re

from src.bct_rag.ingestion.pipeline import run as ingest
from src.bct_rag.indexing.index_qdrant import main as index_qdrant
from src.bct_rag.retrieval.retriever import retrieve, graph_enhanced_retrieve
from src.bct_rag.retrieval.graph_retriever import format_graph_context, GraphRetriever
from src.bct_rag.llm.generator import generate

# Matches any two law-style references (YYYY-NN) in the question,
# regardless of surrounding phrasing or language, combined with a
# negation keyword anywhere in the sentence (English or French).
# Deliberately loose: this only needs to detect the SHAPE of the
# question, not parse full grammar — Claude Desktop may rephrase or
# translate the user's original question before calling the tool.
_LAW_REF_PATTERN = re.compile(r"\b(\d{4}[-\s]\d{1,3})\b")

_NEGATION_KEYWORDS = (
    "mais pas", "sans", "non ", "excluant", "à l'exclusion",
    "but not", "not ", "excluding", "except",
)


def _try_law_diff_query(question: str):
    """
    Detect 'which circulars cite law A but not law B' style questions
    and answer them directly from the graph — this question class
    depends on negation/set-difference logic across the whole corpus,
    which chunk-retrieval + generation cannot reliably answer (the
    correct source document may not even be in the vector top-k, and
    small local models don't reliably follow an abstention instruction
    when surface-level co-occurrence looks plausible).
    """

    law_refs = _LAW_REF_PATTERN.findall(question)

    has_negation = any(kw in question.lower() for kw in _NEGATION_KEYWORDS)

    if len(law_refs) != 2 or not has_negation:
        return None

    law_a = law_refs[0].replace(" ", "-")
    law_b = law_refs[1].replace(" ", "-")

    gr = GraphRetriever()
    rows = gr.find_circulars_citing_but_not(law_a, law_b)
    gr.close()

    if not rows:
        return (
            f"Aucune circulaire ne se base sur la loi n°{law_a} sans se baser également sur la loi n°{law_b}."
        )

    circulars = ", ".join(row["circular_reference"] for row in rows)

    return f"Circulaire(s) se basant sur la loi n°{law_a} mais pas sur la loi n°{law_b} : {circulars}"


def run():
    print("=" * 70)
    print("BCT RAG PIPELINE")
    print("=" * 70)

    ingest()

    index_qdrant()

    print("\nPipeline completed successfully.")


def ask(question: str, use_graph: bool = False):

    if use_graph:

        graph_answer = _try_law_diff_query(question)
        if graph_answer is not None:
            return graph_answer

        enriched = graph_enhanced_retrieve(question)
        docs = [item["vector_result"] for item in enriched]
        graph_context = format_graph_context(enriched)

    else:
        docs = retrieve(question)
        graph_context = None

    print(f"Retrieved {len(docs)} chunks, generating answer...")

    answer = generate(question, docs, graph_context=graph_context)

    return answer


if __name__ == "__main__":
    run()