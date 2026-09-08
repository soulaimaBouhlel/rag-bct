from src.bct_rag.ingestion.pipeline import run as ingest
from src.bct_rag.indexing.index_qdrant import main as index_qdrant
from src.bct_rag.retrieval.retriever import retrieve, graph_enhanced_retrieve
from src.bct_rag.retrieval.graph_retriever import format_graph_context
from src.bct_rag.llm.generator import generate

def run():
    print("=" * 70)
    print("BCT RAG PIPELINE")
    print("=" * 70)

    ingest()

    index_qdrant()

    print("\nPipeline completed successfully.")

def ask(question, use_graph: bool = False):

    if use_graph:
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
