from src.bct_rag.ingestion.pipeline import run as ingest
from src.bct_rag.indexing.index_qdrant import main as index_qdrant
from src.bct_rag.retrieval.retriever import retrieve
from src.bct_rag.llm.generator import generate

def run():
    print("=" * 70)
    print("BCT RAG PIPELINE")
    print("=" * 70)

    ingest()

    index_qdrant()

    print("\nPipeline completed successfully.")

def ask(question):

    docs = retrieve(question)
    print(f"Retrieved {len(docs)} chunks, generating answer...")
    answer = generate(question, docs)

    return answer
if __name__ == "__main__":
    run()
