import json
from bct_rag.vectorstore.chroma_store import ChromaStore

def build():
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    store = ChromaStore()
    store.add_chunks(chunks)

    print(f"Indexed {len(chunks)} chunks into ChromaDB")

if __name__ == "__main__":
    build()