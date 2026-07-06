import chromadb
from bct_rag.embedding.embedder import embed

class ChromaStore:
    def __init__(self, persist_dir="data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name="bct_circulars",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = embed(texts)

        ids = [c["chunk_id"] for c in chunks]

        metadatas = [
            {
                "chunk_type": c["chunk_type"],
                "article_number": c.get("article_number"),
                "circular_ref": c["circular_ref"],
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_text, k=5):
        q_emb = embed(query_text)

        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )