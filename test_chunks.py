import json

from bct_rag.ingestion.embedder import embed


with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

first = chunks[0]

vector = embed(first["text"])

print("Chunk ID:", first["chunk_id"])
print("Token count:", first["token_count"])
print("Embedding dimension:", len(vector))
print("chunks:", len(chunks))
print("max_tokens:", max(c["token_count"] for c in chunks))
print("min_tokens:", min(c["token_count"] for c in chunks))
print("avg_tokens:", sum(c["token_count"] for c in chunks) / len(chunks))