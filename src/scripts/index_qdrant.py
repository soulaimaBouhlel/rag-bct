#!/usr/bin/env python3
"""
Index document chunks into Qdrant.

Pipeline:

    chunks.json
         │
         ▼
    embed_batch()
         │
         ▼
    PointStruct
         │
         ▼
      upsert()
"""

import json
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.bct_rag.embedding.embedder import embed_batch


COLLECTION_NAME = "regulations"
CHUNKS_PATH = "data/chunks.json"
BATCH_SIZE = 100


def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSON."""

    print(f"\n📂 Loading chunks from {path}...")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✅ Loaded {len(chunks)} chunks")

    return chunks


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create the collection if it does not already exist."""

    try:
        client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' already exists")

    except Exception:
        print(f"📦 Creating collection '{collection_name}'...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(f"✅ Collection '{collection_name}' created")


def build_points(chunks: list[dict]) -> tuple[list[PointStruct], int]:
    """
    Generate embeddings and convert chunks into Qdrant PointStructs.

    Returns:
        (points, vector_size)
    """

    print("\n🧠 Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_batch(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Number of embeddings does not match number of chunks."
        )

    if not vectors:
        raise ValueError("No embeddings were generated.")

    vector_size = len(vectors[0])

    print(
        f"✅ Generated {len(vectors)} embeddings "
        f"({vector_size} dimensions)"
    )

    points = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    # Chunk metadata
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk["chunk_type"],
                    "article_number": chunk.get("article_number"),

                    # Document metadata
                    "circular_ref": chunk["circular_ref"],
                    "source_file": chunk["source_file"],
                    "circular_type": chunk["circular_type"],
                    "title": chunk["title"],
                    "objet": chunk["objet"],
                    "chapter": chunk.get("chapter"),

                    # Chunk text
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],

                    # Parent-child information
                    "parent_chunk": chunk.get("parent_chunk"),
                    "chunk_index": chunk.get("chunk_index"),
                    "num_chunks": chunk.get("num_chunks"),
                },
            )
        )

    print(f"✅ Built {len(points)} Qdrant points")

    return points, vector_size


def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    batch_size: int = BATCH_SIZE,
) -> None:
    """Upload points to Qdrant."""

    print("\n📤 Uploading vectors...")

    total = len(points)

    for start in range(0, total, batch_size):

        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

        uploaded = min(start + batch_size, total)

        print(f"   {uploaded}/{total} uploaded")

    print(f"✅ Uploaded {total} vectors")


def main():

    print("\n" + "=" * 70)
    print("INDEXING CHUNKS INTO QDRANT")
    print("=" * 70)

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    chunks = load_chunks(CHUNKS_PATH)

    points, vector_size = build_points(chunks)

    create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_size=vector_size,
    )

    upload_points(
        client=client,
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("\n" + "=" * 70)
    print("✅ Indexing completed successfully")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()