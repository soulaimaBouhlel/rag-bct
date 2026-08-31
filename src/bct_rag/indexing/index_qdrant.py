#!/usr/bin/env python3
"""
Index chunks into Qdrant.

Pipeline:

    chunks.json
         │
         ▼
    load_chunks()
         │
         ▼
    embed_chunks()
         │
         ▼
    build_points()
         │
         ▼
    upload_points()
         │
         ▼
      Qdrant
"""

import json
import os
import uuid

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

def make_point_id(chunk_id: str) -> str:
    """
    Convert a business ID into a deterministic UUID.

    Same chunk_id always produces the same UUID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            chunk_id,
        )
    )
# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def load_chunks(path: str) -> list[dict]:
    """
    Load chunks.json.

    Responsibility:
        JSON -> Python objects
    """

    print(f"\n📂 Loading chunks from {path}...")

    with open(path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"✅ Loaded {len(chunks)} chunks")

    return chunks


# ---------------------------------------------------------
# Embedding
# ---------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Generate embeddings for chunks.

    Responsibility:
        chunks -> vectors
    """

    print("\n🧠 Generating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = embed_batch(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Number of vectors does not match number of chunks."
        )

    print(
        f"✅ Generated {len(vectors)} embeddings "
        f"({len(vectors[0])} dimensions)"
    )

    return vectors


# ---------------------------------------------------------
# Payload creation
# ---------------------------------------------------------

def make_payload(chunk: dict) -> dict:
    """
    Create Qdrant payload from chunk metadata.

    Responsibility:
        chunk -> payload
    """

    return {
        # Chunk information
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

        # Content
        "text": chunk["text"],
        "token_count": chunk["token_count"],

        # Relations
        "parent_chunk": chunk.get("parent_chunk"),
        "chunk_index": chunk.get("chunk_index"),
        "num_chunks": chunk.get("num_chunks"),

        # Graph metadata (Phase 2)
        "annex_number": chunk.get("annex_number"),
        "references": chunk.get(
            "references",
            {"laws": [], "circulars": [], "articles": []},
        ),
    }

# ---------------------------------------------------------
# Point creation
# ---------------------------------------------------------

def build_points(
    chunks: list[dict],
    vectors: list[list[float]],
) -> tuple[list[PointStruct], int]:

    print("\n🔨 Building Qdrant points...")

    if len(chunks) != len(vectors):
        raise RuntimeError(
            "Chunks and vectors count mismatch."
        )

    vector_size = len(vectors[0])

    points = []

    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=make_point_id(chunk["chunk_id"]),
            vector=vector,
            payload=make_payload(chunk),
        )

        points.append(point)

    print(
        f"✅ Created {len(points)} points "
        f"({vector_size} dimensions)"
    )

    return points, vector_size


# ---------------------------------------------------------
# Collection
# ---------------------------------------------------------

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
):
    """
    Create Qdrant collection if missing.
    """

    try:

        client.get_collection(collection_name)

        print(
            f"✅ Collection '{collection_name}' already exists"
        )

    except Exception:

        print(
            f"📦 Creating collection '{collection_name}'..."
        )

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print("✅ Collection created")


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    batch_size: int = BATCH_SIZE,
):
    """
    Upload points into Qdrant.

    Responsibility:
        PointStruct -> Qdrant
    """

    print("\n📤 Uploading points...")

    total = len(points)

    for start in range(0, total, batch_size):

        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

        uploaded = min(start + batch_size, total)

        print(
            f"   {uploaded}/{total} uploaded"
        )

    print(
        f"✅ Uploaded {total} points"
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("INDEXING CHUNKS INTO QDRANT")
    print("=" * 70)


    client = QdrantClient(
        host=os.getenv(
            "QDRANT_HOST",
            "localhost",
        ),
        port=int(
            os.getenv(
                "QDRANT_PORT",
                6333,
            )
        ),
        api_key=os.getenv(
            "QDRANT_API_KEY"
        ),
    )


    chunks = load_chunks(
        CHUNKS_PATH
    )


    vectors = embed_chunks(
        chunks
    )


    points, vector_size = build_points(
        chunks,
        vectors,
    )


    create_collection(
        client,
        COLLECTION_NAME,
        vector_size,
    )


    upload_points(
        client,
        COLLECTION_NAME,
        points,
    )


    print("\n" + "=" * 70)
    print("✅ INDEXING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()