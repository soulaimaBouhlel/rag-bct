#!/usr/bin/env python3
"""
Migration: embeddings.json → Qdrant

Loads all embeddings from JSON and uploads to Qdrant.
Idempotent: safe to run multiple times.
"""

import json
import os
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


class QdrantMigration:
    """Orchestrate migration from JSON to Qdrant."""

    def __init__(
            self,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            api_key: str = None,
            embeddings_file: Path = Path("data/embeddings.json"),
            collection_name: str = "regulations",
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.api_key = api_key
        self.embeddings_file = Path(embeddings_file)
        self.collection_name = collection_name

        # Connect to Qdrant
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=api_key if api_key else None,
            timeout=30.0,
        )

    def load_embeddings(self) -> List[dict]:
        """Load embeddings from JSON file."""
        print(f"\n📂 Loading embeddings from {self.embeddings_file}...")

        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"File not found: {self.embeddings_file}")

        with open(self.embeddings_file, "r", encoding="utf-8") as f:
            embeddings = json.load(f)

        print(f"✅ Loaded {len(embeddings)} embeddings")
        return embeddings

    def create_collection(self) -> None:
        """Create Qdrant collection with optimized schema."""
        print(f"\n🏗️  Creating collection '{self.collection_name}'...")

        # Delete if exists (idempotency)
        try:
            self.client.delete_collection(self.collection_name)
            print(f"   Deleted existing collection (for clean slate)")
        except Exception:
            pass

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=768,  # multilingual-e5-base dimension
                distance=Distance.COSINE,
            ),
        )

        print(f"✅ Created collection '{self.collection_name}'")

    def prepare_points(self, embeddings: List[dict]) -> List[PointStruct]:
        """Convert embeddings to Qdrant point format."""
        print(f"\n⚙️  Preparing {len(embeddings)} points...")

        points = []
        for i, embedding in enumerate(embeddings):
            point = PointStruct(
                id=i,  # Sequential ID (deterministic)
                vector=embedding["embedding"],
                payload={
                    # Core fields (needed for retrieval)
                    "chunk_id": embedding["chunk_id"],
                    "circular_ref": embedding["circular_ref"],
                    "chunk_type": embedding["chunk_type"],
                    "article_number": embedding.get("article_number"),

                    # Document metadata
                    "source_file": embedding["source_file"],
                    "circular_type": embedding["circular_type"],
                    "text": embedding["text"],
                    "title": embedding["title"],
                    "objet": embedding["objet"],
                    "chapter": embedding.get("chapter"),
                    "token_count": embedding["token_count"],

                    # Parent-child tracking (for reconstructing split articles)
                    "parent_chunk": embedding.get("parent_chunk"),
                    "chunk_index": embedding.get("chunk_index"),
                    "num_chunks": embedding.get("num_chunks"),
                },
            )
            points.append(point)

        print(f"✅ Prepared {len(points)} points")
        return points

    def upload_points(self, points: List[PointStruct]) -> None:
        """Upload points to Qdrant in batches."""
        print(f"\n📤 Uploading {len(points)} points (batch size: 100)...")

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            uploaded = min(i + batch_size, len(points))
            print(f"   {uploaded}/{len(points)} uploaded")

        print(f"✅ Upload complete")

    def migrate(self) -> None:
        """Execute full migration."""
        print("\n" + "=" * 70)
        print("MIGRATION: embeddings.json → Qdrant")
        print("=" * 70)

        try:
            embeddings = self.load_embeddings()
            self.create_collection()
            points = self.prepare_points(embeddings)
            self.upload_points(points)

            print("\n" + "=" * 70)
            print("✅ MIGRATION COMPLETE")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"\n❌ MIGRATION FAILED: {e}")
            raise


if __name__ == "__main__":
    # Read configuration from environment
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    migration = QdrantMigration(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        api_key=qdrant_api_key,
    )
    migration.migrate()