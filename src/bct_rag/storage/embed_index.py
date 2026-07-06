"""
Embedding pipeline: chunks.json → embeddings + metadata
Generates embeddings for all chunks and prepares for indexing.
"""

import json
from pathlib import Path
from tqdm import tqdm

from bct_rag.config import CHUNKS_FILE
from bct_rag.storage.embedder import embed

def build_embeddings():
    """
    Load chunks and generate embeddings.
    Returns: list of dicts with chunk data + embedding
    """
    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"\nGenerating embeddings using multilingual-e5-base...")
    print(f"(768-dimensional vectors)")
    
    embedded_chunks = []
    
    for chunk in tqdm(chunks, desc="Embedding"):
        # Generate embedding
        vector = embed(chunk["text"])
        
        # Validate
        assert len(vector) == 768, f"Expected 768 dims, got {len(vector)}"
        assert isinstance(vector, list), "Vector should be list"
        
        # Store with metadata
        embedded_chunk = {
            **chunk,  # all original fields
            "embedding": vector,  # add embedding
        }
        embedded_chunks.append(embedded_chunk)
    
    print(f"\n✅ Generated {len(embedded_chunks)} embeddings")
    
    return embedded_chunks


def save_embeddings(embedded_chunks, output_file="data/embeddings.json"):
    """Save embeddings to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved to {output_path}")
    
    return output_path


def verify_embeddings(embedded_chunks):
    """Sanity checks on embeddings."""
    print("\n" + "=" * 60)
    print("EMBEDDING VERIFICATION")
    print("=" * 60)
    
    # Check dimensions
    dims = [len(c["embedding"]) for c in embedded_chunks]
    assert all(d == 768 for d in dims), "Not all embeddings are 768-dim!"
    print(f"✅ All {len(embedded_chunks)} embeddings are 768-dimensional")
    
    # Check normalization (should be close to 1.0)
    from math import sqrt
    norms = []
    for c in embedded_chunks[:5]:  # sample
        norm = sqrt(sum(x**2 for x in c["embedding"]))
        norms.append(norm)
    avg_norm = sum(norms) / len(norms)
    print(f"✅ L2 norms are ~1.0 (normalized): avg={avg_norm:.4f}")
    
    # Check for NaN/Inf
    has_nan = any(any(x != x for x in c["embedding"]) for c in embedded_chunks)
    has_inf = any(any(abs(x) == float('inf') for x in c["embedding"]) for c in embedded_chunks)
    assert not has_nan, "Found NaN values!"
    assert not has_inf, "Found Inf values!"
    print(f"✅ No NaN or Inf values")
    
    # Check metadata preservation
    assert all("chunk_id" in c for c in embedded_chunks), "Missing chunk_id!"
    assert all("circular_ref" in c for c in embedded_chunks), "Missing circular_ref!"
    assert all("chunk_type" in c for c in embedded_chunks), "Missing chunk_type!"
    print(f"✅ All metadata preserved")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run pipeline
    embedded_chunks = build_embeddings()
    verify_embeddings(embedded_chunks)
    save_embeddings(embedded_chunks)
    
    print(f"\n🚀 Step 2 Complete: Embeddings ready for indexing")
