"""
ChromaDB schema converter.
Transforms embedded chunks into ChromaDB-native format.
"""

def convert_to_chroma_format(embedded_chunks):
    """
    Convert embedded chunks to ChromaDB format.
    
    Args:
        embedded_chunks: list of dicts from embed_index.py
        
    Returns:
        dict with keys: ids, documents, embeddings, metadatas
    """
    chroma_docs = {
        "ids": [],
        "documents": [],
        "embeddings": [],
        "metadatas": []
    }
    
    for chunk in embedded_chunks:
        chroma_docs["ids"].append(chunk["chunk_id"])
        chroma_docs["documents"].append(chunk["text"])
        chroma_docs["embeddings"].append(chunk["embedding"])
        
        # Extract essential metadata for ChromaDB
        # Indexed fields: circular_ref, chunk_type, article_number, chapter (optional)
        # Stored-only fields: all others
        metadata = {
            # Indexed (filterable)
            "circular_ref": chunk["circular_ref"],
            "chunk_type": chunk["chunk_type"],
            "article_number": chunk.get("article_number"),
            "chapter": chunk.get("chapter"),
            
            # Stored only (not indexed)
            "source_file": chunk["source_file"],
            "circular_type": chunk["circular_type"],
            "article_label": chunk.get("article_label", ""),
            "title": chunk["title"],
            "objet": chunk["objet"],
            "page_hint": chunk.get("page_hint", 0),
            "token_count": chunk["token_count"],
            
            # Parent-child tracking
            "parent_chunk": chunk.get("parent_chunk"),
            "chunk_index": chunk.get("chunk_index"),
            "num_chunks": chunk.get("num_chunks"),
        }
        
        chroma_docs["metadatas"].append(metadata)
    
    return chroma_docs


def validate_chroma_format(chroma_docs):
    """Validate the ChromaDB format before indexing."""
    n_ids = len(chroma_docs["ids"])
    n_docs = len(chroma_docs["documents"])
    n_emb = len(chroma_docs["embeddings"])
    n_meta = len(chroma_docs["metadatas"])
    
    assert n_ids == n_docs == n_emb == n_meta, \
        f"Length mismatch: ids={n_ids}, docs={n_docs}, emb={n_emb}, meta={n_meta}"
    
    # Validate IDs are unique
    unique_ids = len(set(chroma_docs["ids"]))
    assert unique_ids == n_ids, \
        f"Duplicate IDs found: {n_ids} total, {unique_ids} unique"
    
    # Validate embedding dimensions
    dims = {len(e) for e in chroma_docs["embeddings"]}
    assert len(dims) == 1, f"Inconsistent embedding dims: {dims}"
    
    dim = list(dims)[0]
    assert dim == 768, f"Expected 768-dim embeddings, got {dim}"
    
    print(f"✅ ChromaDB format valid: {n_ids} documents, 768-dim embeddings")


if __name__ == "__main__":
    # Test
    from embed_index import build_embeddings
    from chroma_schema import convert_to_chroma_format, validate_chroma_format
    
    embedded_chunks = build_embeddings()
    chroma_docs = convert_to_chroma_format(embedded_chunks)
    validate_chroma_format(chroma_docs)
