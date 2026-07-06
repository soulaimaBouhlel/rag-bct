"""
Embedding generation using multilingual-e5-base.
Produces 768-dimensional vectors.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
_model = None

def _get_model():
    """Lazy-load embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts):
    """
    Generate embeddings for text(s).
    Args:
        texts: str or list[str]
    Returns:
        list[float] (single) or list[list[float]] (batch)
    """
    model = _get_model()
    single = isinstance(texts, str)
    
    if single:
        texts = [texts]
    
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    return vectors[0].tolist() if single else vectors.tolist()


if __name__ == "__main__":
    # Test
    sample = "Conditions de financement de l'importation."
    vec = embed(sample)
    print(f"✅ Embedding dimension: {len(vec)}")
    print(f"   Expected: 768")
    print(f"   Sample values: {vec[:5]}")
