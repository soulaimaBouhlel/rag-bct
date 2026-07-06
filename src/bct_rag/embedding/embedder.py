from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

_model = SentenceTransformer(MODEL_NAME)

def embed(texts):
    """
    Accepts string or list[str]
    Returns normalized embedding
    """
    single = isinstance(texts, str)

    if single:
        texts = [texts]

    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vectors[0].tolist() if single else vectors.tolist()