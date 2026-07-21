from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

_model = SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Cannot embed empty text")

    return _model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return _model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()