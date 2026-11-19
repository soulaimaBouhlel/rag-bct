MODEL_NAME = "intfloat/multilingual-e5-base"

_model = None


def _get_model():
    """
    Load the embedding model on first use instead of at import time,
    so modules that only import the pipeline (unit tests, MCP startup)
    don't pay for — or need — the model download.
    """

    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)

    return _model


def embed(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Cannot embed empty text")

    return _get_model().encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()