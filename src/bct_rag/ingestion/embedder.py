from sentence_transformers import SentenceTransformer


# Load once at startup
MODEL_NAME = "intfloat/multilingual-e5-base"

_model = SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    """
    Generate embedding for a single text.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    return _model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding for multiple texts.
    """
    return _model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()


if __name__ == "__main__":
    sample = """
    Conditions de financement de l'importation.
    """

    vector = embed(sample)

    print(f"Dimension: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")
