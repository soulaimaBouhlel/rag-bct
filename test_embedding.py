from bct_rag.ingestion.embedder import embed
import numpy as np


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


text1 = "Conditions de financement de l'importation"
text2 = "Règles de financement des opérations d'importation"
text3 = "Prévisions météorologiques en Tunisie"

e1 = embed(text1)
e2 = embed(text2)
e3 = embed(text3)

print("similar:", cosine(e1, e2))
print("different:", cosine(e1, e3))