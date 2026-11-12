
from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: list[str]
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding
