
import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values.")

        vectors = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(vectors)
        self.documents.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if query_embedding.size != self.dimension:
            raise ValueError("Query embedding dimension mismatch.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query contains non-finite values.")

        if self.index.ntotal == 0:
            return []

        query_vector = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "document": self.documents[idx],
                "score": float(score),
            })

        return results
