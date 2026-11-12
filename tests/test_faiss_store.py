
import unittest
import numpy as np

from src.vectorstore.faiss_store import FAISSVectorStore


class TestFAISSVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = FAISSVectorStore(dimension=3)

        self.chunks = [
            {"text": "Document about cats.", "metadata": {"page": 1}},
            {"text": "Document about dogs.", "metadata": {"page": 2}},
            {"text": "Document about birds.", "metadata": {"page": 3}},
        ]

        self.embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.store.add_documents(
            self.chunks,
            self.embeddings,
        )

    def test_index_size(self):
        self.assertEqual(self.store.index.ntotal, 3)

    def test_search_returns_nearest_document(self):
        query = np.array([0.9, 0.1, 0.0])

        results = self.store.search(query, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["document"]["text"],
            "Document about cats.",
        )

    def test_search_returns_scores(self):
        query = np.array([1.0, 0.0, 0.0])

        results = self.store.search(query, top_k=1)

        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_empty_store(self):
        empty_store = FAISSVectorStore(dimension=3)

        results = empty_store.search(
            np.array([1.0, 0.0, 0.0])
        )

        self.assertEqual(results, [])

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            self.store.search(
                np.array([1.0, 0.0, 0.0]),
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
