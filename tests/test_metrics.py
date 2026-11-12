import unittest

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    judgment_coverage_at_k,
)


class TestRetrievalMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        retrieved = ["A", "B", "C", "D", "E"]
        qrels = {"A": 3, "B": 0, "C": 2, "D": 1, "E": 0, "F": 3}

        self.assertEqual(
            precision_at_k(retrieved, qrels, k=5), 0.4
        )
    def test_recall_at_k(self):
        retrieved = ["A", "B", "C", "D", "E"]
        qrels = {
            "A": 3, "B": 0, "C": 2,
            "D": 1, "E": 0, "F": 3
        }

        self.assertAlmostEqual(
            recall_at_k(retrieved, qrels, k=5), 2 / 3
        )


    def test_reciprocal_rank_at_k(self):
        retrieved = ["A", "B", "C", "D", "E"]
        qrels = {
            "A": 0, "B": 0, "C": 3,
            "D": 2, "E": 0
        }

        self.assertAlmostEqual(
            reciprocal_rank_at_k(retrieved, qrels, k=5), 1 / 3
        )


    def test_no_relevant_results(self):
        retrieved = ["A", "B", "C"]
        qrels = {"A": 0, "B": 1, "C": 0}

        self.assertEqual(precision_at_k(retrieved, qrels), 0.0)
        self.assertEqual(recall_at_k(retrieved, qrels), 0.0)
        self.assertEqual(reciprocal_rank_at_k(retrieved, qrels), 0.0)

    

    def test_unjudged_candidates(self):
        retrieved = ["A", "B", "C", "D", "E"]
        qrels = {"A": 3, "B": 0, "C": 2}

        self.assertAlmostEqual(
            precision_at_k(retrieved, qrels), 2 / 3
        )

        self.assertAlmostEqual(
            recall_at_k(retrieved, qrels), 1.0
        )

        self.assertAlmostEqual(
            reciprocal_rank_at_k(retrieved, qrels), 1.0
        )

        self.assertAlmostEqual(
            judgment_coverage_at_k(retrieved, qrels), 3 / 5
        )

