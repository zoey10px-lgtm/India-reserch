import unittest
from research.benchmark import evaluate
from research.graph import build_edges

class BenchmarkTests(unittest.TestCase):
    def test_abstention_counts_against_accuracy(self):
        r = evaluate([{"predicted_trl": 4}, {"predicted_trl": None}], [4, 5])
        self.assertEqual(r["accuracy"], 0.5)
        self.assertEqual(r["coverage"], 0.5)
    def test_conflict_candidate(self):
        records = [{"text": "a", "entailment": 0.9, "contradiction": 0.05},
                   {"text": "b", "entailment": 0.05, "contradiction": 0.9}]
        self.assertEqual(build_edges(records)[0].relation, "conflict_candidate")

if __name__ == "__main__":
    unittest.main()
