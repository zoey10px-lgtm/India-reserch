import unittest
from research.verification import TargetContext, exact_span, quality_gate, grounded_evidence

class VerificationTests(unittest.TestCase):
    def test_contextual_query(self):
        target = TargetContext("sensor A", "flood warning", "field deployment")
        query = target.query("system validation")
        for phrase in ("sensor A", "flood warning", "field deployment", "system validation"):
            self.assertIn(phrase, query)

    def test_exact_span_must_match_page(self):
        pages = {("doc1", 3): "Prototype A was tested in a controlled laboratory."}
        self.assertTrue(exact_span("doc1", 3, "tested in a controlled laboratory", pages))
        self.assertFalse(exact_span("doc1", 2, "tested in a controlled laboratory", pages))
        self.assertFalse(exact_span("doc1", 3, "deployed in the field", pages))

    def test_quality_gate(self):
        self.assertFalse(quality_gate({"C10": "met", "C11": "met", "C12": "insufficient evidence"})["pass"])
        self.assertTrue(quality_gate({k:"met" for k in ("C10","C11","C12")})["pass"])

if __name__ == "__main__":
    unittest.main()
