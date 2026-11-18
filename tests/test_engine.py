import unittest
from research.engine import Evidence, Criterion, Inference, assess, target_match

class StubNLI:
    def predict(self, premise, hypothesis):
        if "not tested" in premise.lower():
            return Inference(0.02, 0.96, 0.02)
        if premise.lower() == hypothesis.lower():
            return Inference(0.96, 0.02, 0.02)
        return Inference(0.02, 0.02, 0.96)

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.criteria = [Criterion("C04", 4, "prototype tested in laboratory environment")]
    def test_no_evidence_abstains(self):
        result = assess([], self.criteria, "A", StubNLI())
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["predicted_trl"])
    def test_supported_level(self):
        e = [Evidence(2, "prototype tested in laboratory environment", "A", source_id="report")]
        self.assertEqual(assess(e, self.criteria, "A", StubNLI())["predicted_trl"], 4)
    def test_wrong_target_is_rejected(self):
        e = [Evidence(2, "prototype tested in laboratory environment", "B")]
        self.assertEqual(assess(e, self.criteria, "A", StubNLI())["status"], "abstained")
    def test_contradiction_does_not_pass(self):
        e = [Evidence(2, "prototype not tested in laboratory environment", "A")]
        self.assertEqual(assess(e, self.criteria, "A", StubNLI())["status"], "abstained")
    def test_conflict_is_not_met(self):
        e = [Evidence(2, "prototype tested in laboratory environment", "A"),
             Evidence(3, "prototype not tested in laboratory environment", "A")]
        self.assertEqual(assess(e, self.criteria, "A", StubNLI())["criteria"][0]["status"], "conflict")
    def test_no_skip_of_lower_level(self):
        c = self.criteria + [Criterion("C05", 5, "prototype tested in relevant environment")]
        e = [Evidence(3, "prototype tested in relevant environment", "A")]
        self.assertIsNone(assess(e, c, "A", StubNLI())["predicted_trl"])

if __name__ == "__main__":
    unittest.main()
