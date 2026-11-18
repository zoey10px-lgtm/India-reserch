"""Offline unit tests for the real pipeline orchestration (model output stubbed).

These tests validate aggregation and evidence behavior, NOT NLI accuracy.
"""
import unittest
from research.engine import Criterion, Evidence, Inference
from research.pipeline import EvidenceReasoner, EngineSettings


class DeterministicNLI:
    def predict_batch(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            if "not tested" in premise.lower():
                out.append(Inference(0.01, 0.98, 0.01))
            elif "only a future" in hypothesis or ("observed result" in hypothesis and "measured" not in premise):
                out.append(Inference(0.01, 0.01, 0.98))
            else:
                out.append(Inference(0.98, 0.01, 0.01))
        return out


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.engine = EvidenceReasoner(DeterministicNLI(), settings=EngineSettings(
            min_support_weight=0.10, min_target_score=0.35,
        ))
        self.criterion = [Criterion("C04", 4, "prototype tested in laboratory environment")]

    def test_missing_evidence_abstains(self):
        r = self.engine.assess([], self.criterion, "Prototype A")
        self.assertEqual(r["status"], "abstained")

    def test_supported(self):
        e = [Evidence(4, "Prototype A prototype tested in laboratory environment and measured 42 watts", "Prototype A", source_id="report")]
        r = self.engine.assess(e, self.criterion, "Prototype A")
        self.assertEqual(r["predicted_trl"], 4)
        self.assertEqual(r["criteria"][0]["evidence"][0]["page"], 4)

    def test_wrong_target(self):
        e = [Evidence(4, "prototype tested in laboratory environment", "Prototype B", source_id="report")]
        self.assertIsNone(self.engine.assess(e, self.criterion, "Prototype A")["predicted_trl"])

    def test_conflicting_claims(self):
        e = [Evidence(4, "Prototype A prototype tested in laboratory environment and measured 42 watts", "Prototype A"),
             Evidence(5, "Prototype A prototype not tested in laboratory environment", "Prototype A")]
        r = self.engine.assess(e, self.criterion, "Prototype A")
        self.assertEqual(r["criteria"][0]["state"], "conflict")
        self.assertIsNone(r["predicted_trl"])

    def test_no_default_six(self):
        e = [Evidence(2, "A novel theoretical approach was discussed", "Prototype A")]
        self.assertIsNone(self.engine.assess(e, self.criterion, "Prototype A")["predicted_trl"])

if __name__ == "__main__":
    unittest.main()
