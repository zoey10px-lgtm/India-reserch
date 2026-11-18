import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from participant_module.adapter import AssessmentModule
from research.production import Criterion


class AdapterTests(unittest.TestCase):
    def test_missing_models_fail_before_assessment(self):
        module = AssessmentModule()
        config = {"checklist": {"version": "test", "criteria": [
            {"criterion_id": "C04", "description": "Prototype validation"},
        ]}}
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "APCTT_MODEL_STORE"):
            module.load(config, None)
        self.assertIsNone(module.engine)

    def test_abstention_retains_supported_criterion_evidence(self):
        module = AssessmentModule()
        module.checklist = {"version": "test"}
        module.criteria = [Criterion("C04", "Prototype validation", 4)]
        module.engine = Mock()
        case = {
            "schema_version": "test", "case_id": "fake",
            "assessment_target": {"description": "Prototype"}, "documents": [],
        }
        evidence = [{"document_id": "doc", "page": 1, "supporting_span": "source text"}]
        result = {
            "status": "abstained", "predicted_trl": None,
            "abstention_reason": "insufficient or conflicting evidence",
            "criteria": [{"id": "C04", "status": "met", "evidence": evidence}],
        }
        with patch("participant_module.adapter.assess_pdfs", return_value=result):
            output = module.assess(case, SimpleNamespace(remaining_ms=lambda: 180000))
        self.assertEqual(output["criterion_status"], {"C04": "met"})
        self.assertEqual(output["evidence_reference"], {"C04": evidence})
