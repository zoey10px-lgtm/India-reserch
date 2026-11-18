import unittest
from research.criteria_policy import assessment_criteria, parse_criteria, DecisionPolicy

class CriteriaPolicyTests(unittest.TestCase):
    def test_official_ids_not_description_regex(self):
        checklist = {"criteria": [
            {"criterion_id": "C04", "description": "Prototype validated in laboratory"},
            {"criterion_id": "C10", "description": "Requirements versus measurements"},
            {"criterion_id": "C12", "description": "Evidence belongs to assessed technology"},
        ]}
        criteria = parse_criteria(checklist)
        self.assertEqual([c.trl for c in criteria], [4, 0, 0])
        self.assertEqual([c.required for c in criteria], [True, False, False])

    def test_quality_is_not_global_abstention_by_default(self):
        policy = DecisionPolicy()
        self.assertEqual(policy.cap(6, {"C10": "insufficient evidence"}), (6, []))

    def test_explicit_quality_cap(self):
        policy = DecisionPolicy({"C10": 5})
        self.assertEqual(policy.cap(8, {"C10": "insufficient evidence"}), (5, ["C10"]))

    def test_laboratory_profile_uses_worker_checklist_version_without_official_flag(self):
        items = [{"criterion_id": "APCTT-C04", "description": (
            "Laboratory validation (TRL 4): the report states that the assessed component was tested."
        )}]
        official = {"version": "APCTT-2026-CHECKLIST-v1", "criteria": items}
        self.assertTrue(assessment_criteria(official)[0].laboratory_validation)
        self.assertFalse(assessment_criteria({"version": "local-checklist", "criteria": items})[0].laboratory_validation)

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            parse_criteria({"criteria": [{"criterion_id": "CUSTOM", "description": "test"}]})

if __name__ == "__main__":
    unittest.main()
