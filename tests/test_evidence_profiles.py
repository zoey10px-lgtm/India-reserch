import unittest

from research.pdf_parser import Page, extract_windows
from research.production import Criterion, EvidenceEngine, PROOF_RESULTS_QUERY, Passage, Target


class PDFSectionTests(unittest.TestCase):
    def test_reference_section_ends_at_appendix_and_does_not_cross_documents(self):
        pages = [
            Page("report-a", 1, "Results ............ 5\r\nReferences ............ 7\r\n"),
            Page("report-a", 2, "3. Results and Discussion\r\nA prototype passed the test.\r\nReferences\r\nFirst paper."),
            Page("report-a", 3, "Second paper.\r\nAppendix A: Additional data\r\nA distinct experiment was run."),
            Page("report-b", 1, "A second report covers evaluation.\r\nReferences\r\nThird paper."),
            Page("report-c", 1, "A third report covers testing."),
        ]
        evidence = extract_windows(pages, min_chars=1)
        self.assertTrue(any("Results ........" in item.text for item in evidence))
        self.assertTrue(any("passed the test" in item.text and item.section == "results" for item in evidence))
        self.assertTrue(any("distinct experiment" in item.text and item.section == "" for item in evidence))
        self.assertTrue(any("third report" in item.text for item in evidence))
        originals = {(page.document_id, page.number): page.text for page in pages}
        for item in evidence:
            self.assertNotIn("paper.", item.text)
            self.assertIn(item.text, originals[item.source_id, item.page])


class EvidenceProfileTests(unittest.TestCase):
    def test_demonstration_requires_completion_and_relevant_environment_in_one_passage(self):
        class Retriever:
            def retrieve(self, query, passages):
                return [(i, .9) for i in range(len(passages))]

        class NLI:
            def predict_batch(self, pairs):
                scores = []
                for text, claim in pairs:
                    if claim.startswith("This report"):
                        supported = "target prototype" in text
                    elif claim == "The test is only planned or proposed, not completed.":
                        supported = "planned" in text
                    elif "completed demonstration" in claim:
                        supported = "completed" in text
                    elif "operated in open water" in claim:
                        supported = "open water" in text
                    else:
                        supported = True
                    scores.append((.98, .01, .01) if supported else (.01, .01, .98))
                return scores

        engine = EvidenceEngine(NLI(), Retriever())
        criteria = [Criterion("C06", "A prototype was demonstrated", 6, evidence_profile="demonstration")]
        target = Target("Target prototype", environment="Open water")
        documents = [
            Passage("report-a", 1, "The target prototype's demonstration in open water was planned."),
            Passage("report-a", 2, "The target prototype completed a laboratory demonstration."),
            Passage("report-a", 3, "The target prototype is intended for open water."),
            Passage("report-b", 4, "A different prototype completed a demonstration in open water."),
        ]
        output = engine.evaluate(documents, criteria, target)
        self.assertEqual(output["status"], "abstained")
        documents.append(Passage("report-a", 5, "The target prototype completed a demonstration in open water."))
        output = engine.evaluate(documents, criteria, target)
        self.assertEqual(output["predicted_trl"], 6)
        self.assertEqual([e["page"] for e in output["criteria"][0]["evidence"]], [5])

    def test_proof_retrieves_results_from_target_report(self):
        class Retriever:
            def retrieve(self, query, passages, *, eligible_indices=None):
                if query == PROOF_RESULTS_QUERY:
                    self.indices = eligible_indices
                    return [(i, .9) for i in eligible_indices]
                return [(0, .9)]

        class NLI:
            def predict_batch(self, pairs):
                scores = []
                for text, claim in pairs:
                    supported = ("target model" in text if claim.startswith("This report")
                                 else "measured result" in text and "performance" in claim)
                    scores.append((.98, .01, .01) if supported else (.01, .01, .98))
                return scores

        retrieval = Retriever()
        engine = EvidenceEngine(NLI(), retrieval)
        criterion = Criterion("C03", "Measured proof", 3, evidence_profile="proof")
        passages = [
            Passage("target", 1, "The target model estimates the number of vehicles."),
            Passage("target", 2, "The measured result of the evaluation is 98% accuracy.", section="results"),
            Passage("unrelated", 1, "An unrelated model's measured result was 99% accuracy.", section="results"),
        ]
        result = engine.evaluate(passages, [criterion], Target("Target model", "Count vehicles"))
        self.assertEqual(retrieval.indices, [1])
        self.assertEqual(result["predicted_trl"], 3)
        self.assertEqual([e["page"] for e in result["criteria"][0]["evidence"]], [2])
