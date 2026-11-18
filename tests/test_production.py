import unittest
from research.production import (
    LAB_ARTICLE_CLAIMS, LAB_ENVIRONMENT, LAB_OUTCOME, LAB_RESULTS_QUERY,
    EvidenceEngine, Passage, Criterion, Target,
)

class Retriever:
    def retrieve(self, query, passages):
        return [(i,.9) for i in range(len(passages))]

class NLI:
    def predict_batch(self, pairs):
        result=[]
        for text,h in pairs:
            if "only planned" in h:
                result.append((.9,.05,.05) if "not tested" in text else (.02,.03,.95))
            elif "not tested" in text:
                result.append((.01,.98,.01))
            else:
                result.append((.96,.02,.02))
        return result

class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.engine=EvidenceEngine(NLI(),Retriever())
        self.criteria=[Criterion("C04","prototype laboratory validation",4),
                       Criterion("C10","requirements measured against outcomes")]
        self.target=Target("A","flood prediction","laboratory")
    def test_grounded_support(self):
        r=self.engine.evaluate([Passage("doc",2,"A prototype tested in laboratory and results measured","A")],self.criteria,self.target)
        self.assertEqual(r["predicted_trl"],4)
        self.assertEqual(r["criteria"][0]["evidence"][0]["page"],2)
    def test_wrong_target(self):
        r=self.engine.evaluate([Passage("doc",2,"B prototype tested in laboratory","B")],self.criteria,self.target)
        self.assertEqual(r["status"],"abstained")
    def test_conflict(self):
        p=[Passage("doc",2,"A prototype tested in laboratory and results measured","A"),
           Passage("doc",3,"A prototype not tested in laboratory","A")]
        self.assertEqual(self.engine.evaluate(p,self.criteria,self.target)["criteria"][0]["status"],"conflict")
    def test_timeout(self):
        p=[Passage("doc",2,"A prototype tested in laboratory","A")]
        self.assertEqual(self.engine.evaluate(p,self.criteria,self.target,remaining_ms=lambda:1000)["status"],"abstained")
    def test_short_final_criterion_can_finish_inside_remaining_budget(self):
        passages = [Passage("doc", 2, "A prototype tested in laboratory", "A")]
        result = self.engine.evaluate(
            passages, self.criteria[:1], self.target, remaining_ms=lambda: 9000,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["predicted_trl"], 4)

    def test_quality_cap(self):
        p=[Passage("doc",2,"A prototype tested in laboratory and results measured","A")]
        self.assertIsNone(self.engine.evaluate(p,self.criteria,self.target,quality_caps={"C12":3})["predicted_trl"])

    def test_unrelated_partial_evidence_cannot_complete_validation(self):
        class PartialNLI:
            def predict_batch(self, pairs):
                answers = []
                for text, hypothesis in pairs:
                    if "only planned" in hypothesis:
                        answers.append((.01, .01, .98))
                    elif "described artifact" in hypothesis:
                        answers.append((.95, .02, .03) if "artifact" in text else (.02, .02, .96))
                    elif "described setting" in hypothesis:
                        answers.append((.95, .02, .03) if "environment" in text else (.02, .02, .96))
                    elif "observed or measured" in hypothesis:
                        answers.append((.95, .02, .03) if "outcome" in text else (.02, .02, .96))
                    else:
                        answers.append((.95, .02, .03))
                return answers

        engine = EvidenceEngine(PartialNLI(), Retriever())
        passages = [
            Passage("report-a", 1, "A artifact was implemented", "A"),
            Passage("report-b", 2, "A environment was specified", "A"),
            Passage("report-c", 3, "A outcome was measured", "A"),
        ]
        result = engine.evaluate(passages, [Criterion("C04", "validated prototype", 4)], self.target)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["criteria"][0]["status"], "insufficient evidence")


class LaboratoryValidationTests(unittest.TestCase):
    def test_report_scope_and_same_passage_required(self):
        class LabNLI:
            def predict_batch(self, pairs):
                answers = []
                for text, claim in pairs:
                    if claim.startswith("This document describes tests of"):
                        supported = "target" in text
                    elif claim == LAB_ENVIRONMENT:
                        supported = "controlled" in text
                    elif claim == LAB_OUTCOME:
                        supported = "observed" in text
                    elif claim in LAB_ARTICLE_CLAIMS:
                        supported = "test article" in text
                    else:
                        supported = False
                    answers.append((.98, .01, .01) if supported else (.01, .01, .98))
                return answers

        engine = EvidenceEngine(LabNLI(), Retriever())
        criteria = [
            Criterion("C03", "Earlier proof of concept", 3),
            Criterion("C04", "Laboratory validation", 4, laboratory_validation=True),
        ]
        target = Target("Target")
        complete = "test article in controlled conditions with observed results"
        scope = Passage("target-report", 1, "This report tests the target.")
        unrelated = Passage("unrelated-report", 1, complete)
        split = [
            Passage("target-report", 2, "test article in controlled conditions"),
            Passage("target-report", 3, "observed results"),
        ]
        partial = engine.evaluate([scope, unrelated, *split], criteria, target, require_consecutive=False)
        self.assertEqual(partial["criteria"][1]["status"], "insufficient evidence")
        self.assertIsNone(partial["predicted_trl"])

        class HeadingRetriever:
            def retrieve(self, query, passages):
                # Neither retrieval route initially returns the table text.
                return [(1, .9)] if query == LAB_RESULTS_QUERY else [(0, .9)]

        engine = EvidenceEngine(LabNLI(), HeadingRetriever())
        heading = Passage("target-report", 4, "Laboratory Results, Table 1")
        grounded = [scope, heading, Passage("target-report", 4, complete)]
        result = engine.evaluate(grounded, criteria, target, require_consecutive=False)
        self.assertEqual(result["predicted_trl"], 4)
        self.assertEqual(result["criteria"][0]["status"], "insufficient evidence")
        self.assertEqual(result["criteria"][1]["evidence"], [
            {"document_id": "target-report", "page": 4, "supporting_span": complete},
        ])
        self.assertIsNone(engine.evaluate(grounded, criteria, target)["predicted_trl"])

        separate_page = [scope, heading, Passage("target-report", 5, complete)]
        self.assertIsNone(
            engine.evaluate(separate_page, criteria, target, require_consecutive=False)["predicted_trl"]
        )


if __name__=="__main__": unittest.main()
