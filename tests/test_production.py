import unittest
from research.production import EvidenceEngine, Passage, Criterion, Target

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
    def test_quality_cap(self):
        p=[Passage("doc",2,"A prototype tested in laboratory and results measured","A")]
        self.assertIsNone(self.engine.evaluate(p,self.criteria,self.target,quality_caps={"C12":3})["predicted_trl"])

if __name__=="__main__": unittest.main()
