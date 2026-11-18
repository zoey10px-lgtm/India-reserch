"""Grounded evidence-first TRL reasoning core; no fabricated confidence."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Passage:
    document_id: str
    page: int
    text: str
    target: str = ""
    section: str = ""

@dataclass(frozen=True)
class Criterion:
    id: str
    hypothesis: str
    trl: int | None = None

@dataclass(frozen=True)
class Target:
    technology: str
    application: str = ""
    environment: str = ""
    def query(self, criterion):
        return ". ".join(x for x in (criterion.hypothesis, self.technology, self.application, self.environment) if x)

class EvidenceEngine:
    def __init__(self, nli, retriever, *, support=.70, contradiction=.70, triple_threshold=.65):
        self.nli=nli
        self.retriever=retriever
        self.support=support
        self.contradiction=contradiction
        self.triple_threshold=triple_threshold

    @staticmethod
    def _item(c, status, citations=(), count=0):
        return {"id":c.id,"trl":c.trl,"status":status,"evidence":list(citations),
                "diagnostics":{"candidates":count}}

    def _abstain(self, criteria, reason):
        return {"status":"abstained","predicted_trl":None,"confidence":None,
                "abstention_reason":reason,
                "criteria":[self._item(c,"insufficient evidence") for c in criteria]}

    def evaluate(self, passages, criteria, target, *, remaining_ms=lambda:180000, quality_caps=None):
        if not passages or not criteria:
            return self._abstain(criteria,"No readable passages or criteria")
        result=[]
        hypotheses=[
            "The described artifact was implemented and tested.",
            "The passage identifies the environment or conditions of the test.",
            "The passage reports an observed result or measured outcome.",
            "The test is only planned or proposed, not completed.",
        ]
        for criterion in criteria:
            if remaining_ms()<25000:
                return self._abstain(criteria,"Time budget exhausted")
            ranked=self.retriever.retrieve(target.query(criterion),[p.text for p in passages])
            selected=[]; seen=set()
            for idx, relevance in ranked:
                if not 0<=idx<len(passages) or relevance<.1: continue
                p=passages[idx]
                if p.target and p.target.casefold()!=target.technology.casefold(): continue
                key=(p.document_id,p.text.strip().casefold())
                if key in seen: continue
                seen.add(key)
                selected.append((p,max(0.,min(1.,float(relevance)))))
            if not selected:
                result.append(self._item(criterion,"insufficient evidence"));continue
            prompts=[(p.text,h) for p,_ in selected for h in [criterion.hypothesis,*hypotheses]]
            if remaining_ms()<25000:
                return self._abstain(criteria,"Time budget exhausted before inference")
            predictions=self.nli.predict_batch(prompts)
            if len(predictions)!=len(prompts):
                raise ValueError("NLI output length mismatch")
            scored=[]
            for i,(p,relevance) in enumerate(selected):
                row=predictions[i*5:(i+1)*5]
                if any(len(v)!=3 or any(not math.isfinite(float(x)) or x<0 for x in v) or abs(sum(v)-1)>.02 for v in row):
                    raise ValueError("Invalid NLI probability vector")
                e,c,n=row[0]
                scored.append({"document_id":p.document_id,"page":p.page,
                               "supporting_span":p.text,"relevance":relevance,
                               "entailment":e,"contradiction":c,
                               "artifact":row[1][0],"environment":row[2][0],
                               "outcome":row[3][0],"prospective":row[4][0]})
            positive=[r for r in scored if r["entailment"]>=self.support and r["prospective"]<self.support]
            negative=[r for r in scored if r["contradiction"]>=self.contradiction]
            # Allow artifact, environment and outcome in different cited passages.
            triple=bool(positive) and all(any(r[k]>=self.triple_threshold for r in positive)
                                          for k in ("artifact","environment","outcome"))
            if criterion.id in ("C10","C11","C12"): triple=bool(positive)
            state=("conflict" if positive and negative else "met" if triple and not negative else
                   "contradicted" if negative else "insufficient evidence")
            cited=sorted(positive,key=lambda r:-r["entailment"]*r["relevance"])[:3] if state=="met" else []
            # The span is copied unchanged from a Passage; no generated citations.
            citations=[{k:r[k] for k in ("document_id","page","supporting_span")} for r in cited]
            result.append(self._item(criterion,state,citations,len(scored)))
        states={r["id"]:r["status"] for r in result}
        levels=sorted({c.trl for c in criteria if c.trl is not None})
        chosen=None
        for level in levels:
            ids=[c.id for c in criteria if c.trl==level]
            if ids and all(states[cid]=="met" for cid in ids): chosen=level
            else: break
        blocked=[]
        for cid,cap in (quality_caps or {}).items():
            if cid not in ("C10","C11","C12"):
                raise ValueError("Only C10-C12 may impose quality caps")
            if chosen is not None and states.get(cid)!="met" and chosen>cap:
                chosen=int(cap);blocked.append(cid)
        if chosen is not None and levels and chosen<levels[0]: chosen=None
        return {"status":"completed" if chosen is not None else "abstained",
                "predicted_trl":chosen,"criteria":result,"confidence":None,
                "quality_blockers":blocked,
                "abstention_reason":None if chosen is not None else "insufficient or conflicting evidence"}
