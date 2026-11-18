"""Grounded evidence-first TRL reasoning core; no fabricated confidence."""
from dataclasses import dataclass
import math
import re

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
    laboratory_validation: bool = False


LAB_EVIDENCE_QUERY = (
    "Measured results of laboratory tests on a model or component. "
    "Experimental setup, conditions and observed performance."
)
LAB_RESULTS_QUERY = "Laboratory tests of a prototype, model or component with measured outcomes."
LAB_ARTICLE_CLAIMS = (
    "A scaled physical model was used in the test.",
    "An early version, breadboard, or component was used as the test article.",
)
LAB_ENVIRONMENT = "A test article was tested in a controlled laboratory experiment."
LAB_OUTCOME = "The results of the experiment include actual measurements or observed performance."


@dataclass(frozen=True)
class Target:
    technology: str
    application: str = ""
    environment: str = ""
    def query(self, criterion):
        parts = (self.technology, LAB_EVIDENCE_QUERY) if criterion.laboratory_validation else (
            criterion.hypothesis, self.technology, self.application, self.environment,
        )
        return ". ".join(part.rstrip(".") for part in parts if part)

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

    def abstain(self, criteria, reason, *, budget_exhausted=False):
        return {"status":"abstained","predicted_trl":None,"confidence":None,
                "abstention_reason":reason, "budget_exhausted":budget_exhausted,
                "criteria":[self._item(c,"insufficient evidence") for c in criteria]}

    def _predict(self, pairs):
        scores = self.nli.predict_batch(pairs)
        if len(scores) != len(pairs):
            raise ValueError("NLI output length mismatch")
        for row in scores:
            if (len(row) != 3 or any(not math.isfinite(x) or x < 0 for x in row)
                    or abs(sum(row) - 1) > 0.02):
                raise ValueError("Invalid NLI probability vector")
        return scores

    def evaluate(self, passages, criteria, target, *, remaining_ms=lambda:180000,
                 quality_caps=None, require_consecutive=True):
        if not passages or not criteria:
            return self.abstain(criteria,"No readable passages or criteria")
        result=[]
        checks = {}
        passage_texts = [p.text for p in passages]
        hypotheses=[
            "The described artifact was implemented and tested.",
            "Testing took place in a described setting.",
            "A result from the test was observed or measured.",
            "The test is only planned or proposed, not completed.",
        ]
        for criterion in criteria:
            if remaining_ms()<25000:
                return self.abstain(criteria, "Time budget exhausted", budget_exhausted=True)
            ranked = self.retriever.retrieve(target.query(criterion), passage_texts)
            if criterion.laboratory_validation:
                scoped = [
                    passages[index] for index, _ in ranked
                    if 0 <= index < len(passages)
                ]
                scope_claim = f"This document describes tests of {target.technology}."
                scope_scores = self._predict([(passage.text, scope_claim) for passage in scoped])
                scoped_docs = {
                    passage.document_id for passage, score in zip(scoped, scope_scores)
                    if score[0] >= self.support
                }
                result_hits = self.retriever.retrieve(LAB_RESULTS_QUERY, passage_texts)
                # Include adjoining windows on the page. A table heading can rank
                # above the rows; the rows still need to pass the same NLI checks.
                following_windows = [
                    (index + 1, relevance) for index, relevance in result_hits
                    if 0 <= index < len(passages) - 1
                    and passages[index].document_id == passages[index + 1].document_id
                    and passages[index].page == passages[index + 1].page
                ]
                ranked = [
                    *ranked, *self.retriever.retrieve(LAB_EVIDENCE_QUERY, passage_texts),
                    *result_hits, *following_windows,
                ]
            selected=[]; seen=set()
            for idx, relevance in ranked:
                if not 0 <= idx < len(passages) or not math.isfinite(relevance): continue
                p=passages[idx]
                if criterion.laboratory_validation and p.document_id not in scoped_docs:
                    continue
                if p.target and p.target.casefold()!=target.technology.casefold(): continue
                key=(p.document_id,p.text.strip().casefold())
                if key in seen: continue
                seen.add(key)
                selected.append((p,max(0.,min(1.,float(relevance)))))
            if not selected:
                result.append(self._item(criterion,"insufficient evidence"));continue
            if remaining_ms() < 25000:
                return self.abstain(criteria, "Time budget exhausted before inference", budget_exhausted=True)
            predictions = self._predict([(p.text, criterion.hypothesis) for p, _ in selected])
            needs_completed_test = criterion.trl is not None and criterion.trl >= 3
            needs_triple = criterion.trl in (4, 5)
            if criterion.laboratory_validation:
                outcomes = self._predict([(p.text, LAB_OUTCOME) for p, _ in selected])
                measured = [p.text for (p, _), score in zip(selected, outcomes) if score[0] >= self.support]
                claims = (LAB_ENVIRONMENT, *LAB_ARTICLE_CLAIMS)
                facts = self._predict([(text, claim) for text in measured for claim in claims]) if measured else []
                by_passage = {}
                for index, text in enumerate(measured):
                    start = index * len(claims)
                    environment, *article = (row[0] for row in facts[start:start + len(claims)])
                    by_passage[text] = min(environment, max(article))
                support_scores = [
                    min(by_passage.get(p.text, 0.0), outcome[0])
                    for (p, _), outcome in zip(selected, outcomes)
                ]
            else:
                support_scores = [scores[0] for scores in predictions]
            candidates = [
                (p, relevance, score)
                for (p, relevance), score in zip(selected, support_scores)
                if score >= self.support
            ]
            # Artifact, environment and outcome must come from the same passage.
            # Cache these criterion-independent checks across the checklist.
            required = []
            if needs_completed_test:
                required = [hypotheses[3]]
            if not criterion.laboratory_validation:
                if criterion.trl is not None and criterion.trl >= 4:
                    required.append(hypotheses[0])
                if needs_triple:
                    required.extend(hypotheses[1:3])
            missing = list(dict.fromkeys(
                (p.text, h) for p, _, _ in candidates for h in required if (p.text, h) not in checks
            ))
            if missing:
                checks.update((pair, score[0]) for pair, score in zip(missing, self._predict(missing)))
            qualifying = []
            for p, relevance, entailment in candidates:
                if needs_completed_test and checks[p.text, hypotheses[3]] >= self.support:
                    continue
                if any(checks[p.text, h] < self.triple_threshold for h in required if h != hypotheses[3]):
                    continue
                qualifying.append({
                    "document_id": p.document_id, "page": p.page, "supporting_span": p.text,
                    "entailment": entailment, "relevance": relevance,
                })
            negative = any(scores[1] >= self.contradiction for scores in predictions)
            state = ("conflict" if qualifying and negative else "contradicted" if negative else
                     "met" if qualifying else "insufficient evidence")
            cited = sorted(qualifying, key=lambda r: -r["entailment"] * r["relevance"])[:3] if state == "met" else []
            # The span is copied unchanged from a Passage; no generated citations.
            citations=[{k:r[k] for k in ("document_id","page","supporting_span")} for r in cited]
            result.append(self._item(criterion,state,citations,len(selected)))
        states={r["id"]:r["status"] for r in result}
        levels=sorted({c.trl for c in criteria if c.trl is not None})
        chosen=None
        for level in levels:
            ids=[c.id for c in criteria if c.trl==level]
            if all(states[cid] == "met" for cid in ids):
                chosen = level
            elif require_consecutive:
                break
        if remaining_ms() <= 0:
            return self.abstain(criteria, "Time budget exhausted after inference", budget_exhausted=True)
        blocked=[]
        for cid,cap in (quality_caps or {}).items():
            if cid not in ("C10","C11","C12"):
                raise ValueError("Only C10-C12 may impose quality caps")
            matching = next((key for key in states if key == cid or key.endswith("-" + cid)), cid)
            if chosen is not None and states.get(matching)!="met" and chosen>cap:
                chosen=int(cap);blocked.append(cid)
        if chosen is not None and levels and chosen<levels[0]: chosen=None
        return {"status":"completed" if chosen is not None else "abstained",
                "predicted_trl":chosen,"criteria":result,"confidence":None,
                "quality_blockers":blocked,
                "abstention_reason":None if chosen is not None else "insufficient or conflicting evidence"}
