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
    evidence_profile: str = ""


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


PROOF_RESULTS_QUERY = (
    "Results of experiments, validation or analysis. Measured performance, "
    "test scores and observed outcomes for the technology's main function."
)
DEMONSTRATION_QUERY = (
    "Prototype deployments and field trials. Where the device was operated, "
    "test conditions and operational observations."
)
PROSPECTIVE = "The test is only planned or proposed, not completed."


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

    def _passing_support(self, pairs, threshold):
        fast = getattr(self.nli, "passing_support", None)
        if fast:
            return fast(pairs, threshold)
        return [row[0] if row[0] >= threshold else 0.0 for row in self._predict(pairs)]

    @staticmethod
    def _following_windows(passages, hits):
        return [
            (index + 1, relevance) for index, relevance in hits
            if 0 <= index < len(passages) - 1
            and passages[index].document_id == passages[index + 1].document_id
            and passages[index].page == passages[index + 1].page
        ]

    def _scope(self, passages, texts, target):
        ranked = self.retriever.retrieve(target.technology, texts)
        anchors = {}
        for index, passage in enumerate(passages):
            anchors.setdefault(passage.document_id, index)
        indices = dict.fromkeys([*anchors.values(), *(i for i, _ in ranked if 0 <= i < len(passages))])
        claims = (f"This report evaluates {target.technology}.", f"This report describes {target.technology}.")
        pairs = [(passages[index].text, claim) for index in indices for claim in claims]
        scores = self._passing_support(pairs, self.support)
        return {
            passages[index].document_id for i, index in enumerate(indices)
            if any(scores[i * len(claims):(i + 1) * len(claims)])
        }

    def _profile_support(self, profile, selected, target):
        if profile == "proof":
            task = re.split(r"\s+and\s+", target.application.strip().rstrip("."), maxsplit=1)[0]
            task = task[:1].lower() + task[1:]
            claims = ["The results show how well the technology performed its intended function."]
            if task:
                claims.extend((
                    f"The results show the performance of the technology when used to {task}.",
                    f"The results show the performance of the model when used to {task}.",
                ))
        else:
            claims = ["The representative prototype was operated during a completed demonstration."]
            if target.environment.strip():
                environment = target.environment.strip().rstrip(".")
                environment = environment[:1].lower() + environment[1:]
                claims.append(f"The prototype was operated in {environment}.")
            else:
                claims.append("The prototype was operated under conditions representative of its intended use.")
            pairs = [(passage.text, claim) for passage, _ in selected for claim in claims]
            scores = self._passing_support(pairs, self.triple_threshold)
            return [
                min(scores[i:i + len(claims)])
                for i in range(0, len(scores), len(claims))
            ]
        pairs = [(passage.text, claim) for passage, _ in selected for claim in claims]
        scores = self._passing_support(pairs, self.support)
        return [max(scores[start:start + len(claims)]) for start in range(0, len(scores), len(claims))]

    def evaluate(self, passages, criteria, target, *, remaining_ms=lambda:180000,
                 quality_caps=None, require_consecutive=True):
        if not passages or not criteria:
            return self.abstain(criteria,"No readable passages or criteria")
        result=[]
        checks = {}
        profile_scope = None
        passage_texts = [p.text for p in passages]
        hypotheses=[
            "The described artifact was implemented and tested.",
            "Testing took place in a described setting.",
            "A result from the test was observed or measured.",
            PROSPECTIVE,
        ]
        for index, criterion in enumerate(criteria):
            # Reserve progressively less time as the checklist gets shorter.
            # The worker deadline still applies to the entire evaluation.
            reserve_ms = min(25_000, 3_000 + 4_000 * (len(criteria) - index))
            if remaining_ms() < reserve_ms:
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
                ranked = [
                    *ranked, *self.retriever.retrieve(LAB_EVIDENCE_QUERY, passage_texts),
                    *result_hits, *self._following_windows(passages, result_hits),
                ]
            elif criterion.evidence_profile:
                if profile_scope is None:
                    profile_scope = self._scope(passages, passage_texts, target)
                if criterion.evidence_profile == "proof":
                    results = [i for i, passage in enumerate(passages)
                               if passage.section == "results" and passage.document_id in profile_scope]
                    if results:
                        ranked = [
                            *ranked, *self.retriever.retrieve(PROOF_RESULTS_QUERY, passage_texts,
                                                                eligible_indices=results),
                        ]
                    else:
                        ranked = [*ranked, *self.retriever.retrieve(PROOF_RESULTS_QUERY, passage_texts)]
                else:
                    query = " ".join((target.technology, target.environment, DEMONSTRATION_QUERY))
                    demo_hits = self.retriever.retrieve(query, passage_texts)
                    ranked = [*ranked, *demo_hits, *self._following_windows(passages, demo_hits)]

            selected=[]; seen=set()
            for idx, relevance in ranked:
                if not 0 <= idx < len(passages) or not math.isfinite(relevance): continue
                p=passages[idx]
                if criterion.laboratory_validation and p.document_id not in scoped_docs:
                    continue
                if criterion.evidence_profile and p.document_id not in profile_scope:
                    continue
                if p.target and p.target.casefold()!=target.technology.casefold(): continue
                key=(p.document_id,p.text.strip().casefold())
                if key in seen: continue
                seen.add(key)
                selected.append((p,max(0.,min(1.,float(relevance)))))
            if not selected:
                result.append(self._item(criterion,"insufficient evidence"));continue
            if remaining_ms() < reserve_ms:
                return self.abstain(criteria, "Time budget exhausted before inference", budget_exhausted=True)
            predictions = self._predict([(p.text, criterion.hypothesis) for p, _ in selected])
            needs_completed_test = criterion.trl is not None and criterion.trl >= 3
            needs_triple = criterion.trl in (4, 5)
            if criterion.laboratory_validation:
                outcomes = self._passing_support([(p.text, LAB_OUTCOME) for p, _ in selected], self.support)
                measured = [p.text for (p, _), score in zip(selected, outcomes) if score]
                claims = (LAB_ENVIRONMENT, *LAB_ARTICLE_CLAIMS)
                facts = self._passing_support(
                    [(text, claim) for text in measured for claim in claims], self.support,
                )
                by_passage = {}
                for index, text in enumerate(measured):
                    start = index * len(claims)
                    environment, *article = facts[start:start + len(claims)]
                    by_passage[text] = min(environment, max(article))
                support_scores = [
                    min(by_passage.get(p.text, 0.0), outcome)
                    for (p, _), outcome in zip(selected, outcomes)
                ]
            elif criterion.evidence_profile:
                support_scores = self._profile_support(criterion.evidence_profile, selected, target)
            else:
                support_scores = [scores[0] for scores in predictions]
            candidates = [
                (p, relevance, score)
                for (p, relevance), score in zip(selected, support_scores)
                if score >= (self.triple_threshold if criterion.evidence_profile == "demonstration" else self.support)
            ]
            # Artifact, environment and outcome must come from the same passage.
            # Cache these criterion-independent checks across the checklist.
            required = []
            if needs_completed_test:
                required = [hypotheses[3]]
            if not criterion.laboratory_validation and not criterion.evidence_profile:
                if criterion.trl is not None and criterion.trl >= 4:
                    required.append(hypotheses[0])
                if needs_triple:
                    required.extend(hypotheses[1:3])
            missing = list(dict.fromkeys(
                (p.text, h) for p, _, _ in candidates for h in required if (p.text, h) not in checks
            ))
            if missing:
                scores = self._passing_support(missing, min(self.support, self.triple_threshold))
                checks.update(zip(missing, scores))
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
