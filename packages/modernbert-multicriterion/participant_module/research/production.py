"""Grounded evidence-first TRL reasoning core; no fabricated confidence."""
from dataclasses import dataclass
import logging
import math
import re

logger = logging.getLogger(__name__)

VALIDATION_SUPPORT = 0.60
PLAN_MARGIN = 0.15
OPERATION_SEARCH_TERMS = {
    6: ("prototype", "demonstrated", "tested", "validation", "vacuum", "measured"),
    7: ("flight", "mission", "operational", "demonstrated", "prototype", "achieved", "reached"),
    9: ("operational", "deployed", "mission", "operated", "successful"),
}

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
OPERATIONAL_DEPLOYMENT_CLAIMS = (
    "The final system had already entered routine service and was used successfully outside a test or evaluation campaign.",
    "The report states a period of actual operational service for the final system.",
)
PROSPECTIVE = "The test is only planned or proposed, not completed."
CONCEPT_HEADING = re.compile(r"(?im)^[ \t]*(?:abstract|purpose and scope|executive summary)[ \t]*\r?$")
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
COMPARISON_METRIC = re.compile(
    r"\b(?:root[\s-]+mean[\s-]+square[\s-]+error|mean[\s-]+absolute[\s-]+error|"
    r"mean[\s-]+squared[\s-]+error|RMSE|coefficient of determination)\b", re.I,
)
TEST_REFERENCE = re.compile(
    r"\b(test|trial|run|experiment|deployment)\s+(?:no\.?\s*|#\s*)?"
    r"([a-z]?\d+(?:[-.]\d+)*|(?-i:[A-Z]))\b",
    re.IGNORECASE,
)
REVIEW_MARGIN_MS = 8_000
MAX_LINKED_PAIRS = 3


@dataclass(frozen=True)
class Target:
    technology: str
    application: str = ""
    environment: str = ""
    def hypothesis(self, claim: str) -> str:
        technology = " ".join(self.technology.split())
        if not technology:
            return claim
        return f"For the assessed technology ({technology}): {claim}"

    def query(self, criterion):
        parts = (self.technology, LAB_EVIDENCE_QUERY) if criterion.laboratory_validation else (
            criterion.hypothesis, self.technology, self.application, self.environment,
        )
        return ". ".join(part.rstrip(".") for part in parts if part)

class EvidenceEngine:
    def __init__(self, nli, retriever, *, support=.70, contradiction=.70, triple_threshold=.65):
        self.evidence_classifier = None
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
            subject = target.technology.strip().rstrip(".") or "The assessed system"
            environment = target.environment.strip().rstrip(".")
            prototype = f"The prototype of {subject}"
            if environment:
                completed = f"{prototype} was subjected to completed testing in {environment}."
                future = f"Testing of the prototype of {subject} in {environment} is only planned."
            else:
                completed = f"{prototype} was subjected to testing that simulated the conditions of its intended use."
                future = f"Testing of the prototype of {subject} in conditions simulating its intended use is only planned."
            pairs = [(" ".join(passage.text.split()), claim)
                     for passage, _ in selected for claim in (completed, future)]
            predictions = self._predict(pairs)
            return [
                actual[0] if actual[0] - planned[0] >= PLAN_MARGIN else 0.0
                for actual, planned in zip(predictions[::2], predictions[1::2])
            ]
        pairs = [(passage.text, claim) for passage, _ in selected for claim in claims]
        scores = self._passing_support(pairs, self.support)
        return [max(scores[start:start + len(claims)]) for start in range(0, len(scores), len(claims))]

    def _focused_support(self, passages, anchor, claims):
        excerpts = {}
        for passage in passages:
            match = anchor.search(passage.text)
            if match is None:
                continue
            span = passage.text[match.start():match.start() + 560].rstrip()
            if anchor is CONCEPT_HEADING:
                # Avoid adding a clipped sentence about another system to the NLI premise.
                end = max((boundary.end() for boundary in SENTENCE_END.finditer(span)), default=0)
                if end >= 120:
                    span = span[:end]
            if len(span) >= 120:
                excerpts.setdefault((passage.document_id, passage.page, span), None)
        candidates = list(excerpts)[:4]
        scores = self._passing_support(
            [(span, claim) for _, _, span in candidates for claim in claims], self.support,
        )
        for index, (document_id, page, span) in enumerate(candidates):
            if all(scores[index * len(claims):(index + 1) * len(claims)]):
                yield {"document_id": document_id, "page": page, "supporting_span": span}

    @staticmethod
    def _test_reference(passage):
        references = {(kind.casefold(), label.casefold())
                      for kind, label in TEST_REFERENCE.findall(passage.text)}
        return next(iter(references)) if len(references) == 1 else None

    def _linked_support(self, passages, selected, criterion, target, hypotheses):
        # An explicit test reference in both windows is required before joining them.
        positions = {passage: index for index, passage in enumerate(passages)}
        anchors = {}
        for passage, _ in selected:
            reference = self._test_reference(passage)
            if reference:
                anchors.setdefault((passage.document_id, reference), []).append(passage)
        if not anchors:
            return ()
        pairs = {}
        for passage in passages:
            reference = self._test_reference(passage)
            if passage.target and passage.target.casefold() != target.technology.casefold():
                continue
            for anchor in anchors.get((passage.document_id, reference), ()):
                if anchor == passage or abs(anchor.page - passage.page) > 1:
                    continue
                if anchor.text in passage.text or passage.text in anchor.text:
                    continue
                earlier, later = sorted((anchor, passage), key=positions.__getitem__)
                premise = f"{earlier.text}\n\n{later.text}"
                if len(premise) <= 1500:
                    pairs.setdefault(premise, (earlier, later, reference))
                if len(pairs) == MAX_LINKED_PAIRS:
                    break
            if len(pairs) == MAX_LINKED_PAIRS:
                break
        if not pairs:
            return ()

        required = [target.hypothesis(criterion.hypothesis), *hypotheses[:2],
                    "The same test article was used for the test and the measured results."]
        queries = []
        for premise, (_, _, (kind, label)) in pairs.items():
            test = f"{kind.capitalize()} {label.upper()}"
            measured = f"{test} tested {target.technology} and reported measured results for it."
            queries.extend((premise, claim) for claim in (*required, measured, PROSPECTIVE))
        scores = self._passing_support(queries, self.triple_threshold)
        width = len(required) + 2
        for index, (earlier, later, _) in enumerate(pairs.values()):
            criterion_score, *facts, prospective = scores[index * width:(index + 1) * width]
            if criterion_score >= self.support and all(facts) and prospective < self.support:
                return earlier, later
        return ()

    def _has_scoped_contradiction(self, negatives, criterion, target, supporting_passages, *, test_specific):
        scope = f"The passage discusses test evidence or maturity of {target.technology} itself."
        target_name = r"\s+".join(map(re.escape, target.technology.split()))
        target_pattern = re.compile(rf"(?<!\w){target_name}(?!\w)", re.IGNORECASE)
        # Distinct numbered tests in the same document cannot refute one another
        # just by reporting different conditions or outcomes.
        eligible = []
        named_claims = []
        for passage in negatives:
            reference = self._test_reference(passage)
            if test_specific and supporting_passages and reference and all(
                passage.document_id == support.document_id
                and (support_ref := self._test_reference(support))
                and support_ref[0] == reference[0] and support_ref != reference
                for support in supporting_passages
            ):
                continue
            sentences = re.split(r"(?<=[.!?])\s+", passage.text.strip())
            named_sentences = [sentence for sentence in sentences if target_pattern.search(sentence)]
            if named_sentences:
                if len(sentences) == 1:
                    # This is already a high-scoring denial and names the target.
                    return True
                named_claims.extend((sentence, target.hypothesis(criterion.hypothesis)) for sentence in named_sentences)
                continue
            eligible.append((passage.text, scope))
        return (bool(named_claims) and any(scores[1] >= self.contradiction
                                          for scores in self._predict(named_claims))) or any(
            self._passing_support(eligible, self.support)
        )

    def evaluate(self, passages, criteria, target, *, remaining_ms=lambda:180000,
                 quality_caps=None, require_consecutive=True, source_pages=()):
        if not passages or not criteria:
            return self.abstain(criteria,"No readable passages or criteria")
        result=[]
        checks = {}
        profile_scope = None
        source_pages = source_pages or passages
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
            query = target.query(criterion)
            terms = OPERATION_SEARCH_TERMS.get(criterion.trl)
            if terms:
                query += ". " + " ".join(terms)
                ranked = self.retriever.retrieve(query, passage_texts, evidence_terms=terms)
                if logger.isEnabledFor(logging.INFO):
                    logger.info("%s pool_pages=%s reranked_pages=%s", criterion.id,
                                sorted({passages[i].page for i in self.retriever.last_pool}),
                                sorted({passages[i].page for i, _ in ranked}))
            else:
                ranked = self.retriever.retrieve(query, passage_texts)
            if criterion.laboratory_validation:
                result_hits = self.retriever.retrieve(LAB_RESULTS_QUERY, passage_texts)
                # Include adjoining windows on the page. A table heading can rank
                # above the rows; the rows still need to pass the same NLI checks.
                ranked = [
                    *ranked, *self.retriever.retrieve(LAB_EVIDENCE_QUERY, passage_texts),
                    *result_hits, *self._following_windows(passages, result_hits),
                ]
            elif criterion.evidence_profile in ("proof", "demonstration"):
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
                    if remaining_ms() > reserve_ms + REVIEW_MARGIN_MS:
                        eligible = [i for i, passage in enumerate(passages)
                                    if passage.document_id in profile_scope]
                        if eligible:
                            result_query = (
                                f"{target.technology}. Completed field or flight tests, observed "
                                "performance, measured test results and achieved readiness."
                            )
                            ranked.extend(self.retriever.retrieve(
                                result_query, passage_texts, eligible_indices=eligible,
                            ))

            if criterion.evidence_profile == "demonstration" and remaining_ms() > reserve_ms + REVIEW_MARGIN_MS:
                neighbors = {}
                for index, relevance in ranked:
                    if not 0 <= index < len(passages):
                        continue
                    current = passages[index]
                    for neighbor_index in range(index + 1, min(index + 3, len(passages))):
                        following = passages[neighbor_index]
                        if (current.document_id, current.page) != (following.document_id, following.page):
                            break
                        neighbors[neighbor_index] = max(relevance, neighbors.get(neighbor_index, 0.0))
                ranked.extend(sorted(neighbors.items(), key=lambda item: -item[1])[:12])

            selected=[]; seen=set()
            page_counts = {}
            page_limit = 2
            high_relevance_cutoff = 0.5
            for idx, relevance in sorted(ranked, key=lambda hit: -hit[1]):
                if not 0 <= idx < len(passages) or not math.isfinite(relevance): continue
                p=passages[idx]
                if criterion.evidence_profile == "proof" and p.document_id not in profile_scope:
                    continue
                if p.target and p.target.casefold()!=target.technology.casefold(): continue
                key=(p.document_id,p.text.strip().casefold())
                if key in seen: continue
                page = (p.document_id, p.page)
                if page_counts.get(page, 0) >= page_limit:
                    continue
                if len(selected) >= 5 and relevance < high_relevance_cutoff:
                    continue
                seen.add(key)
                page_counts[page] = page_counts.get(page, 0) + 1
                selected.append((p,max(0.,min(1.,float(relevance)))))
            if logger.isEnabledFor(logging.INFO):
                logger.info("%s retrieved=%s selected=%s", criterion.id,
                            sorted({passages[i].page for i, _ in ranked if 0 <= i < len(passages)}),
                            sorted({passage.page for passage, _ in selected}))
            trace = criterion.id in ("APCTT-C06", "APCTT-C09")
            if trace and logger.isEnabledFor(logging.DEBUG):
                ranked_pages = sorted({(passages[i].document_id, passages[i].page)
                                       for i, _ in ranked if 0 <= i < len(passages)})
                selected_pages = sorted({(p.document_id, p.page) for p, _ in selected})
                logger.debug("%s ranked pages=%s selected pages=%s",
                             criterion.id, ranked_pages, selected_pages)
            if not selected and criterion.evidence_profile != "proof":
                result.append(self._item(criterion,"insufficient evidence"));continue
            if remaining_ms() < reserve_ms:
                return self.abstain(criteria, "Time budget exhausted before inference", budget_exhausted=True)
            hypothesis = target.hypothesis(criterion.hypothesis)
            predictions = self._predict([(p.text, hypothesis) for p, _ in selected])
            if logger.isEnabledFor(logging.INFO):
                for (passage, _), scores in zip(selected, predictions):
                    logger.info("%s nli_page=%s score=%.4f hypothesis=%r premise=%r",
                                criterion.id, passage.page, scores[0], hypothesis, passage.text)
            needs_completed_test = criterion.trl is not None and criterion.trl >= 3
            needs_triple = criterion.trl in (4, 5)
            learned_support = None
            if self.evidence_classifier is not None and trace and selected and logger.isEnabledFor(logging.DEBUG):
                learned_support = self.evidence_classifier.support(
                    target.technology, criterion.hypothesis, [p for p, _ in selected],
                    remaining_ms, reserve_ms + REVIEW_MARGIN_MS,
                )
            if criterion.laboratory_validation:
                lab_claim = target.hypothesis(
                    "A breadboard or component was tested in a laboratory and its performance was measured."
                )
                support_scores = self._passing_support(
                    [(p.text, lab_claim) for p, _ in selected], VALIDATION_SUPPORT,
                )
                if logger.isEnabledFor(logging.INFO):
                    for (passage, _), score in zip(selected, support_scores):
                        logger.info("%s lab_page=%s score=%.4f hypothesis=%r premise=%r",
                                    criterion.id, passage.page, score, lab_claim, passage.text)
            elif criterion.evidence_profile in ("proof", "demonstration"):
                support_scores = self._profile_support(criterion.evidence_profile, selected, target)
            else:
                support_scores = [scores[0] for scores in predictions]
            if criterion.evidence_profile in ("demonstration", "operation") and logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s ModernBERT scored=%s", criterion.id, learned_support is not None)
                for i, ((passage, relevance), score) in enumerate(zip(selected, support_scores)):
                    learned = None if learned_support is None else round(learned_support[i], 3)
                    logger.debug("%s candidate %s p%s relevance=%.3f nli=%.3f support=%.3f modernbert=%s: %s",
                                 criterion.id, passage.document_id, passage.page, relevance,
                                 predictions[i][0], score, learned, passage.text)
            acceptance = (VALIDATION_SUPPORT
                          if criterion.laboratory_validation or criterion.evidence_profile == "demonstration"
                          else self.support)
            candidates = [
                (p, relevance, score)
                for (p, relevance), score in zip(selected, support_scores)
                if score >= acceptance
            ]
            # First check single passages. Only unmet validation criteria with
            # enough time can fall back to checking explicitly linked excerpts.
            operational = criterion.evidence_profile == "operation"
            required = list(OPERATIONAL_DEPLOYMENT_CLAIMS) if operational else []
            if needs_completed_test:
                required.append(hypotheses[3])
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
                prospective = needs_completed_test and checks[p.text, hypotheses[3]] >= self.support
                failed_claims = [h for h in required if h != hypotheses[3]
                                 and checks[p.text, h] < self.triple_threshold]
                if prospective or failed_claims:
                    if operational or criterion.evidence_profile == "demonstration":
                        logger.debug("%s rejected %s p%s prospective=%s unmet=%s: %s",
                                     criterion.id, p.document_id, p.page, prospective, failed_claims, p.text)
                    continue
                qualifying.append((p, entailment * relevance))

            supporting_passages = tuple(p for p, _ in sorted(qualifying, key=lambda item: -item[1])[:3])
            if (not supporting_passages and needs_triple and target.technology.strip()
                    and remaining_ms() > reserve_ms + REVIEW_MARGIN_MS):
                supporting_passages = self._linked_support(passages, selected, criterion, target, hypotheses)

            negatives = [p for (p, _), scores in zip(selected, predictions)
                         if scores[1] >= self.contradiction]
            negative = bool(negatives)
            if negative and target.technology.strip() and remaining_ms() > reserve_ms + REVIEW_MARGIN_MS:
                negative = self._has_scoped_contradiction(
                    negatives, criterion, target, supporting_passages, test_specific=needs_triple,
                )
            state = ("conflict" if supporting_passages and negative else "contradicted" if negative else
                     "met" if supporting_passages else "insufficient evidence")
            citations = [{"document_id": p.document_id, "page": p.page, "supporting_span": p.text}
                         for p in supporting_passages] if state == "met" else []
            if state == "insufficient evidence" and remaining_ms() > reserve_ms + REVIEW_MARGIN_MS:
                if criterion.evidence_profile == "concept" and target.technology.strip():
                    citations = list(self._focused_support(source_pages, CONCEPT_HEADING, (
                        f"This passage describes {target.technology}.",
                        "The passage describes the intended use of the technology.",
                        "The passage explains what the technology does.",
                    )))[:3]
                elif criterion.evidence_profile == "proof":
                    scope = profile_scope
                    if not scope and target.technology.strip():
                        scope = {entry["document_id"] for entry in self._focused_support(
                            source_pages, CONCEPT_HEADING,
                            (f"This passage describes {target.technology}.",),
                        )}
                    if scope and remaining_ms() > reserve_ms + 8000:
                        citations = list(self._focused_support(
                            (p for p in source_pages if p.document_id in scope),
                            COMPARISON_METRIC, (
                                "Simulated output was compared against measured observations.",
                                "The report gives a quantitative result of validation of a model.",
                            ),
                        ))[:3]
                if citations:
                    state = "met"
            if logger.isEnabledFor(logging.INFO):
                logger.info("%s state=%s strongest_support=%.3f qualifying=%d", criterion.id, state,
                            max(support_scores, default=0.0), len(qualifying))
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
