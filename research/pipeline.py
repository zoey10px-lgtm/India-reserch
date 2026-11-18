"""Production-oriented evidence-first pipeline.

All judgments use the supplied NLI model. No keyword-only substitute.
Retrieval supports a real embedding encoder, with sparse fallback ONLY for
candidate generation. The sparse fallback cannot declare a criterion met.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .engine import Criterion, Evidence, Inference, tokens
from .graph import build_edges


@dataclass
class EngineSettings:
    candidate_limit: int = 24
    top_evidence: int = 3
    min_target_score: float = 0.35
    min_entailment: float = 0.72
    min_contradiction: float = 0.72
    min_support_weight: float = 0.22
    min_retrieval_score: float = 0.10


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b)) / (na*nb) if na and nb else 0.0


class EvidenceReasoner:
    def __init__(self, nli, encoder=None, settings: EngineSettings | None = None):
        if nli is None:
            raise ValueError("Real NLI is required; refusing to fabricate assessments")
        self.nli = nli
        self.encoder = encoder
        self.settings = settings or EngineSettings()

    def _retrieve(self, evidence: list[Evidence], criterion: Criterion) -> list[tuple[float, Evidence]]:
        query = criterion.hypothesis
        query_tokens = tokens(query)
        dense = None
        if self.encoder is not None:
            vectors = self.encoder.encode(
                [query] + [e.text for e in evidence],
                normalize_embeddings=True, show_progress_bar=False
            )
            dense = [max(0.0, float(_cosine(vectors[0], v))) for v in vectors[1:]]
        scored = []
        for i, e in enumerate(evidence):
            sparse = len(tokens(e.text) & query_tokens) / max(1, len(query_tokens))
            score = 0.35*sparse + 0.65*dense[i] if dense is not None else sparse
            if score >= self.settings.min_retrieval_score:
                scored.append((score, i, e))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(score, e) for score, _, e in scored[:self.settings.candidate_limit]]

    @staticmethod
    def _target_score(e: Evidence, target: str) -> float:
        if e.target and target and _norm(e.target) != _norm(target):
            return 0.0
        if e.target and target:
            return 1.0
        if not target:
            return 0.45
        target_words = tokens(target)
        overlap = len(target_words & tokens(e.text)) / max(1, len(target_words))
        return 0.35 + 0.55*overlap

    def _criterion(self, criterion: Criterion, evidence: list[Evidence], target: str) -> dict[str, Any]:
        candidates = self._retrieve(evidence, criterion)
        dedup = set()
        kept = []
        for retrieval, e in candidates:
            signature = _norm(e.text)
            if signature in dedup:
                continue
            dedup.add(signature)
            target_score = self._target_score(e, target)
            if target_score < self.settings.min_target_score:
                continue
            kept.append((retrieval, target_score, e))
        pairs = [(e.text, criterion.hypothesis) for _, _, e in kept]
        predictions = self.nli.predict_batch(pairs) if hasattr(self.nli, "predict_batch") else [
            self.nli.predict(p, h) for p, h in pairs
        ]
        records = []
        for (retrieval, target_score, e), p in zip(kept, predictions):
            # Score represents relative evidence utility, not a calibrated probability.
            weight = max(0.0, retrieval) * target_score
            records.append({
                "document_id": e.source_id, "page": e.page,
                "supporting_span": e.text, "text": e.text,
                "retrieval": round(retrieval, 4),
                "target_match": round(target_score, 4),
                "weight": round(weight, 4),
                "entailment": p.entailment,
                "contradiction": p.contradiction,
                "neutral": p.neutral,
            })
        graph = build_edges(records)
        supporters = [r for r in records if r["entailment"] >= self.settings.min_entailment
                      and r["weight"] >= self.settings.min_support_weight]
        opposers = [r for r in records if r["contradiction"] >= self.settings.min_contradiction
                    and r["weight"] >= self.settings.min_support_weight]
        if supporters and opposers:
            state = "conflict"
        elif supporters:
            state = "met"
        elif opposers:
            state = "contradicted"
        else:
            state = "insufficient evidence"
        # No numeric confidence until validated calibration data is available.
        ranked = sorted(records, key=lambda r: -r["weight"] * max(r["entailment"], r["contradiction"]))
        selected = ranked[:self.settings.top_evidence]
        return {
            "id": criterion.id, "trl": criterion.trl, "state": state,
            "evidence": [{"document_id": r["document_id"], "page": r["page"],
                          "supporting_span": r["supporting_span"]} for r in selected if state == "met"],
            "debug": {"candidates": len(records), "supports": len(supporters),
                      "contradictions": len(opposers),
                      "edges": [vars(edge) for edge in graph],
                      "records": records},
        }

    def assess(self, evidence: list[Evidence], criteria: list[Criterion], target: str) -> dict[str, Any]:
        if not evidence or not criteria:
            return {"status": "abstained", "predicted_trl": None, "criteria": []}
        results = [self._criterion(c, evidence, target) for c in criteria]
        # TRL levels are cumulative: a higher level cannot be claimed when
        # a required lower-level criterion is missing or contradictory.
        by_level = sorted({c.trl for c in criteria})
        chosen = None
        for level in by_level:
            group = [(c, r) for c, r in zip(criteria, results) if c.trl == level and c.required]
            if not group or not all(r["state"] == "met" for _, r in group):
                break
            chosen = level
        return {
            "status": "completed" if chosen is not None else "abstained",
            "predicted_trl": chosen,
            "criteria": results,
            "confidence": None,
            "notes": "Uncalibrated NLI probabilities are not assessment confidence",
        }
