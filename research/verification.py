"""Evidence quality, provenance and TRL gates.

No unsupported 'confidence' is invented. Every supporting span must occur in
the extracted source page. C10–C12 are quality criteria, not TRL levels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math
import re


@dataclass(frozen=True)
class TargetContext:
    technology: str
    application: str
    environment: str

    @classmethod
    def from_case(cls, target: Mapping) -> "TargetContext":
        return cls(
            str(target.get("assessment_target") or target.get("technology") or target.get("name") or ""),
            str(target.get("intended_application") or ""),
            str(target.get("intended_environment") or ""),
        )

    def query(self, criterion: str) -> str:
        return " | ".join(x for x in (
            criterion, f"technology: {self.technology}",
            f"intended application: {self.application}",
            f"intended environment: {self.environment}",
        ) if x.split(": ")[-1].strip())


@dataclass(frozen=True)
class Triple:
    artifact: bool
    environment: bool
    outcome: bool
    prospective: bool

    @property
    def verified(self) -> bool:
        return self.artifact and self.environment and self.outcome and not self.prospective


def verify_triple(text: str, nli) -> Triple:
    """NLI-based action/environment/outcome gate, not keyword assertions.

    Hypotheses describe completed observed facts. Thresholds are provisional
    and MUST be calibrated against held-out annotated passages.
    """
    hypotheses = [
        "The described target artifact or prototype was implemented and tested.",
        "The test took place in an identified environment with stated conditions.",
        "The report presents an observed result or measured outcome from that test.",
        "The passage describes only a future, proposed, or uncompleted test.",
    ]
    if hasattr(nli, "predict_batch"):
        scores = nli.predict_batch([(text, h) for h in hypotheses])
    else:
        scores = [nli.predict(text, h) for h in hypotheses]
    return Triple(
        artifact=scores[0].entailment >= 0.65,
        environment=scores[1].entailment >= 0.65,
        outcome=scores[2].entailment >= 0.65,
        prospective=scores[3].entailment >= 0.65,
    )


def exact_span(document_id: str, page: int, span: str,
               page_texts: Mapping[tuple[str, int], str]) -> bool:
    """Strict substring validation. Do not paraphrase cited PDF text."""
    original = page_texts.get((document_id, page), "")
    return bool(span and original and span in original)


def grounded_evidence(records: list[dict], page_texts: Mapping[tuple[str, int], str]) -> list[dict]:
    return [r for r in records if exact_span(
        str(r["document_id"]), int(r["page"]), str(r["supporting_span"]), page_texts
    )]


def quality_gate(statuses: Mapping[str, str], required: tuple[str, ...] = ("C10", "C11", "C12")) -> dict:
    """Conservative policy: if a required quality gate is unverified, abstain.

    Exact TRL-specific cap policies must be confirmed against official rubric.
    """
    missing = [c for c in required if statuses.get(c) != "met"]
    return {"pass": not missing, "missing": missing}


def monotonic_gate(criteria: list[dict], level: int, required_ids: set[str]) -> dict:
    """Only explicit prerequisite IDs can constrain progression.

    Never infer TRL from C01..C12 numbering alone.
    """
    statuses = {r["id"]: r.get("state", r.get("status")) for r in criteria}
    missing = sorted(cid for cid in required_ids if statuses.get(cid) != "met")
    return {"pass": not missing, "missing": missing, "level": level}


def confidence_features(records: list[dict], quality: Mapping[str, str]) -> dict:
    """Transparent raw signals, NOT calibrated probability."""
    if not records:
        return {"retrieval": 0.0, "nli_margin": 0.0, "quality_coverage": 0.0,
                "conflict": 0.0, "calibrated_confidence": None}
    retrieval = sum(max(0.0, min(1.0, float(r.get("retrieval", 0)))) for r in records)/len(records)
    margin = sum(max(0.0, float(r.get("entailment", 0)) - float(r.get("contradiction", 0))) for r in records)/len(records)
    coverage = sum(quality.get(k) == "met" for k in ("C10", "C11", "C12"))/3
    conflict = float(any(r.get("contradiction", 0) >= 0.7 for r in records))
    return {"retrieval": retrieval, "nli_margin": margin,
            "quality_coverage": coverage, "conflict": conflict,
            "calibrated_confidence": None}
