"""Official-facing adapter for the evidence-grounded engine.

Model directories must be supplied by the official catalogue/runtime. Never
silently substitute lexical inference when a model is missing. This adapter
is schema-conservative: validate against the official kit before submission.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import os
import sys

from .research.production import EvidenceEngine, Criterion, Passage, Target
from .research.offline_nli import OfflineNLI
from .research.two_stage_retrieval import TwoStageRetriever
from .research.pdf_parser import read_pdf, extract_windows
from .research.criteria_policy import parse_criteria


class _NLIAdapter:
    def __init__(self, model):
        self.model = model

    def predict_batch(self, pairs):
        predictions = self.model.predict_batch(pairs)
        return [(p.entailment, p.contradiction, p.neutral) for p in predictions]


class AssessmentModule:
    def __init__(self):
        self.checklist = {}
        self.criteria = []
        self.engine = None
        self.settings = {}

    def load(self, config: Mapping, asset_root: Path) -> None:
        self.checklist = config["checklist"]
        self.settings = config.get("module", {}).get("settings", {}) or {}
        parsed = parse_criteria(self.checklist, self.settings.get("criterion_levels"))
        self.criteria = [Criterion(c.id, c.hypothesis, c.trl if c.required else None) for c in parsed]
        store = Path(os.environ.get("APCTT_MODEL_STORE", "/nonexistent"))
        names = [self.settings.get("nli_model_path"),
                 self.settings.get("encoder_model_path", "all-minilm-l6-v2"),
                 self.settings.get("reranker_model_path")]
        if not all(names):
            self.engine = None
            return
        paths = [Path(x) if Path(x).is_absolute() else store / x for x in names]
        device = config.get("compute", {}).get("device", "cpu")
        try:
            nli = OfflineNLI(str(paths[0]), device=device)
            retrieval = TwoStageRetriever(str(paths[1]), str(paths[2]), device=device)
            self.engine = EvidenceEngine(_NLIAdapter(nli), retrieval)
        except Exception as exc:
            print(f"offline model initialization failed: {exc!r}", file=sys.stderr)
            self.engine = None

    def _base(self, case):
        return {
            "schema_version": case["schema_version"],
            "checklist_version": self.checklist["version"],
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
        }

    def _abstain(self, case, reason):
        ids = [c.id for c in self.criteria]
        return {**self._base(case),
            "assessment_status": "abstained",
            "predicted_trl": None,
            "criterion_status": {cid: "insufficient evidence" for cid in ids},
            "evidence_reference": {cid: [] for cid in ids},
            "missing_evidence": {cid: reason for cid in ids},
            "abstention_reason": "insufficient evidence",
            "error_code": None,
        }

    def assess(self, case: Mapping, context) -> dict:
        if self.engine is None:
            return self._abstain(case, "Catalogue-approved offline models are not configured")
        try:
            pages = []
            for doc in case["documents"]:
                if context.remaining_ms() < 30000:
                    return self._abstain(case, "Time budget too low to read all documents")
                pages.extend(read_pdf(
                    str(doc["path"]), str(doc["document_id"]),
                    remaining_ms=context.remaining_ms, enable_ocr=True,
                ))
            passages = []
            for e in extract_windows(pages):
                passages.append(Passage(e.source_id, e.page, e.text, section=getattr(e, "section", "")))
            info = case["assessment_target"]
            target = Target(
                str(info.get("description") or info.get("target_id") or ""),
                str(case.get("intended_application") or ""),
                str(case.get("intended_environment") or ""),
            )
            result = self.engine.evaluate(
                passages, self.criteria, target,
                remaining_ms=context.remaining_ms,
                quality_caps=self.settings.get("quality_caps"),
            )
            if result["status"] == "abstained":
                return self._abstain(case, result["abstention_reason"])
            states = {r["id"]: r for r in result["criteria"]}
            return {**self._base(case),
                "assessment_status": "completed",
                "predicted_trl": result["predicted_trl"],
                "criterion_status": {c.id: (
                    "met" if states[c.id]["status"] == "met" else "insufficient evidence"
                ) for c in self.criteria},
                "evidence_reference": {c.id: states[c.id]["evidence"] for c in self.criteria},
                "missing_evidence": {c.id: "Insufficient or conflicting grounded evidence"
                    for c in self.criteria if states[c.id]["status"] != "met"},
                "abstention_reason": None, "error_code": None,
                "trl_rationale": "Evidence-grounded cumulative TRL decision",
            }
        except Exception as exc:
            print(f"assessment failed: {exc!r}", file=sys.stderr)
            return self._abstain(case, "Evaluation failed or exceeded available resources")

    def reset_case(self):
        pass

    def close(self):
        self.engine = None
