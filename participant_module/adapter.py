"""APCTT AssessmentModule integration (offline, evidence-first).

Requires organizer-approved local NLI model. If unavailable, abstains instead
of pretending keyword matches are model judgments.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import re
import sys
from typing import Any

from .research.engine import Criterion
from .research.offline_nli import OfflineNLI
from .research.pdf_parser import read_pdf, extract_windows
from .research.pipeline import EvidenceReasoner


class AssessmentModule:
    def __init__(self):
        self.checklist = {}
        self.criteria = []
        self.reasoner = None

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        self.checklist = config["checklist"]
        self.criteria = []
        for item in self.checklist["criteria"]:
            desc = str(item.get("description", ""))
            match = re.search(r"\bTRL\s*([1-9])\b", desc, re.I)
            if not match:
                # Never invent the level for a criterion with ambiguous metadata.
                continue
            self.criteria.append(Criterion(
                id=str(item["criterion_id"]), trl=int(match.group(1)),
                hypothesis=re.sub(r"\(?TRL\s*[1-9]\)?", "", desc, flags=re.I).strip(),
            ))
        settings = config.get("module", {}).get("settings", {}) or {}
        model_rel = settings.get("nli_model_path") or os.environ.get("APCTT_NLI_MODEL_PATH")
        if not model_rel:
            self.reasoner = None
            return
        store = Path(os.environ.get("APCTT_MODEL_STORE", "/nonexistent"))
        model_path = Path(model_rel)
        if not model_path.is_absolute():
            model_path = store / model_path
        try:
            device = str(config.get("compute", {}).get("device", "cpu"))
            nli = OfflineNLI(str(model_path), device=device)
            self.reasoner = EvidenceReasoner(nli=nli)
        except Exception as exc:
            print(f"NLI initialization failed: {exc!r}", file=sys.stderr)
            self.reasoner = None

    def _base(self, case):
        return {
            "schema_version": case["schema_version"],
            "checklist_version": self.checklist["version"],
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
        }

    def _abstain(self, case, message):
        result = self._base(case)
        ids = [c["criterion_id"] for c in self.checklist["criteria"]]
        result.update({
            "assessment_status": "abstained", "predicted_trl": None,
            "criterion_status": {cid: "insufficient evidence" for cid in ids},
            "evidence_reference": {cid: [] for cid in ids},
            "missing_evidence": {cid: message for cid in ids},
            "abstention_reason": "insufficient evidence",
            "error_code": None,
        })
        return result

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        if self.reasoner is None:
            return self._abstain(case, "Approved offline NLI model not configured.")
        if len(self.criteria) != len(self.checklist["criteria"]):
            return self._abstain(case, "Checklist level mapping incomplete.")
        try:
            pages = []
            for doc in case["documents"]:
                if context.remaining_ms() < 25000:
                    break
                pages.extend(read_pdf(
                    str(doc["path"]), str(doc["document_id"]),
                    remaining_ms=context.remaining_ms,
                ))
            evidence = extract_windows(pages)
            target_info = case.get("assessment_target", {})
            target = " ".join(str(v) for v in target_info.values() if isinstance(v, (str, int)))
            result = self.reasoner.assess(evidence, self.criteria, target)
            if result["status"] == "abstained":
                return self._abstain(case, "Required evidence is missing or contradictory.")
            response = self._base(case)
            states = {r["id"]: r for r in result["criteria"]}
            response.update({
                "assessment_status": "completed",
                "predicted_trl": result["predicted_trl"],
                "criterion_status": {
                    c.id: "met" if states[c.id]["state"] == "met" else "insufficient evidence"
                    for c in self.criteria
                },
                "evidence_reference": {c.id: states[c.id]["evidence"] for c in self.criteria},
                "missing_evidence": {
                    c.id: "Evidence not established or conflicts with other passages."
                    for c in self.criteria if states[c.id]["state"] != "met"
                },
                "abstention_reason": None, "error_code": None,
                "trl_rationale": "Highest consecutively supported TRL based on NLI-verified evidence.",
            })
            return response
        except Exception as exc:
            print(f"assessment error: {exc!r}", file=sys.stderr)
            return self._abstain(case, "The assessment could not be completed.")

    def reset_case(self) -> None:
        pass

    def close(self) -> None:
        self.reasoner = None
