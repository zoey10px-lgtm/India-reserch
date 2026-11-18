"""Official-facing adapter for the evidence-grounded engine.

The kit provides the local model store, the checklist and the case deadline.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import os

# The staging tool vendors research inside the participant package.
if __package__ and (Path(__file__).parent / "research").is_dir():
    from .research.catalog import DEFAULT_MODELS, default_model_paths, resolve_catalog_model
    from .research.criteria_policy import parse_criteria
    from .research.production import Criterion, Target
    from .research.runtime import assess_pdfs, load_engine
else:
    from research.catalog import DEFAULT_MODELS, default_model_paths, resolve_catalog_model
    from research.criteria_policy import parse_criteria
    from research.production import Criterion, Target
    from research.runtime import assess_pdfs, load_engine


class AssessmentModule:
    def __init__(self):
        self.checklist = {}
        self.criteria = []
        self.engine = None
        self.settings = {}

    def load(self, config: Mapping, asset_root: Path) -> None:
        self.engine = None
        self.checklist = config["checklist"]
        self.settings = config.get("module", {}).get("settings", {}) or {}
        parsed = parse_criteria(self.checklist, self.settings.get("criterion_levels"))
        self.criteria = [Criterion(c.id, c.hypothesis, c.trl if c.required else None) for c in parsed]
        ids = {role: self.settings.get(f"{role}_catalog_id", model[0])
               for role, model in DEFAULT_MODELS.items()}
        catalog = self.settings.get("catalog_path") or os.environ.get("APCTT_MODEL_CATALOG")
        store = os.environ.get("APCTT_MODEL_STORE")
        if not store:
            raise ValueError("Set APCTT_MODEL_STORE to the organizer model directory")
        if catalog:
            paths = {role: resolve_catalog_model(cid, catalog, store) for role, cid in ids.items()}
        else:
            if any(cid != DEFAULT_MODELS[role][0] for role, cid in ids.items()):
                raise ValueError("Configure APCTT_MODEL_CATALOG when overriding a model ID")
            paths = default_model_paths(store)
        device = config.get("compute", {}).get("device", "cpu")
        if device == "cpu":
            import torch

            torch.set_num_threads(self.settings.get("cpu_threads", min(4, os.cpu_count() or 1)))
        self.engine = load_engine(
            paths, device=device, encoder_query_prefix=self.settings.get("encoder_query_prefix", ""),
        )

    def _base(self, case):
        return {
            "schema_version": case["schema_version"],
            "checklist_version": self.checklist["version"],
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
        }

    def assess(self, case: Mapping, context) -> dict:
        if self.engine is None:
            raise RuntimeError("AssessmentModule.load() must load all three models before assessment")
        info = case["assessment_target"]
        target = Target(
            str(info.get("description") or info.get("target_id") or ""),
            str(case.get("intended_application") or ""),
            str(case.get("intended_environment") or ""),
        )
        result = assess_pdfs(
            self.engine,
            [(doc["document_id"], doc["path"]) for doc in case["documents"]],
            self.criteria, target, remaining_ms=context.remaining_ms,
            quality_caps=self.settings.get("quality_caps"),
        )
        states = {row["id"]: row for row in result["criteria"]}
        status = result["status"]
        if result.get("budget_exhausted"):
            return {
                **self._base(case), "assessment_status": "failed", "predicted_trl": None,
                "criterion_status": {}, "evidence_reference": {}, "missing_evidence": {},
                "abstention_reason": None, "error_code": "ASSESSMENT_TIME_BUDGET",
            }
        abstention = None
        if status == "abstained":
            abstention = "conflicting evidence" if any(
                row["status"] == "conflict" for row in states.values()
            ) else "insufficient evidence"
        output = {
            **self._base(case),
            "assessment_status": status,
            "predicted_trl": result["predicted_trl"],
            "abstention_reason": abstention,
            "error_code": None,
        }
        if result["status"] == "completed":
            output["trl_rationale"] = "Evidence-grounded cumulative TRL decision"
        contract_states = {"met": "met", "contradicted": "not met"}
        output["criterion_status"] = {
            cid: contract_states.get(row["status"], "insufficient evidence")
            for cid, row in states.items()
        }
        output["evidence_reference"] = {cid: row["evidence"] for cid, row in states.items()}
        output["missing_evidence"] = {
            cid: result["abstention_reason"] or "Insufficient or conflicting grounded evidence"
            for cid, state in output["criterion_status"].items() if state == "insufficient evidence"
        }
        return output

    def reset_case(self):
        pass

    def close(self):
        self.engine = None
