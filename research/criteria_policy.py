"""Explicit criterion-to-TRL mapping and quality policies.

Do not infer stage from incidental 'TRL n' in descriptive text when an
official criterion ID exists. Quality criteria C10-C12 are NEVER TRL levels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import re
from .engine import Criterion
from .production import Criterion as EvidenceCriterion

QUALITY_IDS = frozenset(("C10", "C11", "C12"))
APCTT_CHECKLIST_V1 = "APCTT-2026-CHECKLIST-v1"


def is_official_checklist(checklist: Mapping) -> bool:
    # The APCTT 0.5.0 worker strips the official flag from its checklist payload.
    return checklist.get("version") == APCTT_CHECKLIST_V1


@dataclass(frozen=True)
class DecisionPolicy:
    """TRL quality gate thresholds must be verified against the official rubric.

    A quality failure invalidates the affected evidence; it is NOT necessarily
    a blanket failure of every lower TRL. The optional level caps must be
    explicitly configured, not guessed.
    """
    quality_caps: Mapping[str, int] | None = None

    def cap(self, level: int, states: Mapping[str, str]) -> tuple[int, list[str]]:
        blocked = []
        if not self.quality_caps:
            return level, blocked
        for cid, max_level in self.quality_caps.items():
            if cid not in QUALITY_IDS:
                raise ValueError(f"Not a quality criterion: {cid}")
            if states.get(cid) != "met" and level > int(max_level):
                level = int(max_level)
                blocked.append(cid)
        return level, blocked


def parse_criteria(checklist: Mapping, explicit_levels: Mapping[str, int] | None = None) -> list[Criterion]:
    explicit_levels = explicit_levels or {}
    items = checklist["criteria"]
    result = []
    for item in items:
        cid = str(item["criterion_id"])
        canonical = re.search(r"(?:^|-)C(0?[1-9]|1[0-2])$", cid, re.I)
        short_id = f"C{int(canonical.group(1)):02d}" if canonical else cid
        desc = str(item.get("description", "")).strip()
        if not desc:
            raise ValueError(f"Missing description for {cid}")
        if short_id in QUALITY_IDS:
            level = 0
        elif cid in explicit_levels:
            level = int(explicit_levels[cid])
        elif canonical and 1 <= int(canonical.group(1)) <= 9:
            # Only use this mapping when official C01-C09 IDs are present.
            level = int(canonical.group(1))
        else:
            raise ValueError(f"Unknown criterion-to-TRL mapping: {cid}")
        if level < 0 or level > 9:
            raise ValueError(f"Invalid TRL for {cid}: {level}")
        if is_official_checklist(checklist) and ": " in desc:
            desc = desc.split(": ", 1)[1]
            desc = re.sub(r"^the report states that ", "", desc, flags=re.I)
            desc = desc[0].upper() + desc[1:]
        result.append(Criterion(cid, level, desc, required=cid not in QUALITY_IDS))
    if len({c.id for c in result}) != len(result):
        raise ValueError("Duplicate checklist criterion IDs")
    return result



def assessment_criteria(checklist: Mapping, explicit_levels: Mapping[str, int] | None = None) -> list[EvidenceCriterion]:
    profiles = {
        (3, "proof of concept (trl 3)"): "proof",
        (4, "laboratory validation (trl 4)"): "lab",
        (6, "prototype in relevant environment (trl 6)"): "demonstration",
    }
    descriptions = {str(item["criterion_id"]): item["description"] for item in checklist["criteria"]}
    result = []
    for criterion in parse_criteria(checklist, explicit_levels):
        heading = descriptions[criterion.id].partition(":")[0].strip().casefold()
        profile = profiles.get((criterion.trl, heading), "") if is_official_checklist(checklist) else ""
        result.append(EvidenceCriterion(
            criterion.id, criterion.hypothesis, criterion.trl if criterion.required else None,
            laboratory_validation=profile == "lab", evidence_profile=profile if profile != "lab" else "",
        ))
    return result
