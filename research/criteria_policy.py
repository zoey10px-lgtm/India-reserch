"""Explicit criterion-to-TRL mapping and quality policies.

Do not infer stage from incidental 'TRL n' in descriptive text when an
official criterion ID exists. Quality criteria C10-C12 are NEVER TRL levels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from .engine import Criterion

QUALITY_IDS = frozenset(("C10", "C11", "C12"))

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
        desc = str(item.get("description", "")).strip()
        if not desc:
            raise ValueError(f"Missing description for {cid}")
        if cid in QUALITY_IDS:
            level = 0
        elif cid in explicit_levels:
            level = int(explicit_levels[cid])
        elif cid.startswith("C") and cid[1:].isdigit() and 1 <= int(cid[1:]) <= 9:
            # Only use this mapping when official C01-C09 IDs are present.
            level = int(cid[1:])
        else:
            raise ValueError(f"Unknown criterion-to-TRL mapping: {cid}")
        if level < 0 or level > 9:
            raise ValueError(f"Invalid TRL for {cid}: {level}")
        result.append(Criterion(cid, level, desc, required=cid not in QUALITY_IDS))
    if len({c.id for c in result}) != len(result):
        raise ValueError("Duplicate checklist criterion IDs")
    return result
