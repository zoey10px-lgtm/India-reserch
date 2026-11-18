"""Resolve only organizer-declared catalogue models from local model store.

The official kit's contracts/model-catalog.json is the source of truth.
No hardcoded public checkpoints, network calls or unapproved downloads.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

def resolve_catalog_model(catalog_id: str, catalog_path: str | Path,
                          store: str | Path | None = None) -> Path:
    if not catalog_id or not isinstance(catalog_id, str):
        raise ValueError("catalog_id required")
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    entries = catalog if isinstance(catalog, list) else next(
        (catalog[k] for k in ("models", "entries", "catalog") if isinstance(catalog.get(k), list)), None)
    if entries is None:
        raise ValueError("Unsupported catalogue structure: inspect official JSON")
    entry = next((x for x in entries if x.get("catalog_id", x.get("id")) == catalog_id), None)
    if entry is None:
        raise ValueError(f"Not in official model catalogue: {catalog_id}")
    # Prefer explicit organizer-provided path; otherwise require org/name/revision.
    rel = entry.get("store_path") or entry.get("relative_path")
    if rel is None:
        org = entry.get("org") or entry.get("organization")
        name = entry.get("name")
        revision = entry.get("revision")
        if not (org and name and revision):
            raise ValueError(f"Cannot derive store path for {catalog_id}")
        rel = f"{org}/{name}/{revision}"
    relative = Path(str(rel))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Catalogue store path must be relative and safe")
    root = Path(store or os.environ["APCTT_MODEL_STORE"]).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise FileNotFoundError(f"Approved model not installed: {catalog_id} ({resolved})")
    return resolved
