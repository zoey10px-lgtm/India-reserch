"""Resolve only organizer-declared catalogue models from local model store.

The official kit's contracts/model-catalog.json is the source of truth.
Default checkpoints are pinned to the organizer’s published catalogue.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

# Pinned entries from APCTT kit 0.5.0, contracts/model-catalog.json.
DEFAULT_MODELS = {
    "encoder": ("all-minilm-l6-v2", "sentence-transformers/all-MiniLM-L6-v2",
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "reranker": ("ms-marco-minilm-l6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2",
                 "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    "nli": ("deberta-v3-base-zeroshot-v2.0-c", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c",
            "bddf8c5411c34ac3565e16e04384fd68b2618dda"),
}


def default_model_paths(store):
    return {role: Path(store) / repo / rev for role, (_, repo, rev) in DEFAULT_MODELS.items()}


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
        repo = entry.get("repo")
        if not repo and org and name:
            repo = f"{org}/{name}"
        if not (repo and revision):
            raise ValueError(f"Cannot derive store path for {catalog_id}")
        rel = f"{repo}/{revision}"
    relative = Path(str(rel))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Catalogue store path must be relative and safe")
    root = Path(store or os.environ["APCTT_MODEL_STORE"]).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise FileNotFoundError(f"Approved model not installed: {catalog_id} ({resolved})")
    return resolved
