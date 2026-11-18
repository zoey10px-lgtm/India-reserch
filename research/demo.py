"""Assess PDF evidence using three installed neural checkpoints."""
from pathlib import Path
from time import monotonic
import argparse
import json
import os
import sys

from .criteria_policy import assessment_criteria, is_official_checklist
from .production import Target
from .runtime import assess_pdfs, load_engine
from .catalog import DEFAULT_MODELS, default_model_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="+")
    parser.add_argument("--target", required=True)
    parser.add_argument("--application", default="")
    parser.add_argument("--environment", default="")
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--model-store", type=Path, default=Path(os.environ.get("APCTT_MODEL_STORE", ".models")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--budget-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.threads < 1 or args.budget_seconds < 1:
        parser.error("threads and budget-seconds must be positive")
    started = monotonic()
    deadline = started + args.budget_seconds

    def remaining_ms():
        return max(0, int((deadline - monotonic()) * 1000))

    try:
        checklist = json.loads(args.checklist.read_text(encoding="utf-8"))
        criteria = assessment_criteria(checklist, checklist.get("criterion_levels"))
        for pdf in args.pdf:
            if not pdf.is_file():
                raise FileNotFoundError(pdf)
        import torch

        torch.set_num_threads(args.threads)
        engine = load_engine(
            default_model_paths(args.model_store), device=args.device,
        )
        result = assess_pdfs(
            engine, [(f"doc-{index}", pdf) for index, pdf in enumerate(args.pdf, 1)], criteria,
            Target(args.target, args.application, args.environment), remaining_ms=remaining_ms,
            quality_caps=checklist.get("quality_caps"),
            require_consecutive=not is_official_checklist(checklist),
        )
    except Exception as exc:
        print(f"PDF assessment failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    result["models"] = {
        role: {"id": catalog_id, "repo": name, "revision": revision}
        for role, (catalog_id, name, revision) in DEFAULT_MODELS.items()
    }
    result["documents"] = {f"doc-{i}": str(pdf) for i, pdf in enumerate(args.pdf, 1)}
    result["device"] = args.device
    result["elapsed_seconds"] = round(monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
