"""Run a CPU PDF inference check, optionally against the APCTT kit contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace
from urllib.request import urlopen
import venv

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / ".pc-validation"
VENV = ROOT / ".venv-pc"
SAMPLE_URL = "https://www.sandia.gov/app/uploads/sites/273/dlm_uploads/RM3-TANK-TEST-NREL-TP-5000-62951.pdf"
SAMPLE_HASH = "99fdab9c06990ed2db1b7f92ade9d140f12a756b79439be78943027d11d14242"
SAMPLE_TARGET = "RM3 wave energy converter"
SAMPLE_APPLICATION = "Electric power generation from ocean waves"
SAMPLE_ENVIRONMENT = "Ocean wave conditions"
KIT_ASSETS = (
    "contracts/checklist.official.json", "contracts/model-catalog.json",
    "src/apctt_contracts/results.py", "config/dev.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Use your PDF instead of the public RM3 report")
    parser.add_argument("--target")
    parser.add_argument("--application")
    parser.add_argument("--environment")
    parser.add_argument("--checklist", type=Path, help="Research checklist JSON for a custom PDF")
    parser.add_argument("--kit", type=Path, help="Extracted APCTT participant kit 0.5.0 directory")
    parser.add_argument("--model-store", type=Path, default=Path(os.environ.get("APCTT_MODEL_STORE", ROOT / ".models")))
    parser.add_argument("--budget-seconds", type=int, default=180)
    parser.add_argument("--offline", action="store_true", help="Use the packages, models and PDF already installed")
    args = parser.parse_args()
    if args.budget_seconds <= 30:
        parser.error("--budget-seconds must be greater than 30")
    if args.pdf and (not args.target or not (args.checklist or args.kit)):
        parser.error("--pdf requires --target and either --checklist or --kit")
    if args.pdf and args.kit and not (args.application and args.environment):
        parser.error("--kit with a custom PDF requires --application and --environment")
    if not args.pdf and any((args.target, args.application, args.environment, args.checklist)):
        parser.error("--target, --application, --environment and --checklist require --pdf")
    for file in (args.pdf, args.checklist):
        if file and not file.is_file():
            parser.error(f"File not found: {file}")
    if args.kit and any(not (args.kit / file).is_file() for file in KIT_ASSETS):
        parser.error(f"Point --kit to the extracted kit directory containing contracts/, src/ and config/: {args.kit}")
    return args


def launch_venv(offline):
    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.prefix).resolve() == VENV.resolve():
        return None
    if not python.is_file():
        if offline:
            raise RuntimeError("PC environment is missing. Run once with internet access.")
        print("Creating a Python 3.12 environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


def install_dependencies(kit):
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install"]
    print("Installing/checking CPU inference dependencies...", flush=True)
    subprocess.run([*pip, "--index-url", "https://download.pytorch.org/whl/cpu", "torch>=2.6,<3"], check=True)
    subprocess.run([*pip, "-r", str(ROOT / "requirements-semantic.txt")], check=True)
    if kit:
        subprocess.run([*pip, "pydantic>=2.8,<3"], check=True)


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def sample_pdf(offline):
    path = RESULT_DIR / "rm3.pdf"
    if not path.exists():
        if offline:
            raise RuntimeError(f"Sample PDF is missing: {path}. Run once with internet access.")
        print("Downloading the public RM3 PDF...", flush=True)
        pending = path.with_suffix(".part")
        try:
            with urlopen(SAMPLE_URL, timeout=60) as response, pending.open("wb") as file:
                shutil.copyfileobj(response, file)
            if file_hash(pending) != SAMPLE_HASH:
                raise ValueError("The downloaded RM3 PDF does not match its expected SHA-256")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    if file_hash(path) != SAMPLE_HASH:
        raise ValueError(f"RM3 PDF checksum mismatch. Move {path} and retry.")
    return path


def verify_spans(evidence, pages):
    count = 0
    for items in evidence:
        for item in items:
            page = pages.get((item["document_id"], item["page"]), "")
            span = item.get("supporting_span", "")
            if not span.strip() or span not in page:
                raise ValueError(f"Evidence does not match extracted PDF page {item['page']}")
            count += 1
    return count


def save_report(filename, result):
    path = RESULT_DIR / filename
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  Report: {path}", flush=True)


def run_research(args, pdf, target, application, environment, pages, env):
    checklist = args.checklist or ROOT / "docs/rm3-smoke-checklist.json"
    completed = subprocess.run([
        sys.executable, "-m", "research.demo", str(pdf), "--target", target,
        "--application", application, "--environment", environment,
        "--checklist", str(checklist.resolve()), "--model-store", str(args.model_store.resolve()),
        "--budget-seconds", str(args.budget_seconds),
    ], cwd=ROOT, env=env, stdout=subprocess.PIPE, encoding="utf-8", check=True)
    result = json.loads(completed.stdout)
    save_report("research-result.json", result)
    if result.get("budget_exhausted"):
        raise ValueError("Research evaluation ran out of time. Increase --budget-seconds for a local rerun.")
    spans = verify_spans((item["evidence"] for item in result["criteria"]), pages)
    if not args.pdf and (result["status"] != "completed" or result["predicted_trl"] != 3 or not spans):
        raise ValueError("RM3 research smoke check expected TRL 3 and at least one verified citation")
    print(f"  {result['status']}, TRL={result['predicted_trl']}, {spans} exact citations, "
          f"{result['elapsed_seconds']:.1f}s", flush=True)


def run_official(args, pdf, target, application, environment, pages):
    sys.path.insert(0, str(args.kit.resolve() / "src"))
    from apctt_contracts.config import Checklist
    from apctt_contracts.results import validate_result
    from participant_module.adapter import AssessmentModule

    checklist = json.loads((args.kit / "contracts/checklist.official.json").read_text(encoding="utf-8"))
    schema = json.loads((args.kit / "config/dev.json").read_text(encoding="utf-8"))["schema_version"]
    case = {
        "schema_version": schema, "checklist_version": checklist["version"],
        "case_id": "PC-VERIFY", "input_profile": "pdf_report", "assessment_as_of": None,
        "assessment_target": {"target_id": "PC-TARGET", "type": "technology", "description": target},
        "intended_application": application, "intended_environment": environment,
        "title": pdf.stem,
        "documents": [{
            "document_id": "doc-1", "page_count": len(pages), "path": str(pdf), "title": pdf.stem,
            "document_date": None, "language": "en", "format": "pdf", "sha256": file_hash(pdf),
        }],
    }
    config = {
        "checklist": checklist,
        "compute": {"device": "cpu"},
        "module": {"settings": {"catalog_path": str(args.kit / "contracts/model-catalog.json")}},
    }
    module = AssessmentModule()
    started = monotonic()
    deadline = started + args.budget_seconds
    try:
        module.load(config, ROOT)
        result = module.assess(case, SimpleNamespace(
            remaining_ms=lambda: max(0, int((deadline - monotonic()) * 1000)),
        ))
    finally:
        module.reset_case()
        module.close()
    issues = validate_result(result, case=case, checklist=Checklist.model_validate(checklist))
    elapsed = round(monotonic() - started, 3)
    save_report("official-result.json", {
        "result": result, "contract_issues": [issue.as_dict() for issue in issues],
        "elapsed_seconds": elapsed, "pdf": str(pdf), "checklist": str(args.kit.resolve() / "contracts/checklist.official.json"),
    })
    if issues:
        raise ValueError(f"Official kit contract reported {len(issues)} issue(s)")
    if result["assessment_status"] == "failed":
        raise ValueError(f"Official adapter reported {result['error_code']}")
    spans = verify_spans(result["evidence_reference"].values(), pages)
    print(f"  Contract passed. {result['assessment_status']}, TRL={result['predicted_trl']}, "
          f"{spans} exact citations, {elapsed:.1f}s", flush=True)


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Install 64-bit Python 3.12 and rerun.")
    args = parse_args()
    child_code = launch_venv(args.offline)
    if child_code is not None:
        return child_code
    RESULT_DIR.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from research.setup_models import WEIGHT_HASHES
    from research.catalog import default_model_paths

    os.environ.update(
        APCTT_MODEL_STORE=str(args.model_store.resolve()), TOKENIZERS_PARALLELISM="false",
    )
    if not args.offline:
        install_dependencies(args.kit)
        print("Installing/checking the three APCTT model checkpoints...", flush=True)
        download_env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "120"}
        subprocess.run([sys.executable, "-m", "research.setup_models", "--model-store",
                        str(args.model_store.resolve())], cwd=ROOT, env=download_env, check=True)
    else:
        for role, folder in default_model_paths(args.model_store).items():
            weights = folder / "model.safetensors"
            if not weights.is_file() or file_hash(weights) != WEIGHT_HASHES[role]:
                raise ValueError(f"{role} checkpoint missing or invalid: {weights}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1")
    from research.pdf_parser import read_pdf

    pdf = args.pdf.resolve() if args.pdf else sample_pdf(args.offline)
    target = args.target or SAMPLE_TARGET
    application = args.application or (SAMPLE_APPLICATION if not args.pdf else "")
    environment = args.environment or (SAMPLE_ENVIRONMENT if not args.pdf else "")
    pages = {(p.document_id, p.number): p.text for p in read_pdf(str(pdf), "doc-1")}
    if not pages:
        raise ValueError(f"No PDF pages extracted: {pdf}")
    if not args.pdf or args.checklist:
        print("Running the offline research check...", flush=True)
        # Keep the sample retrieval inputs identical to the documented reproduction.
        run_research(args, pdf, target, application if args.pdf else "",
                     environment if args.pdf else "", pages, os.environ)
    if args.kit:
        print("Running the participant adapter with the official checklist and validator...", flush=True)
        run_official(args, pdf, target, application, environment, pages)
    print("PC verification complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
