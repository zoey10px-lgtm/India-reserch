"""Build and check a CPU Stage 1 archive with the APCTT kit."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from stage_submission import ROOT, stage_research

NREL_RM3 = "https://docs.nlr.gov/docs/fy15osti/62951.pdf"
NREL_HERO = "https://docs.nlr.gov/docs/fy26osti/92489.pdf"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def report_source(report_number, url):
    return {
        "source": f"NREL technical report {report_number}",
        "version": report_number,
        "date": "2026-11-19",
        "role": "Development and offline verification",
        "label_origin": "No expert TRL labels",
        "linked_project": report_number,
        "license": url,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path, help="Extracted APCTT 0.5.0 participant kit")
    parser.add_argument("--team", required=True, help="Team ID registered for the competition")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="Directory for the submission ZIP")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.team):
        parser.error("--team must be a valid APCTT team identifier")
    kit = args.kit.resolve()
    checker = kit / "tools" / "make_submission.py"
    base_path = kit / "contracts" / "base-environment.json"
    if not checker.is_file() or not base_path.is_file():
        parser.error("--kit must point to the extracted directory containing tools/ and contracts/")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{args.team}_submission.zip"
    if archive.exists():
        parser.error(f"{archive} already exists. Choose a different output folder.")

    sys.path.insert(0, str(ROOT))
    from research.catalog import DEFAULT_MODELS

    bases = json.loads(base_path.read_text(encoding="utf-8"))
    models = [{"catalog_id": model[0]} for model in DEFAULT_MODELS.values()]
    metadata = {
        "team": args.team, "package_id": "india-research", "package_version": "0.1.0",
        "declared_package_sha256": None,
    }

    with TemporaryDirectory(prefix="india-apctt-") as temporary:
        source = Path(temporary)
        module = source / "participant_module"
        module.mkdir()
        (source / "assets").mkdir()
        for filename in ("__init__.py", "adapter.py"):
            shutil.copy2(ROOT / "participant_module" / filename, module / filename)
        stage_research(module / "research")
        shutil.copy2(ROOT / "LICENSE", module / "LICENSE")
        (module / "THIRD_PARTY_NOTICES.md").write_text(
            "# Third-party notices\n\nThe organizer supplies the common Python environment "
            "and the declared catalogue checkpoints. The archive contains original project code.\n",
            encoding="utf-8",
        )
        (source / "README.md").write_text(
            "# India Research Stage 1\n\n"
            "The module extracts PDF pages, ranks passages with a MiniLM encoder and a "
            "cross-encoder, and evaluates the organizer checklist with DeBERTa. "
            "It returns criterion statuses, exact PDF spans and an estimated TRL or abstention.\n\n"
            "Entry point: participant_module.adapter:AssessmentModule. "
            "Model weights come from APCTT_MODEL_STORE. The declared compute profile is CPU.\n",
            encoding="utf-8",
        )
        write_json(source / "submission.json", {
            **metadata,
            "submission_version": "apctt-submission-v3", "base_id": bases["base_id"],
            "license": "MIT", "compute": {"device": "cpu", "precision": "float32"},
            "settings": {}, "models": models,
        })
        write_json(source / "declaration.json", {
            **metadata,
            "declaration_version": "apctt-declaration-example-v1",
            "team_training": "no_team_training", "training": None,
            "data_sources": [
                report_source("NREL/TP-5000-62951", NREL_RM3),
                report_source("NREL/TP-5700-92489", NREL_HERO),
                {
                    "source": "Rahmani et al., Identifying Structural Priors in a Hybrid Differentiable Model for Stream Water Temperature Modeling",
                    "version": "Water Resources Research 59(12), e2023WR034420",
                    "date": "2026-11-19",
                    "role": "Offline verification on a software-model report",
                    "label_origin": "No expert TRL labels",
                    "linked_project": "https://doi.org/10.1029/2023WR034420",
                    "license": "CC-BY-4.0",
                },
                {
                    "source": "APCTT participant kit EXAMPLE_REPORT.pdf",
                    "version": "0.5.0",
                    "date": "2026-11-19",
                    "role": "Offline verification with the kit's synthetic example",
                    "label_origin": "Kit example with missing measured results",
                    "linked_project": "APCTT 2026 Stage 1",
                    "license": "MIT",
                },
            ],
            "inference_assets": {
                "source_code": "participant_module/ (identified by declared_package_sha256)",
                "weights": [], "prompts_or_rules": ["participant_module/research/production.py"],
                "reference_assets": [],
                "dependency_declarations": [bases["base_id"], *[m["catalog_id"] for m in models]],
            },
        })
        packed = subprocess.run(
            [sys.executable, str(checker), str(source), "--output", str(output)], cwd=kit,
        )
        if packed.returncode not in (0, 2):
            return packed.returncode
    return subprocess.run([sys.executable, str(checker), "--check", str(archive)], cwd=kit).returncode


if __name__ == "__main__":
    raise SystemExit(main())
