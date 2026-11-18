"""Stage the standalone submission package for official kit validation.

Run: python tools/stage_submission.py
Then use the official APCTT make_submission.py to produce/validate the ZIP.
This script does NOT claim to validate organizer schema or model catalogue.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research"
DEST = ROOT / "participant_module" / "research"

def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SRC, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "demo.py", "setup_models.py"))
    (ROOT / "participant_module" / "__init__.py").touch(exist_ok=True)
    print(f"Staged {DEST}; now run the official kit validation.")

if __name__ == "__main__":
    main()
