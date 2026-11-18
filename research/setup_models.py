"""Download and verify three pinned APCTT checkpoints for local development."""
from pathlib import Path
import argparse
import hashlib
import os

from .catalog import DEFAULT_MODELS, default_model_paths

# Organizer-published SHA-256 hashes for the model.safetensors files.
WEIGHT_HASHES = {
    "encoder": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "reranker": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    "nli": "605e40d2020f4c66666db3e98a4bc277b6214565eac95f35c1a5027a93ee0ef0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, default=Path(os.environ.get("APCTT_MODEL_STORE", ".models")))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for role, path in default_model_paths(args.model_store).items():
        catalog_id, name, revision = DEFAULT_MODELS[role]
        snapshot_download(
            repo_id=name, revision=revision, local_dir=path,
            allow_patterns=["*.json", "model.safetensors", "vocab.txt", "spm.model", "LICENSE*"],
        )
        with (path / "model.safetensors").open("rb") as model_file:
            if hashlib.file_digest(model_file, "sha256").hexdigest() != WEIGHT_HASHES[role]:
                raise SystemExit(f"Checksum mismatch for {catalog_id}: {path / 'model.safetensors'}")
        print(f"{role}: {path.resolve()}")


if __name__ == "__main__":
    main()
