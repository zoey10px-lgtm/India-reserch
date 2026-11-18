# India Research — Evidence-grounded TRL engine

An offline PDF assessment engine for APCTT's Technology Readiness Level (TRL) task. The CLI and the participant adapter use the same inference pipeline.

1. Extract page text, retaining the exact source spans.
2. Embed passages with MiniLM and rerank the candidates with a cross-encoder.
3. Check each criterion with DeBERTa. For contradictions, also test whether the passage supports an explicit denial. For laboratory and relevant-environment validation, require the test article, environment and measured or observed outcome in the same passage.
4. Return criterion judgments, supporting pages and the highest supported consecutive TRL in the supplied checklist. A missing checkpoint is a model initialization error. Insufficient evidence yields an abstention.

## Install and run

Use Python 3.12. Install a CPU or CUDA build of PyTorch as appropriate for your machine.

```bash
python -m pip install -r requirements-semantic.txt
python -m research.setup_models
python -m research.demo /path/to/report.pdf --target "Technology under assessment" \
  --checklist /path/to/checklist.json
```

`setup_models` downloads and verifies three checkpoints against the APCTT 0.5.0 catalogue's pinned revisions and SHA-256 weight hashes. It uses `.models` by default. Both commands accept `--model-store` and honor `APCTT_MODEL_STORE`. Model inference loads from local files.

| Stage | APCTT catalogue ID |
| --- | --- |
| Embedding | `all-minilm-l6-v2` |
| Reranking | `ms-marco-minilm-l6-v2` |
| NLI | `deberta-v3-base-zeroshot-v2.0-c` |

The CLI accepts the kit's `contracts/checklist.official.json` or a research checklist with `criterion_id` and `description` for each criterion. The JSON output contains the model revisions, status, estimated TRL and per-criterion page citations. For a scored criterion, each quote is copied from the extracted PDF page. The engine leaves statistical confidence unset until calibrated on labeled data.

## Reproduce the public PDF check

```bash
curl -fL https://www.sandia.gov/app/uploads/sites/273/dlm_uploads/RM3-TANK-TEST-NREL-TP-5000-62951.pdf -o rm3.pdf
python -m research.demo rm3.pdf --target "RM3 wave energy converter" \
  --checklist docs/rm3-smoke-checklist.json
```

The example checklist is a small integration check for the report's wave-tank experiment. The official evaluation uses the kit's twelve-criterion checklist.

## Run in the organizer's development server

1. Run `python tools/stage_submission.py` to vendor the inference code in `participant_module/`.
2. In APCTT kit 0.5.0, use `participant_module.adapter:AssessmentModule` as the entry point in a copy of `config/dev.json`. Set `module_root` to the directory containing `participant_module/`, and set `package_root` and `asset_root` to the staged `participant_module/` for local checks. Set `example_only` to `false`.
3. Use the kit's official checklist. Set `worker.python` to the Python interpreter with the inference dependencies and set `worker.extra_env.APCTT_MODEL_STORE` to the absolute model-store path. The three catalogue IDs above resolve with empty module settings. To override an ID, supply the kit's catalogue JSON through `module.settings.catalog_path` or `APCTT_MODEL_CATALOG`.
4. Start the kit server. Generate a request with the kit's `tools/make_request.py`, using target, application and environment values appropriate for the PDF, then submit it to `POST /v1/assessments`.

`python -m unittest discover -s tests` checks model failure handling, criterion logic and exact-span behavior with deterministic fixtures.

The APCTT 0.5.0 CPU development server accepted the adapter's twelve-criterion result for the 26-page RM3 report in 73.5 seconds, including loading. The official-checklist run abstained. Evaluating classification and citation accuracy requires labeled reports. This repository contains the implementation and development instructions. Event CUDA performance and submission ZIP validation require separate runs.
