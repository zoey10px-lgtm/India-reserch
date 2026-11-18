# India Research — Evidence-grounded TRL engine

Independent research prototype for APCTT TRL 4–9 assessment. **Not yet an official submission**.

## Design
1. Extract page-grounded evidence (PDF parser is a separate integration step).
2. Retrieve candidates by token overlap, optionally augment with an offline embedding retriever.
3. Check whether evidence refers to the *target* technology.
4. Infer entailment / contradiction / neutral against criterion hypotheses (pluggable NLI).
5. Deduplicate evidence, flag conflicting claims, aggregate per criterion.
6. Decide the highest supported TRL **without a default TRL 6**. Abstain on insufficient evidence.

The initial engine runs using Python's standard library and includes a deterministic lexical inference fallback for smoke tests. **The fallback is not a substitute for a trained NLI model and must not be used to claim model accuracy.**

## Run
```bash
python -m unittest discover -s tests -v
python -m research.demo
```

## Next integration gates
- Confirm exact APCTT checklist IDs, official input/output contracts, model catalogue and offline paths.
- Implement catalogue NLI adapter and calibrated scores; do not download models at runtime.
- Implement page-preserving PDF extraction, table support, time-budget enforcement.
- Implement `participant_module.adapter.AssessmentModule` matching the official kit.
- Compare against professor v4 on identical held-out cases (TRL MAE, accuracy, evidence precision, abstention, runtime).
- Build FastAPI/Render dashboard only after offline engine contract tests pass.

No private PDFs or benchmark labels are committed. MIT license applies to original code.

## Implementation update (2026-11-18)
Implemented modules: `research/pdf_parser.py`, `ocr.py`, `two_stage_retrieval.py`, `offline_nli.py`, `verification.py`, `pipeline.py`, `graph.py`, `benchmark.py` and `participant_module/adapter.py`.

**Critical limitations:** The real NLI/embedding/reranker model paths have not been provided or exercised in this environment. The adapter's official C01–C12 mappings and exact output contract must be verified with the competition kit. OCR is available as a module but not yet wired into `read_pdf`. The current evidence graph implements duplicate/conflict candidates, not temporal/entity coreference. Confidence remains uncalibrated (null). The web dashboard and Render deployment are not built. Do not submit this repository without official validation.

### Required model settings
Configure `module.settings.nli_model_path`, `encoder_model_path`, `reranker_model_path` as locally available, catalogue-approved model directories. The adapter uses `APCTT_MODEL_STORE` to resolve relative paths. It never downloads weights.

### Testing
`python -m unittest discover -s tests -v` tests deterministic orchestration, not actual model quality. Official kit validation and model inference tests remain required.

## Current engine integration (revision 2026-11-18)
- `participant_module/adapter.py` now calls `research/production.py` via a self-contained staging copy.
- `research/production.py` implements batched criterion NLI plus artifact/environment/outcome checks, target filtering, grounded citations, conflicts, sequential TRL decisions, and configurable quality caps.
- `research/pdf_parser.py` preserves exact source spans and invokes conditional OCR.
- `tests/test_production.py` covers five deterministic pipeline scenarios. These are NOT real-model benchmarks.
- Stage dependencies with `python tools/stage_submission.py` before attempting the official submission packager.

**Still required for a verified, submit-ready engine:** approved catalogue IDs and actual model weights; official `DATA_CONTRACT` and model-store paths; full integration test on the organizer's kit; runtime profiling against the three-minute cap; accuracy validation on labeled reports; calibrated confidence; any required package manifest/hash regeneration. Do not submit an unvalidated zip. A missing model causes abstention, not a fake prediction.
