# APCTT 0.5 integration audit (2026-11-19)

## Verified from repository source
- `participant_module/adapter.py` exposes `load`, `assess`, `reset_case`, and `close`.
- The adapter deliberately abstains if model catalogue IDs or the catalogue path are missing; model loading exceptions also result in abstention.
- `research/production.py` calls retrieval and batched NLI, checks artifact/environment/outcome evidence, and computes cumulative TRL.
- Existing code does not establish that any catalogue-approved model weights were loaded or run.

## Blocking checks before calling this a completed engine
1. Compare adapter output field names and permitted values against the **actual** 0.5.0 `DATA_CONTRACT.md` and checklist.
2. Verify catalogue IDs and local model paths; run real embedding, reranker, and NLI inference, not mocked tests.
3. Verify TRL 4–9 mapping and whether earlier levels must be inferred, rather than assuming C01–C09 map directly to TRL 1–9.
4. Test a completed experiment against a merely proposed future experiment, a contradictory result, an unrelated target, and absent evidence.
5. Benchmark accuracy on independently labeled cases, including evidence citation correctness and abstention.
6. Measure CPU runtime and memory under the organizer's time limit; then compare CUDA.
7. Stage dependencies and run the official submission validator. No completion claim until these checks pass.

## Specific code risks observed
- `research/production.py` considers a criterion met when artifact/environment/outcome are independently supported across positive passages; this may combine unrelated experiments unless entity and experiment identity are checked.
- `participant_module/adapter.py` maps internal `contradicted` and `conflict` statuses to `insufficient evidence`; preserve diagnostic distinctions internally and verify permitted external enum values.
- Broad exception handling returns abstention; useful for robustness but can hide systematic model/contract failures without structured error diagnostics.

This audit records findings, not test success. Do not submit this branch as validated.
