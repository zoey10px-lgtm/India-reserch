# Implementation gates

## Completed in first commit
- Dependency-free criterion engine with explicit abstention
- Explicit target mismatch rejection
- Injected NLI interface and a **non-ML** lexical placeholder
- Evidence deduplication, polarity conflict handling
- Benchmark metric helpers and unit-test fixtures

## Not yet implemented (do not present as finished)
1. Official APCTT assessment adapter: confirm exact schema using official kit.
2. Page-grounded PDF parser and target attribution for unlabeled passages.
3. Catalogue-approved offline NLI and embedding adapters; verify license/model IDs.
4. Real graph relations (entity, time, version, provenance), not just conflict candidates.
5. Calibration using held-out labeled evidence, rather than invented confidence.
6. Criterion completeness and official TRL aggregation semantics.
7. Benchmark professor v4 and proposed model on identical datasets.
8. Render UI, API, upload limits, asynchronous jobs, result visualizations.
9. Offline runtime profiling and submission zip validation.

## Research safety checks
- Do not leak held-out labels into prompts, retrieval or thresholds.
- Do not report accuracy until the actual model is run.
- Do not claim `confidence` is a probability before calibration.
- Do not assume a missing criterion is false.
- Preserve baseline code separately only with appropriate permissions.
