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
