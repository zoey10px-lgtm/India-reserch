# ModernBERT PDF evaluation experiment

Copy participant_module into the APCTT project. Use the bundled assets/modernbert_trl as the worker checkpoint. Keep the checkpoint and tokenizer files together.

Set APCTT_MODEL_STORE to the offline model store. The package declares
all-minilm-l6-v2, ms-marco-minilm-l6-v2, and modernbert-base. When using
APCTT_MODEL_CATALOG, point it to the matching model catalog JSON.

Model weights load from the fine-tuned checkpoint. The adapter checks
the catalog tokenizer against the training tokenizer and uses the
checkpoint tokenizer if they differ. The adapter filters structural PDF metadata and scores pooled excerpts across
C01 through C09. Each excerpt may support multiple criteria. TRL 1–7
use threshold 0.75 and TRL 8–9 use threshold 0.98. The highest accepted
stage is returned.
This is the selected project package. It does not include an official submission manifest.

Exact provisional TRL matches: 3/6 on the original evaluation subset and 0/5 on the additional subset (3/11 combined).

Full checkpoint bundle: `Oops-Spotters_India_Research_ModernBERT_final.zip`. The checkpoint is supplied separately from this source directory.

Full ZIP SHA-256: `f88d9cba41184f45226a182695a2acdd9819c0dad817050b909e864afaa65e41`
