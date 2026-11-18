# Stage 1 validation

## Oops-Spotters 0.2.0

The APCTT 0.5.0 development server loaded the extracted final archive with verified package digest `929ae30d85645d529eafbefb72eae5a7ca3ce12e24af41d1e142d55d6f129573`. The official packer and archive checker returned `VALID_WITH_REVIEW` for the two NREL data-source notices.

- The HERO report returned TRL 2. C06 was insufficient and the previously cited planning paragraph was excluded. Other parts of the report describe ocean deployments. This run did not establish a C06 citation for them.
- The stream temperature model returned TRL 3 and met C03. The exact spans on PDF pages 11 and 12 describe test-period results and measured performance metrics.
- The official example with missing measurements abstained.
- RM3 returned TRL 4 with C04 met in 146.7 seconds in the final archive run, including model load. The cited span is on PDF page 13; the previous C01 bibliography citation is absent.
- A preliminary RM3 run hit the fixed 25-second reserve near the end of the checklist. The final package scales this reserve to the remaining criteria and keeps the worker's 180-second case deadline. The HERO, stream temperature and missing-measurements runs above predate this reserve-only change.
- With the actual catalogue models, a controlled example describing a planned field test did not satisfy C06. A separate example describing completed operation in the intended conditions satisfied C06.

Bibliography sections are omitted from evidence. C03 searches results sections within reports linked to the target. C06 checks completed operation and the target environment in the same passage. The tests do not establish expert-label accuracy or GPU performance.

## Historical baseline 0.1.0

The CPU dry-run archive was built with APCTT participant kit 0.5.0. Its participant module and assets had SHA-256 package digest `fab5a768f42bbbd4c3a8f32ff23f906c610072723978ba28a85bf9a924c437cd`. The development server verified the digest after fresh extraction. The archive checker returned `VALID_WITH_REVIEW` because the two NREL report sources use publication notices as their declared terms. The dry-run team ID is `INDIA-RESEARCH-DRY-RUN`.

## Results

| Case | PDF pages | Engine result | Criteria marked met | Exact spans | Seconds |
| --- | ---: | --- | --- | ---: | ---: |
| RM3 wave tank testing | 26 | TRL 4 | C01, C04 | 2 | 131.6 |
| HERO deployment report | 36 | TRL 6 | C02, C06 | 2 | 87.4 |
| Kit example missing measurements | 2 | Abstained | None | 0 | 16.3 |
| Stream temperature model paper | 21 | Abstained | None | 0 | 138.7 |

The first three cases ran through the kit's HTTP server and separate CPU worker. The stream temperature case ran the extracted adapter directly with three CPU threads and passed the kit's result validator. Timings include model loading. The test machine ran PyTorch 2.15.0+cpu, with no CUDA device. The archive delivered for review has the same participant source, model IDs, settings and asset digest as the tested extraction. Its declaration also records the supplemental validation inputs.

## Evidence review

- RM3 C04 on PDF page 13 includes measurements of the locked floating-point absorber in a wave tank. C01 cites a bibliography on PDF page 26, which does not state the underlying principles.
- HERO C06 on PDF page 5 cites program plans and the intended build and test activity. The quoted span does not establish a completed demonstration. TRL 6 is the engine's output. The demonstration criterion needs stronger evidence grounding.
- The kit example reports a laboratory test without measured results. The engine abstained. The stream temperature paper reports completed experiments, and the engine left all criteria at insufficient evidence. The software evidence path requires investigation.

The kit's three designated USGS cases could not be downloaded by the official downloader in this environment (HTTP 403). The public manifest provides organizer estimates but no expert criterion labels or gold evidence pages. This run therefore establishes package and runtime behavior and exposes evidence errors. It does not estimate accuracy or GPU performance.

## Sources

- [NREL RM3 wave tank technical report](https://docs.nlr.gov/docs/fy15osti/62951.pdf)
- [NREL HERO deployment technical report](https://docs.nlr.gov/docs/fy26osti/92489.pdf)
- [Rahmani et al., Water Resources Research](https://doi.org/10.1029/2023WR034420)
- APCTT participant kit 0.5.0, `contracts/public-cases.json` and `docs/EXAMPLE_REPORT.pdf`
