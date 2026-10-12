# Benchmark execution report

**Evidence status: synthetic software demonstration**

Run UTC: 2026-10-12T16:48:46.635292+00:00. Patients: 360; hours: 15168.
Data SHA256: `2885311d57613663c6393f592462638e2675c168019cc6de28d0b3b00fe9a5d7`. Protocol SHA256: `fc905650c0da63faf0a9b960e4a7faadcf9de55d265cb9860b082d302b244dfc`.

These are retrospective software/research results. Synthetic results are not clinical evidence. Reduced-cohort results are preliminary, not the complete 40,336-patient benchmark.

Selected family: **logistic**, selected solely using development threshold-partition utility.

| Model | Evaluation set | AP | Utility | Brier | Event sensitivity | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|
| logistic | test | 0.9883 | 0.9308 | 0.0234 | 1.0000 | 1.5652 |
| logistic | external | 0.9713 | 0.8645 | 0.0344 | 1.0000 | 2.6354 |
| xgboost | test | 0.9818 | 0.8819 | 0.0291 | 1.0000 | 2.4348 |
| xgboost | external | 0.9717 | 0.8550 | 0.0393 | 1.0000 | 3.0940 |
| gru | test | 0.8773 | 0.8114 | 0.0668 | 1.0000 | 2.2609 |
| gru | external | 0.7987 | 0.7950 | 0.1235 | 0.9792 | 3.0879 |

95% patient-cluster intervals, exact denominators, excluded events, calibration bins, subgroups, stress experiments and selective coverage are in `report.json`.

## Interpretation limits

- Held-out B represents one hospital transfer, not universal transportability.
- Alert burden and pre-onset analysis use the prespecified six-hour window and six-hour refractory period.
- First-row-positive and unobserved inferred-onset events are excluded only from pre-onset event analysis; counts are reported.
- Perturbations are simulated; they do not identify actual clinical error mechanisms.
- Model contributions describe mathematical association, not causes or treatment recommendations.
- No prospective validation, clinician usability study or patient benefit has been established.

## Reproducibility

Configuration: `2f2e1c57cc6f3cca42c9349e9602fcd9b295ab94325c2eee4c15c538379cb6e7`. Split manifest: `split_manifest.json`. Dependencies: `environment.json`.
Primary bootstrap replicates: 50. See JSON for valid-replicate counts and undefined results.
