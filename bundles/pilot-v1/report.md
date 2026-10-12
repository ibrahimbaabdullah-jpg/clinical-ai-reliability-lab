# Benchmark execution report

**Evidence status: reduced-cohort preliminary experiment**

Run UTC: 2026-10-12T16:58:15.050506+00:00. Patients: 2400; hours: 91276.
Data SHA256: `bb6581a682bfe75d9933435b8668fb17c57664db4948e63f98699dad2eb5b935`. Protocol SHA256: `fc905650c0da63faf0a9b960e4a7faadcf9de55d265cb9860b082d302b244dfc`.

These are retrospective software/research results. Synthetic results are not clinical evidence. Reduced-cohort results are preliminary, not the complete 40,336-patient benchmark.

Selected family: **xgboost**, selected solely using development threshold-partition utility.

| Model | Evaluation set | AP | Utility | Brier | Event sensitivity | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|
| logistic | test | 0.1570 | 0.2945 | 0.0169 | 0.4444 | 0.5656 |
| logistic | external | 0.0272 | 0.0281 | 0.0140 | 0.4259 | 0.7519 |
| xgboost | test | 0.0871 | 0.2978 | 0.0170 | 0.3333 | 0.2243 |
| xgboost | external | 0.0578 | 0.1025 | 0.0136 | 0.1852 | 0.0846 |
| gru | test | 0.0367 | 0.1346 | 0.0172 | 0.4444 | 0.8631 |
| gru | external | 0.0291 | 0.0255 | 0.0137 | 0.1481 | 0.3109 |

95% patient-cluster intervals, exact denominators, excluded events, calibration bins, subgroups, stress experiments and selective coverage are in `report.json`.

## Interpretation limits

- Held-out B represents one hospital transfer, not universal transportability.
- Alert burden and pre-onset analysis use the prespecified six-hour window and six-hour refractory period.
- First-row-positive and unobserved inferred-onset events are excluded only from pre-onset event analysis; counts are reported.
- Perturbations are simulated; they do not identify actual clinical error mechanisms.
- Model contributions describe mathematical association, not causes or treatment recommendations.
- No prospective validation, clinician usability study or patient benefit has been established.

## Reproducibility

Configuration: `d225613229c34dd7a12dea1232371408508f15aa8ce1326224540d2a9c85b740`. Split manifest: `split_manifest.json`. Dependencies: `environment.json`.
Primary bootstrap replicates: 200. See JSON for valid-replicate counts and undefined results.
