# Benchmark execution report

**Evidence status: complete public cohort retrospective experiment**

Run UTC: 2026-10-12T17:37:52.416100+00:00. Patients: 40336; hours: 1552210.
Data SHA256: `709d62e4470467bf7aa7c5579e01538de78717faa5b8a87cfeeb1b7dfe55f05e`. Protocol SHA256: `fc905650c0da63faf0a9b960e4a7faadcf9de55d265cb9860b082d302b244dfc`.

These are retrospective software/research results. Synthetic results are not clinical evidence. Reduced-cohort results are preliminary, not the complete 40,336-patient benchmark.

Selected family: **xgboost**, selected solely using development threshold-partition utility.

| Model | Evaluation set | AP | Utility | Brier | Event sensitivity | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|
| logistic | test | 0.1027 | 0.3970 | 0.0216 | 0.8375 | 1.4931 |
| logistic | external | 0.0534 | 0.1605 | 0.0143 | 0.6148 | 0.9111 |
| xgboost | test | 0.1622 | 0.4087 | 0.0198 | 0.7688 | 1.1589 |
| xgboost | external | 0.0797 | 0.2744 | 0.0137 | 0.6049 | 0.5895 |
| gru | test | 0.1043 | 0.3280 | 0.0205 | 0.7750 | 1.5933 |
| gru | external | 0.0523 | 0.1603 | 0.0137 | 0.6689 | 1.1067 |

95% patient-cluster intervals, exact denominators, excluded events, calibration bins, subgroups, stress experiments and selective coverage are in `report.json`.

## Interpretation limits

- Held-out B represents one hospital transfer, not universal transportability.
- Alert burden and pre-onset analysis use the prespecified six-hour window and six-hour refractory period.
- First-row-positive and unobserved inferred-onset events are excluded only from pre-onset event analysis; counts are reported.
- Perturbations are simulated; they do not identify actual clinical error mechanisms.
- Model contributions describe mathematical association, not causes or treatment recommendations.
- No prospective validation, clinician usability study or patient benefit has been established.

## Reproducibility

Configuration: `e5db7e3a5d40a707c5a15ff545388c1680c6ffee0707b7a5c166f1536a76305c`. Split manifest: `split_manifest.json`. Dependencies: `environment.json`.
Primary bootstrap replicates: 1000. See JSON for valid-replicate counts and undefined results.
