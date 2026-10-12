# Verification record — research release 0.2

Executed on 2026-10-12. Software checks and retrospective performance are separate forms of evidence.

## Complete-cohort execution

- All **40,336** released records (**1,552,210** hourly rows) were acquired from official PhysioNet sources. Every file's source, bytes, SHA256, exact expected filename set and header were checked. The manifest is local provenance, not a publisher signature.
- The frozen full configuration trained regularized logistic regression, XGBoost and a causal GRU on hospital A development partitions. Model and threshold selection were saved before held-out scoring. The selected family was XGBoost.
- The complete report contains six primary evaluations (three families × A-test/B), **18 selected-model stress conditions**, four component ablations evaluated in both held-out sets, and selective-prediction results for all three families. Primary, stress and ablation intervals use **1,000 patient bootstrap replicates**. Subgroup and selective-policy confidence intervals are not computed.
- [Full report and weights](../bundles/full-v1/), [failure analysis](../reports/full_failure_analysis.md), [cohort profile](../reports/full-v1-cohort/cohort_profile.md), [scientific figures](../reports/figures/full-v1/) and [measured resource usage](../reports/full-v1-telemetry.json) preserve the actual results. No patient-level data or predictions are included in the Git release or hosted application.
- The earlier synthetic experiment (360 patients / 15,168 hours) and real-data pilot (2,400 records / 91,276 hours) remain in separate bundles. The full B set contains the previously inspected pilot B records; it is not a wholly new confirmatory validation.

## Software behaviors verified

- Actual scikit-learn, XGBoost and PyTorch fits; an end-to-end training test; schema rejection of malformed records; deterministic patient-separated splits; and immutable completed output directories.
- Exact forward-only feature equivalence, train-only imputation, GRU sequence-prefix/padding behavior, delayed availability, and perturbation-prefix invariance. Prefix tests allow only the documented tiny floating-point differences for model scores; input features remain exactly compared.
- Official scorer bytes match their pinned SHA256. A golden fixture reproduces normalized utility. Weighted patient-cluster bootstrap AUROC/AP/Brier match explicitly repeated sklearn samples. Optimized batch ingestion and evaluation match their reference implementations.
- API synthetic replay, loaded-model/source disclosure, invalid inputs, future-observation mutation, unknown model/patient rejection and static routes. Only generated records are replayable.
- Exported logistic coefficients, XGBoost trees and GRU weights execute directly in JavaScript. Tests compare scores and frozen-threshold decisions with Python across generated trajectories; additional runtime contracts cover all models, perturbations, timeline prefixes, all four pages and JSON/CSV downloads. These are numerical/DOM-contract checks, not browser visual QA.
- Figure tests check source-backed values, confidence-interval and coverage reporting. Scientific figure PNGs are visually inspected; SVGs and underlying numeric JSON are preserved.
- Local MLflow SQLite tracking records actual experiment parameters, metrics and artifacts.

## Installed distributions, CI and containers

The [GitHub Actions workflow](https://github.com/ibrahimbaabdullah-jpg/clinical-ai-reliability-lab/actions/workflows/tests.yml) executes four jobs: the test suite, isolated installed-wheel verification, the minimal CPU container and the deep CPU container. Containers are **actually built and run**, including non-root identity, health check, default startup and trusted mounted-model replay. The wheel's training command and real HTTP routes run outside the source checkout.

The historical failure at commit `a6db880` came from model-score rounding differences between CPU environments. The comparison tolerances were corrected without weakening exact causal-feature tests. Subsequent runs at `8ff9541` and `7274308` passed all four jobs. CI results are attached to each GitHub commit; the workflow link above exposes the corresponding remote execution. Local final-release checks are recorded below.

## Limits

- Browser visual QA remains **unverified**. The browser-control skill required by the Sites workflow was unavailable. Numerical model parity, Node contracts and successful deployment do not establish desktop/mobile appearance, browser compatibility or accessibility conformance.
- Python 3.12 / Linux x86_64 and CPU inference are tested. Python 3.11/3.13, Intel macOS and GPU training are not verified. Direct versions and a Linux/Python-specific installed transitive dependency snapshot are recorded; no cross-platform or wheel-hash lockfile is claimed.
- Starlette's TestClient/httpx dependency emits a deprecation warning; it does not fail the tests.
- Patient bootstrap is conditional on the fitted models and fixed split. It does not capture repeat training, alternative partitions or repeated perturbation realizations. There are no paired significance tests or subgroup/selective-policy confidence intervals.
- A retrospective result from one transferred hospital does not demonstrate clinical benefit, safety, fairness or present-day transportability. Research manuscript author, funding, conflict-of-interest and local ethics declarations require factual author review before submission.
- The repository and hosted demonstration remain private. No public-facing clinical API or EHR integration is deployed.

The protocol's initial Git commit precedes fitting. This is a software/research provenance record, not a registered clinical trial.

## Final local release checks

- Full Python suite: **88 passed, 4 skipped**, one deprecation warning, 63.04 seconds. The four skipped tests require a running HTTP/container target and are exercised by the separate installed-wheel/container CI jobs.
- After final figure-label polish, the four figure tests passed again.
- Full-v1 report rendering contract, exported fitted-model parity over all 120 generated replay trajectories, and the complete static-runtime contract passed.
- SHA256 of the full report and all three fitted model files was unchanged before and after tests.
- All four scientific PNGs were inspected for labels, clipping, intervals and source consistency; tiny nonzero histogram proportions use `<0.01%`. Browser visual QA remains unperformed.
- [Local verification log](../reports/release-tests.log) preserves executed commands and counts. The [release evidence manifest](../reports/release_verification.json) links these checks to the report digest.
