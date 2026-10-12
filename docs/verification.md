# Verification record — 2026-10-12

## Executed

- **60 Python tests passed** on Python 3.12 / Linux x86_64, including actual scikit-learn, XGBoost and PyTorch fits.
- The full synthetic pipeline trained all three families on 360 generated patients / 15,168 hourly rows, then evaluated transfer, 18 perturbation conditions across two held-out sets, feature/calibration ablations and selective prediction.
- Official scoring code bytes match the pinned upstream SHA256. A golden fixture reproduces its normalized utility exactly. Weighted cluster-bootstrap AUROC/AP/Brier match explicitly repeated sklearn samples.
- Leakage tests cover patient-separated partitions, train-only imputation, forward-only features, GRU sequence prefixes/padding, delayed availability and perturbation-prefix invariance.
- Schema tests reject malformed PSV rows, changed column counts, invalid labels/times and duplicate records. Batched and unbatched feature construction agree exactly above 20,000 rows.
- API tests cover synthetic replay, actual inference, model-source disclosure, invalid inputs, unknown models/patients, future-value mutation and static routes. JavaScript syntax passes.
- A complete end-to-end integration test verifies that model selection is saved before B evaluation and that all experiment/report paths produce JSON-safe outputs.
- Local MLflow SQLite tracking was exercised successfully.
- 2,400 official PhysioNet patient files were downloaded with seed-2019 uniform filename sampling, 1,200 per hospital; recorded hashes, source URLs, byte counts and exact file set were verified before fitting.
- **The real-data pilot completed:** 2,400 patients / 91,276 patient-hours, all three fitted families, 200 bootstrap replicates, 18 stress conditions, four ablations, and selective coverage. Aggregate report and trained artifacts are in `bundles/pilot-v1`. XGBoost was selected on development data; external pre-onset sensitivity was 10/54 (18.5%), so these models cannot support a clinical-use claim.
- A pure-JavaScript rendering contract check against the actual experiment report passes: correct source disclosures, selected-family marker, missingness severity units, calibration and withheld-event fields. This is not browser visual QA.

## Pending or limited

- The complete 40,336-patient / 1,000-bootstrap configuration has not been executed. The clinical-data experiment provided here is preliminary and reduced-cohort.
- Docker is not installed in the execution environment, so no image-build or container-runtime success is claimed.
- Browser visual QA could not run: the available cloud browser blocked localhost with `ERR_BLOCKED_BY_CLIENT`. API and JavaScript checks passed; desktop/mobile layout appearance has not been visually verified.
- The GitHub Actions workflow is supplied; its remote outcome must be checked separately from local test results.
- Python 3.11/3.13, Intel macOS, GPU training and hosted deployment have not been tested.
- One dependency deprecation warning was emitted by Starlette's current TestClient/httpx integration; it did not cause failures.

## Interpretation

Passing tests establishes specific software behaviors, not clinical validity. Synthetic event rates and physiology were deliberately enriched for demonstration. The preliminary sample cannot establish a final model winner, clinical benefit, generalizability across hospitals, fairness or safety of abstention. The repository's initial protocol commit precedes model fitting; this is not clinical trial registration.
