# Architecture and researcher workflow

## Data path

`download_data.py` records source URLs and SHA256 hashes. `data.py` validates hourly PSV files, namespaces patients by hospital, assigns frozen patient splits and stores Parquet. DuckDB produces cohort summaries without loading all hourly rows.

`features.py` builds only current/past predictors. `models.py` owns fitted preprocessing, tune-set model selection and model serialization. `evaluation.py` scores complete patient trajectories with the unmodified official challenge utility implementation and a separate pre-onset analysis. `perturbations.py` changes raw available measurements before rebuilding features.

`cli.py` coordinates an immutable run directory containing configuration, provenance, split manifest, fitted models, calibration, thresholds, patient predictions, aggregate results and optional local MLflow tracking. No external API is needed.

## Interfaces

FastAPI serves the local dashboard and JSON endpoints. With no run configured, it fits a small synthetic logistic example on first access. With `CLINICAL_LAB_RUN`, it loads a trusted experiment and displays its actual results. **Patient replay always uses synthetic records**, even when the selected model was trained on PhysioNet. Applying a real-data model to generated patients is an out-of-distribution illustration.

The replay endpoint runs the real fitted predictor on the supplied prefix of a bundled synthetic timeline. It does not call an LLM. Missingness, delay and noise controls trigger the same raw-observation perturbation mechanism as the offline experiments. Contributions for logistic/XGBoost are additive uncalibrated log-odds contributions, not causes or clinical recommendations. GRU attribution is intentionally unavailable until a validated method is added.

The hosted edition runs `inference.js` directly in the browser using parameters exported from those same fitted models. It includes causal feature construction, train-fitted imputation/scaling, actual logistic coefficients, XGBoost trees, GRU weights, calibration and frozen thresholds. It is a direct export rather than a surrogate. Parity tests compare both numeric scores and alert decisions with Python. The hosted application contains aggregate reports and newly generated unlabeled timelines, with no patient uploads, clinical records, joblib files or inference server. Logistic contributions are available; native XGBoost TreeSHAP remains a Python-only capability.

`scripts/build_site.py` creates the complete static deployment from an explicit completed run. The Site's authenticated owner-private access controls protect the hosted edition. Browser perturbations share the benchmark's definitions but use a different deterministic generator, so individual draws are illustrative; stored aggregate results always remain the Python benchmark output.

## What a user can do

1. Open **Overview** to compare actual A-test and B results, study scope and model selection.
2. Open **Patient replay**, choose a synthetic trajectory and model, and move through observed hours. Compare original and perturbed scores and inspect missing/stale measurements.
3. Open **Reliability** to inspect calibration, transport, stress tests, feature/calibration ablations, recorded demographic subgroups, confidence intervals and selective prediction. Download the complete JSON report or primary metrics CSV.
4. Open **Methods** to understand label alignment, alert episode definitions, exclusions and limits.
5. Run the CLI to reproduce training and generate machine-readable and Markdown reports.

## Operational boundaries

The Python API's default bind address is localhost, and the API itself has no authentication layer. The separate hosted static edition uses the Site platform's owner authentication. No EHR integration, patient uploads, multitenancy or public clinical inference endpoint is supplied. This software does not diagnose sepsis, issue treatment advice or establish clinical benefit.

Only trusted local joblib model artifacts may be loaded: pickle deserialization can execute code. Browser requests cannot choose artifact paths. Raw data, patient-level predictions, credentials, environments and local tracking databases are ignored by Git. Source data and third-party code retain their own license notices.
