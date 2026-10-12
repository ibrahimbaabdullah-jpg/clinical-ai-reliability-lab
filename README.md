# Clinical AI Reliability Lab

**A reproducible research platform for ICU sepsis models, hospital transfer and data-quality stress testing.**

Which models provide useful early sepsis warnings, and how do calibration, missed events and alert burden change when observations or the hospital population change?

This project trains real machine-learning models; an LLM API is not required. It combines Python data engineering, classical and temporal models, statistical evaluation, model interpretation, an inference API and a research dashboard.

## Evidence status

Version **0.2** completes the frozen **40,336-record / 1,552,210-hour** retrospective benchmark: all three model families, 1,000 patient-bootstrap replicates, hospital transfer, 18 stress conditions, four ablations and selective prediction. The protocol and full configuration were frozen before fitting. Fitted models, aggregate reports and reproducibility manifests are preserved in [bundles/full-v1](bundles/full-v1/).

**[Open the interactive dashboard](https://clinical-ai-reliability-lab.ibrahimbaabdullah.chatgpt.site)** · **[دليل الاستخدام بالعربية](docs/user_guide_ar.md)** · [Full benchmark](bundles/full-v1/report.md) · [Failure analysis](reports/full_failure_analysis.md)

The hosted site and GitHub repository are currently private to the owner. The dashboard runs the actual exported fitted models on synthetic timelines; the benchmark panels use the recorded real-data evaluation.

### Hospital B results

B contains 20,000 records, 761,995 hours and 906 eligible pre-onset events. The table shows fixed development-selected thresholds; it is not a comparison after tuning thresholds on B.

| Model | Average precision | Brier score ↓ | Detected eligible events | False alerts / eligible day |
|---|---:|---:|---:|---:|
| Logistic regression | 0.0534 | 0.01425 | 557/906 (61.5%) | 0.911 |
| XGBoost — development-selected | 0.0797 | 0.01367 | 548/906 (60.5%) | 0.590 |
| Temporal GRU | 0.0523 | 0.01369 | 606/906 (66.9%) | 1.107 |

The selected XGBoost model **missed 358 of 906 eligible B events**. Event sensitivity was 60.5% (95% patient-bootstrap interval 57.4–63.5%). All estimates, confidence intervals, exclusions and denominators are in the [machine-readable report](bundles/full-v1/report.json). The false-alert denominator is eligible **non-warning** patient-time, not all patient-days. No clinical benefit, prospective validation or safe clinical deployment is established.

The complete B cohort includes the 1,200 B records previously inspected in the pilot (6%). No model choice was changed using B, but this is **not a wholly new, never-inspected confirmatory external validation**. The [verification record](docs/verification.md) separates executed checks from remaining limitations.

## What the application does

| Workspace | Researcher actions and benefit |
|---|---|
| Overview | Compare fitted models on internal and external cohorts, with the dataset scope visible. |
| Patient replay | Select a synthetic patient and observed hour, inspect scores, missing/stale observations and available feature contributions. Logistic contributions work in both editions; XGBoost TreeSHAP is available in the Python edition. |
| Reliability | Examine calibration, bootstrap intervals, hospital transfer, stress tests and selective-prediction results. |
| Methods | Inspect the outcome definition, split design, alert policy and limitations. |

The replay endpoint executes the fitted model on a timeline prefix. It always uses synthetic patient records. A real-data model applied to synthetic trajectories is explicitly an out-of-distribution demonstration. No patient uploads or treatment recommendations are supported.

## Quick start — local dashboard

Python 3.12 is the tested version (project range: 3.11–3.13). Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,tracking,reporting,acquisition]'
python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The first request trains a small deterministic synthetic logistic baseline locally. It requires no account, API key or downloaded clinical dataset. **http://127.0.0.1:8000/openapi.json** exposes the API schema. On Windows activate the virtual environment using its Scripts/activate script.

## Load the completed benchmark or train new models

Install PyTorch for your platform. The verified Linux CPU command is:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
CLINICAL_LAB_RUN=bundles/full-v1 python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000

# Or reproduce the smaller synthetic workflow in a new run directory
python -m clinical_lab.cli run --synthetic --config configs/demo.yaml --output artifacts/demo
CLINICAL_LAB_RUN=artifacts/demo python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

On an Intel Mac, the default logistic dashboard can run independently of PyTorch; use a compatible Linux CPU environment or the optional deep-learning Docker image for the full GRU environment. The current PyTorch dependency targets modern wheels and has not been validated on Intel macOS.

Each run saves actual fitted models, calibrators and thresholds, frozen split manifests, input hashes, dependency versions, Parquet predictions, JSON results and a Markdown report. Raw observations and patient-level outputs are local and ignored by Git. Selected aggregate reports and trusted model checkpoints from the executed runs are preserved in [bundles](bundles/). Use a **new output directory** for each completed experiment; the CLI refuses to overwrite completed reports.

## PhysioNet data and experiments

The target is the public [PhysioNet/CinC 2019 v1.0.0 dataset](https://physionet.org/content/challenge-2019/1.0.0/): 20,336 hospital A and 20,000 hospital B patient records. One pipe-delimited file represents an ICU stay, with hourly rows, 40 predictors and the supplied label. Public downloads need no account.

```bash
# Preliminary verification: uniform filename sample, no outcome selection
python scripts/download_data.py --output data/raw --limit-per-hospital 1200 --seed 2019
python -m clinical_lab.cli run --data data/raw --config configs/pilot.yaml --output artifacts/pilot

# Complete cohort: separate location preserves the preliminary sample
python scripts/download_data.py --source s3 --output data/full --workers 8
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/run_measured.py --telemetry artifacts/full-repeat-telemetry.json run --data data/full --config configs/benchmark.yaml --output artifacts/full-repeat
```

The downloader uses official URLs, bounded concurrency and SHA256 integrity manifests. The CLI verifies recorded file hashes before fitting. Those are local download-integrity checks, not publisher signatures. See [provenance](docs/data_provenance.md): the landing-page and bundled license notices differ, so this repository does not redistribute patient records or assert an unambiguous data license.

The executed full configuration uses 1,000 patient-bootstrap replicates and ran on CPU in **34.8 minutes**, with **6.54 GiB peak child-process RSS**. This is an observed run in a shared Linux environment, not a guaranteed runtime or total-machine memory requirement. [Telemetry](reports/full-v1-telemetry.json) records the exact command, wall/CPU time and memory definition. A GPU is optional; these results were produced without one.

## Prespecified evaluation

The [protocol](docs/protocol.md) was committed before model fitting. Hospital A is split by patient into **55% train / 15% tune / 10% calibrate / 10% threshold / 10% internal test**, stratified by ever-positive status. Hospital B is external only. Fit preprocessing on train, select hyperparameters on tune AP, fit sigmoid calibration on calibrate, and select thresholds and the final family on threshold-partition challenge utility. Freeze these choices before inspecting test metrics.

**The supplied labels already start six hours before inferred onset and remain positive afterward.** Do not shift them again. The main hourly score is a probability of the supplied challenge label, not a validated six-hour incident-sepsis risk among event-free patients.

- Original task: official challenge utility, AUROC, average precision, Brier score and fixed-bin calibration.
- Separate pre-onset analysis: warnings in `[onset−6, onset)`, six-hour alert refractory period, eligible event sensitivity, missed events, warning time and false alert episodes per patient-day. Left-censored and beyond-followup onset cases are counted and excluded only from this endpoint.
- Reliability: hospital transfer; deletion 10/30/50%; delay 1/3/6 hours; Gaussian errors at 0.1/0.25/0.5 training SD; logistic feature ablations; selected-family calibration ablation.
- Selective prediction: development-derived cutoffs, retained coverage/errors and withheld events, with all-event sensitivity counting withholding as no alert. It is a heuristic experiment, not a safety guarantee.
- Uncertainty and subgroups: patient-cluster confidence intervals for primary, stress and ablation metrics; recorded age/Gender strata with small-group suppression. Subgroup and selective-policy intervals are not implemented. Missing demographics and excluded event counts remain visible.

Every primary estimate has exact denominators and valid bootstrap replicate counts in `report.json`. Reports explicitly distinguish synthetic, preliminary and complete-cohort evidence. Marginal intervals are descriptive, not paired statistical significance tests.

## Model families and tools

| Component | Implementation |
|---|---|
| Interpretable baseline | scikit-learn regularized logistic regression, train-only median imputation and scaling |
| Boosted trees | XGBoost histogram trees with tune-set model selection |
| Temporal learning | PyTorch unidirectional GRU, packed variable-length sequences, masked loss and early stopping |
| Data | Pandas/NumPy, Parquet and DuckDB |
| Tracking | Optional local MLflow with SQLite; no cloud account |
| App | FastAPI local dashboard and a private hosted HTML/CSS/JavaScript edition with exported fitted-model inference |
| Reproducibility | Configurations, manifests, tests, GitHub Actions workflow and Dockerfile |

`requirements-tested.txt` records direct dependency versions; `requirements-linux-py312.txt` adds the installed transitive dependency closure for Linux x86_64 / Python 3.12. CI uses the latter as pip constraints. These are version snapshots, not cross-platform wheel-hash lockfiles. Each run records its actual environment. No LLM account or API is needed.

## Tests and containers

```bash
python -m pytest

# Default synthetic logistic dashboard
docker build -t clinical-ai-reliability-lab .
docker run --rm -p 127.0.0.1:8000:8000 clinical-ai-reliability-lab

# Include GRU loading for a previously trained run
docker build --build-arg WITH_DEEP=1 -t clinical-ai-reliability-lab:deep .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/bundles/full-v1:/app/run:ro" -e CLINICAL_LAB_RUN=/app/run clinical-ai-reliability-lab:deep
```

Only load trusted model artifacts: joblib deserialization can execute Python. The default API binds locally; authentication and public-service hardening are separate deployment work. GitHub Actions builds and exercises both CPU images, the installed wheel, and the static inference runtime; see [verification](docs/verification.md). Browser visual QA remains unverified because the required browser-control capability was unavailable.

## Skills demonstrated by concrete evidence

| Skill | Reviewable project evidence |
|---|---|
| Problem formulation | [Prespecified population, targets and alert tradeoffs](docs/protocol.md) |
| Python and data engineering | [Validated ingestion](clinical_lab/data.py), [causal features](clinical_lab/features.py), Parquet/DuckDB and source integrity manifests |
| Statistics | [Patient-cluster intervals, calibration and event denominators](clinical_lab/evaluation.py), with source-backed scientific figures |
| Model development | [Three fitted families, tuning and early stopping](clinical_lab/models.py), saved preprocessing/calibration and component ablations |
| Rigorous evaluation | Frozen hospital transfer, observation stress, withheld-event analysis and [leakage/regression tests](tests/) |
| Reproducible software | Training CLI, experiment tracking, [API](clinical_lab/api.py), numerical browser-model parity, CI, installed wheel and Docker |
| Responsible communication | [Model card](docs/model_card.md), actual [failure analysis](reports/full_failure_analysis.md), synthetic-only replay and [TRIPOD+AI gap mapping](docs/tripod_ai_mapping.md) |

## Documentation and research outputs

- [Full-cohort benchmark and saved fitted models](bundles/full-v1/)
- [Full failure analysis](reports/full_failure_analysis.md) and [scientific figures](reports/figures/full-v1/)
- [Aggregate cohort characteristics and predictor missingness](reports/full-v1-cohort/cohort_profile.md)
- [Arabic user guide](docs/user_guide_ar.md) and [deployment instructions](docs/deployment.md)
- [Evaluation protocol](docs/protocol.md)
- [Data provenance and download integrity](docs/data_provenance.md)
- [Architecture and researcher workflows](docs/architecture.md)
- [Model card](docs/model_card.md)
- [Manuscript draft](docs/manuscript_draft.md) — unfinished research draft, not submitted or publication-ready
- [TRIPOD+AI status mapping](docs/tripod_ai_mapping.md)
- [Verification and known limitations](docs/verification.md)

Core tools run without personal accounts or paid model APIs. A private hosted demonstration is deployed; access has not been made public. Software, dataset and vendored-code licensing are separate; the original project code has no public reuse license selected yet. The official scorer retains its BSD-2-Clause license.

## References

1. Reyna et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, v1.0.0. https://doi.org/10.13026/v64v-d857
2. Reyna et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge*. https://doi.org/10.1097/CCM.0000000000004145
3. [Official evaluation implementation, pinned version](https://github.com/physionetchallenges/evaluation-2019/tree/4edb43177ee7d89dc86ab211705753c2d27b71e4)
4. Collins et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods*. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
