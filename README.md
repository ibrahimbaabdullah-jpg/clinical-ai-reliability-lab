# Clinical AI Reliability Lab

**A reproducible research platform for ICU sepsis models, hospital transfer and data-quality stress testing.**

Which models provide useful early sepsis warnings, and how do calibration, missed events and alert burden change when observations or the hospital population change?

This project trains real machine-learning models; an LLM API is not required. It combines Python data engineering, classical and temporal models, statistical evaluation, model interpretation, an inference API and a research dashboard.

## Evidence status

Version 0.1 is a working research implementation. All three model families have been trained in a synthetic end-to-end experiment. A deterministic **2,400-patient PhysioNet sample** is used for preliminary integration/feasibility evaluation. See [verification](docs/verification.md) for what actually ran and [aggregate reports](reports/) for results.

**The complete 40,336-patient benchmark, publication manuscript results, prospective validation and public hosted deployment are not yet established.** Synthetic examples and a reduced-cohort experiment cannot substantiate clinical performance. The repository remains private during development.

## What the application does

| Workspace | Researcher actions and benefit |
|---|---|
| Overview | Compare fitted models on internal and external cohorts, with the dataset scope visible. |
| Patient replay | Select a synthetic patient and observed hour, inspect scores, missing/stale observations and logistic/XGBoost contributions. |
| Reliability | Examine calibration, bootstrap intervals, hospital transfer, stress tests and selective-prediction results. |
| Methods | Inspect the outcome definition, split design, alert policy and limitations. |

The replay endpoint executes the fitted model on a timeline prefix. It always uses synthetic patient records. A real-data model applied to synthetic trajectories is explicitly an out-of-distribution demonstration. No patient uploads or treatment recommendations are supported.

## Quick start — local dashboard

Python 3.12 is the tested version (project range: 3.11–3.13). Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,tracking]'
python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The first request trains a small deterministic synthetic logistic baseline locally. It requires no account, API key or downloaded clinical dataset. **http://127.0.0.1:8000/openapi.json** exposes the API schema. On Windows activate the virtual environment using its Scripts/activate script.

## Train all three families

Install PyTorch for your platform. The verified Linux CPU command is:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
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
python scripts/download_data.py --output data/full
python -m clinical_lab.cli run --data data/full --config configs/benchmark.yaml --output artifacts/full
```

The downloader uses official URLs, bounded concurrency and SHA256 integrity manifests. The CLI verifies recorded file hashes before fitting. Those are local download-integrity checks, not publisher signatures. See [provenance](docs/data_provenance.md): the landing-page and bundled license notices differ, so this repository does not redistribute patient records or assert an unambiguous data license.

The complete configuration uses larger tuning grids and 1,000 patient-bootstrap replicates; it takes substantially more compute than the demo. Full-cohort runtime and peak memory have not yet been measured. A GPU is optional, not required by this CPU implementation.

## Prespecified evaluation

The [protocol](docs/protocol.md) was committed before model fitting. Hospital A is split by patient into **55% train / 15% tune / 10% calibrate / 10% threshold / 10% internal test**, stratified by ever-positive status. Hospital B is external only. Fit preprocessing on train, select hyperparameters on tune AP, fit sigmoid calibration on calibrate, and select thresholds and the final family on threshold-partition challenge utility. Freeze these choices before inspecting test metrics.

**The supplied labels already start six hours before inferred onset and remain positive afterward.** Do not shift them again. The main hourly score is a probability of the supplied challenge label, not a validated six-hour incident-sepsis risk among event-free patients.

- Original task: official challenge utility, AUROC, average precision, Brier score and fixed-bin calibration.
- Separate pre-onset analysis: warnings in `[onset−6, onset)`, six-hour alert refractory period, eligible event sensitivity, missed events, warning time and false alert episodes per patient-day. Left-censored and beyond-followup onset cases are counted and excluded only from this endpoint.
- Reliability: hospital transfer; deletion 10/30/50%; delay 1/3/6 hours; Gaussian errors at 0.1/0.25/0.5 training SD; logistic feature ablations; selected-family calibration ablation.
- Selective prediction: development-derived cutoffs, retained coverage/errors and withheld events, with all-event sensitivity counting withholding as no alert. It is a heuristic experiment, not a safety guarantee.
- Uncertainty and subgroups: patient-cluster confidence intervals; recorded age/Gender strata with small-group suppression. Missing demographics and excluded event counts remain visible.

Every primary estimate has exact denominators and valid bootstrap replicate counts in `report.json`. Reports explicitly distinguish synthetic, preliminary and complete-cohort evidence. Marginal intervals are descriptive, not paired statistical significance tests.

## Model families and tools

| Component | Implementation |
|---|---|
| Interpretable baseline | scikit-learn regularized logistic regression, train-only median imputation and scaling |
| Boosted trees | XGBoost histogram trees with tune-set model selection |
| Temporal learning | PyTorch unidirectional GRU, packed variable-length sequences, masked loss and early stopping |
| Data | Pandas/NumPy, Parquet and DuckDB |
| Tracking | Optional local MLflow with SQLite; no cloud account |
| App | FastAPI and responsive local HTML/CSS/JavaScript dashboard |
| Reproducibility | Configurations, manifests, tests, GitHub Actions workflow and Dockerfile |

`requirements-tested.txt` records the directly tested dependency versions. Each run records its actual environment. The optional LLM explanation layer is outside v0.1.

## Tests and containers

```bash
python -m pytest

# Default synthetic logistic dashboard
docker build -t clinical-ai-reliability-lab .
docker run --rm -p 127.0.0.1:8000:8000 clinical-ai-reliability-lab

# Include GRU loading for a previously trained run
docker build --build-arg WITH_DEEP=1 -t clinical-ai-reliability-lab:deep .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts/demo:/app/run:ro" -e CLINICAL_LAB_RUN=/app/run clinical-ai-reliability-lab:deep
```

Only load trusted model artifacts: joblib deserialization can execute Python. The default API binds locally; authentication and public-service hardening are separate deployment work. Docker configuration is supplied; see verification for whether a container build has actually been tested.

## Documentation and research outputs

- [Evaluation protocol](docs/protocol.md)
- [Data provenance and download integrity](docs/data_provenance.md)
- [Architecture and researcher workflows](docs/architecture.md)
- [Model card](docs/model_card.md)
- [Manuscript draft](docs/manuscript_draft.md) — unfinished research draft, not submitted or publication-ready
- [TRIPOD+AI status mapping](docs/tripod_ai_mapping.md)
- [Verification and known limitations](docs/verification.md)

Core tools run without personal accounts. No paid service, API subscription or public deployment is enabled. Software, dataset and vendored-code licensing are separate; the original project code has no public reuse license selected yet. The official scorer retains its BSD-2-Clause license.

## References

1. Reyna et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, v1.0.0. https://doi.org/10.13026/v64v-d857
2. Reyna et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge*. https://doi.org/10.1097/CCM.0000000000004145
3. [Official evaluation implementation, pinned version](https://github.com/physionetchallenges/evaluation-2019/tree/4edb43177ee7d89dc86ab211705753c2d27b71e4)
4. Collins et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods*. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
