# Clinical AI Reliability Lab

A retrospective research platform for training ICU sepsis early-warning models and measuring how reliability changes with missing observations, delayed data, measurement error and hospital transfer.

**Research question:** Which models provide useful early sepsis warnings, and how do calibration, missed events and alert burden change when data quality or the hospital population changes?

## Status

Implementation is in progress. The [evaluation protocol](docs/protocol.md) is recorded before model fitting. No validated clinical performance or completed benchmark is claimed. The repository is private during development.

## Planned scope

- PhysioNet/CinC 2019: public hospital A (20,336 stays) and B (20,000 stays), hourly observations.
- Three trained families: regularized logistic regression, XGBoost and a causal GRU.
- Patient-separated development, calibration and threshold sets; untouched hospital B evaluation.
- Challenge-task utility and a distinct pre-onset warning analysis. Supplied labels already begin six hours before onset.
- Data-quality stress experiments, component ablations and selective prediction with withheld-event accounting.
- Reproducible Python pipeline, Parquet/DuckDB storage, local MLflow, FastAPI, Docker and a dashboard with synthetic patient replay.
- Model cards, benchmark report and a manuscript draft structured around TRIPOD+AI.

The dashboard will let researchers compare models, replay synthetic trajectories, inspect missing/stale observations and inspect reliability changes. It is **not for clinical care**. An LLM explanation layer is outside the first version.

## Cost and accounts

Development uses locally installed open-source tools. scikit-learn, XGBoost, PyTorch, DuckDB, FastAPI and local MLflow need no personal accounts or API keys. Hosted compute/deployment is separate and may have costs; no paid service is enabled by this repository.

## Sources

- [PhysioNet/CinC 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/)
- [Official evaluation code](https://github.com/physionetchallenges/evaluation-2019)
- [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)

Dataset licensing and software licensing are separate. Do not put real clinical records, credentials or downloaded datasets in Git.
