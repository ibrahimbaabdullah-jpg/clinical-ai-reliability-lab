# Local execution and hosted deployment

Two runtimes share the dashboard: a local FastAPI application executes the Python models; the hosted edition executes exported fitted-model weights in the browser. Both replay **generated synthetic patients only**. Reports preserve the source and cohort status of their original experiment.

The registered owner-private deployment target is:

<https://clinical-ai-reliability-lab.ibrahimbaabdullah.chatgpt.site/>

Registration does not establish successful deployment or access for other users. Confirm the deployed revision and visibility before distributing this link. This guide does not assert that a full-cohort run or hosted release has completed.

## Local Python dashboard

Run from the repository root. Python 3.12 is tested; the package supports Python 3.11–3.13.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Without `CLINICAL_LAB_RUN`, the first request trains and caches a small synthetic logistic demonstration. No account, clinical-data download, or API key is needed. On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell instead of the `source` command.

For the tested Linux CPU environment, constrain the direct dependencies with `-c requirements-tested.txt`. Loading or training the GRU also needs PyTorch:

```bash
python -m pip install -c requirements-tested.txt -e '.[dev]'
python -m pip install -c requirements-tested.txt torch --index-url https://download.pytorch.org/whl/cpu
CLINICAL_LAB_RUN="$PWD/bundles/pilot-v1" python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

`requirements-tested.txt` describes Linux x86_64 / Python 3.12 direct versions, including the CPU PyTorch wheel; it is not a platform-independent or transitive lockfile. The synthetic logistic dashboard can run independently of PyTorch on an Intel Mac. The current GRU environment has not been validated on Intel macOS; use the compatible Linux CPU environment or Docker for all three families.

Use a **trusted** run directory containing `report.json`, `<family>.joblib`, and `training_noise_scales.json`. Joblib loading executes Python serialization; never load an uploaded or untrusted model. Stop the prior server before restarting with a different run. The app caches its loaded experiment.

Endpoints: `GET /api/status`, `GET /api/patients`, `GET /api/report`, `POST /api/replay`, and `GET /openapi.json`. Swagger/ReDoc are disabled. CORS is not enabled. The example binds to localhost; authentication and public-service operation of this Python API are separate work.

## Train a new demonstration

With PyTorch installed:

```bash
python -m clinical_lab.cli run --synthetic --config configs/demo.yaml --output artifacts/my-demo
CLINICAL_LAB_RUN="$PWD/artifacts/my-demo" python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Choose a new output directory for each completed experiment. The CLI refuses to overwrite a completed report. See the README and [protocol](protocol.md) for real-data acquisition and complete-cohort training.

`bundles/full-v1` is the intended location for an eventual completed full-cohort release. Use it only after its models, report, and manifests exist and its evidence status is verified; directory naming alone is not proof of complete-cohort evaluation.

## Docker

Default image: local synthetic logistic dashboard, without PyTorch or MLflow.

```bash
docker build -t clinical-ai-reliability-lab .
docker run --rm -p 127.0.0.1:8000:8000 clinical-ai-reliability-lab
```

To load a three-family run, include CPU PyTorch and mount the trusted artifacts read-only:

```bash
docker build --build-arg WITH_DEEP=1 -t clinical-ai-reliability-lab:deep .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/bundles/pilot-v1:/app/run:ro" \
  -e CLINICAL_LAB_RUN=/app/run \
  clinical-ai-reliability-lab:deep
```

The container runs as a non-root user. Its health check initializes `/api/status`, including a mounted run. `.dockerignore` excludes datasets, bundles, credentials, and Git state from the build context. These commands describe the supplied configuration; consult [verification](verification.md) for executed Docker validation.

## Export the hosted browser edition

Export from the same compatible Python environment that can load the trusted model artifacts:

```bash
python scripts/export_browser_bundle.py \
  --run bundles/pilot-v1 \
  --output artifacts/site/data \
  --synthetic-patients 120 \
  --seed 2019
```

This writes:

| Asset | Contents |
|---|---|
| `models.json` | Actual fitted logistic, XGBoost, and GRU parameters, preprocessing, calibration, and thresholds available in the run |
| `synthetic-patients.json` | Generated observations only; no clinical records or outcome labels |
| `training-noise-scales.json` | Frozen scales derived from training measurements |
| `report.json` | Recorded aggregate experiment results and provenance |
| `browser-manifest.json` | Model/source hashes, file hashes, runtime format, and attribution availability |

Do not publish raw clinical data, patient-level predictions, or Python joblib artifacts with the website. The export intentionally uses portable JSON and synthetic trajectories.

Prepare static files and opt into browser transport:

```bash
mkdir -p artifacts/site/static
cp clinical_lab/web/style.css clinical_lab/web/app.js clinical_lab/web/site-adapter.js clinical_lab/web/inference.js artifacts/site/static/
python - <<'PY'
from pathlib import Path
source = Path('clinical_lab/web/index.html').read_text()
source = source.replace('<html lang="en">', '<html lang="en" data-runtime="static" data-base="/data">')
source = source.replace('<script src="/static/app.js" defer></script>', '<script type="module" src="/static/site-adapter.js"></script>')
Path('artifacts/site/index.html').write_text(source)
PY
```

The static root must serve `/index.html`, `/static/*`, and `/data/*` over HTTPS. Deploy `artifacts/site` through the authorized hosting workflow. Keep the owner-private visibility until a broader release is authorized. An optional local HTTP preview can serve this directory; modules must be served over HTTP, not opened as `file://` URLs.

`site-adapter.js` loads the exported assets, installs the dashboard transport, and imports `app.js`. Inference is calculated on each requested synthetic timeline prefix from the fitted weights. The original local index continues using FastAPI; do not replace it with the exported index.

Hosted logistic contributions are exact contributions before calibration. Portable XGBoost and GRU attributions are unavailable and disclosed; local Python additionally provides native XGBoost TreeSHAP. Browser perturbations use seeded Mulberry32/Box–Muller draws: severity definitions and training-only noise scales match the protocol, but individual random realizations differ from NumPy benchmark experiments.

## Verify the release

```bash
python -m pytest
node tests/test_web_contract.cjs bundles/pilot-v1/report.json
node tests/test_site_runtime.cjs artifacts/site/data
```

Numerical export tests compare fitted Python predictions and browser predictions; runtime tests exercise all three families, four replay modes, causal prefixes, provenance, validation, and JSON/CSV downloads. Node tests do not establish visual layout or accessibility conformance.

After deployment, verify the actual hosted asset responses, evidence banner, model availability, replay controls, report downloads, and owner access. Record the deployed revision and any unavailable checks in `docs/verification.md`. A browser inspection that could not be performed must remain marked pending; do not substitute a successful deployment response for visual QA.
