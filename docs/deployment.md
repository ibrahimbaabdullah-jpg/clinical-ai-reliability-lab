# Local execution and hosted deployment

Two runtimes share the dashboard: a local FastAPI application executes the Python models; the hosted edition executes exported fitted-model weights in the browser. Both replay **generated synthetic patients only**. Reports preserve the source and cohort status of their original experiment.

An owner-private version has been deployed at:

<https://clinical-ai-reliability-lab.ibrahimbaabdullah.chatgpt.site/>

Sites project: `appgprj_6acd164ff5488191a79f697ed90c3dd5`. The same URL can serve successive releases. Its loaded `report.json`, browser manifest, and deployed revision identify the evidence actually being displayed; a working URL alone does not prove that the full-cohort bundle has been promoted.

The release workflow below uses `bundles/full-v1`. Commands targeting that directory require the completed report, model checkpoints, and manifests. An earlier checkout can substitute `bundles/pilot-v1` to inspect the preserved preliminary experiment. Neither bundle changes the synthetic-only replay restriction.

## Local Python dashboard

Run from the repository root. Python 3.12 is tested; the package supports Python 3.11–3.13.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Without `CLINICAL_LAB_RUN`, the first request trains and caches a small synthetic logistic demonstration. No account, clinical-data download, or API key is needed. On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell instead of the `source` command.

For the tested Linux CPU environment, constrain the dependency closure with `-c requirements-linux-py312.txt`. Loading or training the GRU also needs PyTorch:

```bash
python -m pip install -c requirements-linux-py312.txt -e '.[dev]'
python -m pip install -c requirements-linux-py312.txt torch --index-url https://download.pytorch.org/whl/cpu
CLINICAL_LAB_RUN="$PWD/bundles/full-v1" python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

`requirements-tested.txt` records direct versions; `requirements-linux-py312.txt` also records their installed transitive dependency closure for Linux x86_64 / Python 3.12, including the CPU PyTorch wheel. These are constraints snapshots, not platform-independent or wheel-hash lockfiles. The synthetic logistic dashboard can run independently of PyTorch on an Intel Mac. The current GRU environment has not been validated on Intel macOS; use the compatible Linux CPU environment or Docker for all three families.

Use a **trusted** run directory containing `report.json`, `<family>.joblib`, and `training_noise_scales.json`. Joblib loading executes Python serialization; never load an uploaded or untrusted model. Stop the prior server before restarting with a different run. The app caches its loaded experiment.

Endpoints: `GET /api/status`, `GET /api/patients`, `GET /api/report`, `POST /api/replay`, and `GET /openapi.json`. Swagger/ReDoc are disabled. CORS is not enabled. The example binds to localhost; authentication and public-service operation of this Python API are separate work.

## Train a new demonstration

With PyTorch installed:

```bash
python -m clinical_lab.cli run --synthetic --config configs/demo.yaml --output artifacts/my-demo
CLINICAL_LAB_RUN="$PWD/artifacts/my-demo" python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

Choose a new output directory for each completed experiment. The CLI refuses to overwrite a completed report. See the README and [protocol](protocol.md) for real-data acquisition and complete-cohort training.

## Promote the completed full-cohort bundle

The full dataset contains **40,336 ICU stays and 1,552,210 patient-hours**: 20,336 hospital A stays and 20,000 hospital B stays. Training and development-family selection precede held-out evaluation. Completed evaluation and saved reports are required before exporting a release; fitted checkpoint files alone do not establish completion.

Keep the preliminary `bundles/pilot-v1` and synthetic `bundles/demo-v1` evidence separate. Assemble `bundles/full-v1` from the completed full run's aggregate reports, trusted fitted models, configuration, manifests, and training-only noise scales. Do not add clinical records or patient-level predictions to the release bundle.

Check the release report before loading or exporting it:

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path('bundles/full-v1')
report = json.loads((run / 'report.json').read_text())
metadata = report['metadata']
assert metadata['source'] == 'physionet-2019'
assert metadata['complete_public_cohort'] is True
assert metadata['patients'] == 40336
assert metadata['patient_hours'] == 1552210
assert metadata['patients_by_hospital'] == {'A': 20336, 'B': 20000}
for family in ('logistic', 'xgboost', 'gru'):
    assert (run / f'{family}.joblib').is_file()
    assert 'test' in report['models'][family]
    assert 'external' in report['models'][family]
assert (run / 'training_noise_scales.json').is_file()
print('Completed full-cohort bundle metadata verified')
PY
```

These checks establish basic release consistency; retain the run's file hashes, frozen selection record, protocol, and executed verification evidence for provenance.

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
  -v "$PWD/bundles/full-v1:/app/run:ro" \
  -e CLINICAL_LAB_RUN=/app/run \
  clinical-ai-reliability-lab:deep
```

The container runs as a non-root user. Its health check initializes `/api/status`, including a mounted run. `.dockerignore` excludes datasets, bundles, credentials, and Git state from the build context. These commands describe the supplied configuration; consult [verification](verification.md) for executed Docker validation.

## Export the hosted browser edition

Export from the same compatible Python environment that can load the trusted model artifacts:

```bash
python scripts/export_browser_bundle.py \
  --run bundles/full-v1 \
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
node tests/test_web_contract.cjs bundles/full-v1/report.json
node tests/test_site_runtime.cjs artifacts/site/data
```

Numerical export tests compare fitted Python predictions and browser predictions; runtime tests exercise all three families, four replay modes, causal prefixes, provenance, validation, and JSON/CSV downloads. Node tests do not establish visual layout or accessibility conformance.

After deployment, verify actual hosted asset responses and match their report/manifest to the intended release. Check owner access, the evidence banner, model availability, replay controls, and report downloads when supported browser inspection becomes available. Record the deployed revision and any unavailable checks in `docs/verification.md`.

**Visual QA remains unverified in this execution:** the Sites-required `control-browser` skill was unavailable. Node rendering contracts, numerical model parity, successful asset checks, and a deployment response are not desktop/mobile visual or accessibility QA. Keep those distinctions in release notes until a supported browser inspection is performed.
