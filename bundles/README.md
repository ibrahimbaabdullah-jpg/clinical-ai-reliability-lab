# Verified experiment bundles

These are real saved fits, not hand-authored example scores.

- `demo-v1`: three models trained on generated data; 360 synthetic patients. Not clinical evidence.
- `pilot-v1`: three models trained/evaluated using a uniform 2,400-patient PhysioNet sample. Preliminary retrospective evidence; not the full public dataset or a final model.

Each bundle includes model/calibrator/threshold joblib objects, aggregate `report.json`, protocol/configuration snapshots, run/environment manifests, selected family and train-only noise scales. `SHA256SUMS.json` records every bundled file's local integrity checksum. Raw observations and patient-level predictions are excluded. Full local run outputs also contain the deterministic split manifest, reconstructible from the downloader seed and run configuration.

Load only these trusted project artifacts; joblib files can execute Python when deserialized. Install the tested dependencies and PyTorch before loading GRU bundles. Dependency compatibility is necessary; hashes alone are not signatures.

```bash
CLINICAL_LAB_RUN=bundles/pilot-v1 python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

The report then displays actual pilot results, while patient replay remains explicitly synthetic. Scores on synthetic patients are not evidence of clinical performance.
