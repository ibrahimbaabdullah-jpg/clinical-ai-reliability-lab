# Executed model bundles

These directories preserve **trusted fitted model artifacts and aggregate results** from distinct experiments. They contain no raw clinical observations, patient-level predictions or split identifiers.

| Bundle | Evidence and intended interpretation |
|---|---|
| `full-v1` | Complete public PhysioNet cohort: 40,336 records / 1,552,210 hours, full prespecified tuning, 1,000 patient-bootstrap replicates. Main retrospective research release. |
| `pilot-v1` | Uniform 2,400-record PhysioNet sample, smaller settings and 200 bootstrap replicates. Preliminary feasibility evidence; overlaps the full cohort. |
| `demo-v1` | 360 generated patients / 15,168 hours and 50 bootstrap replicates. Synthetic software demonstration only. |

The full run includes three trained model families, six primary held-out evaluations, 18 selected-family stress conditions, four component ablations and selective prediction. `selection.json` identifies the family chosen on hospital A before test scoring. `report.json` preserves exact metrics, counts, intervals, tuning results and provenance; `report.md` summarizes them. `SHA256SUMS.json` verifies the bundled file bytes.

Start the completed benchmark's Python dashboard after installing the tested dependencies, including CPU PyTorch for GRU:

```bash
CLINICAL_LAB_RUN=bundles/full-v1 python -m uvicorn clinical_lab.api:app --host 127.0.0.1 --port 8000
```

The report shows the actual retrospective results. **Patient replay always uses synthetic timelines**; a real-data model applied to them is an out-of-distribution illustration. The [private hosted edition](https://clinical-ai-reliability-lab.ibrahimbaabdullah.chatgpt.site) directly executes exported model parameters in the browser.

Only load these joblib files from a trusted checkout: Python deserialization can execute code. Data license notices and original code reuse permissions remain separate. Read the [model card](../docs/model_card.md), [failure analysis](../reports/full_failure_analysis.md), and [verification record](../docs/verification.md) before interpreting any performance result.
