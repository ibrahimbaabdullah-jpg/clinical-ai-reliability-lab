# Executed research evidence

## Complete public-cohort benchmark

- [Full machine-generated report](../bundles/full-v1/report.md) and [JSON evidence](../bundles/full-v1/report.json): all 40,336 records, 1,552,210 hours, three trained models and 1,000 patient-bootstrap replicates.
- [Failure analysis](full_failure_analysis.md): hospital transfer, missed events, alert burden, stress conditions, ablations, calibration and selective withholding.
- [Cohort profile](full-v1-cohort/cohort_profile.md) and [source-backed JSON](full-v1-cohort/cohort_profile.json): population, recorded stay duration, source age/Gender, all 40 predictors' missingness and distributions.
- [Scientific figures](figures/full-v1/): PNG/SVG pairs for primary performance, calibration, observation stress and selective withholding. `figure-data.json` preserves exact plotted estimates; `figure-manifest.json` links the figure bytes to the source-report SHA256.
- [Execution telemetry](full-v1-telemetry.json): observed wall time, peak process memory, CPU time and exit status.
- [Trusted fitted models and manifests](../bundles/full-v1/) and [manuscript draft](../docs/manuscript_draft.md).

The main study remains retrospective, with one external hospital. The full external set overlaps the previously inspected pilot. Model selection uses hospital A only; no hospital B retuning or clinical-use claim is made.

## Earlier experiments retained separately

- [Real-data pilot](pilot.md): uniform 2,400-record preliminary analysis; [pilot failure analysis](pilot_failure_analysis.md).
- [Synthetic execution](synthetic.md): software demonstration only.

References inside copied historical summaries to run-local files identify their original experiment directories. Selected aggregate reports, configurations and weights are preserved in the corresponding [bundle](../bundles/); raw data, split identifiers and patient-level predictions remain excluded from Git and the hosted Site.
