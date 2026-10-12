# Execution notes — 2026-10-12

The study protocol and full-cohort configuration were committed before the first model fit. The reduced pilot used 1,200 uniformly sampled filenames from each hospital. Its results were inspected for feasibility and failure analysis before the full-cohort run.

The complete cohort therefore overlaps the inspected pilot, including 1,200 hospital B records. Hospital B remains excluded from fitting, preprocessing, hyperparameter tuning, calibration, threshold selection and model-family selection. The full-cohort protocol and configuration were retained unchanged after the pilot. This is a retrospective external-hospital experiment, **not wholly new, never-inspected confirmatory validation**. No independent prospective validation is claimed.

Implementation changes before the full run preserve the study definitions: bounded multi-file parsing with exact value/dtype equivalence, cached score ordering in patient-cluster bootstrap, releasing unused feature matrices, and packaging the frozen protocol inside installed wheels. Patient-level resampling, its seed and number of replicates, model grids, outcomes, stress severities and selection criteria are unchanged. Run manifests record source, data, configuration, protocol and split hashes.

The full download uses the official public PhysioNet S3 mirror. The 2,400 already verified pilot files are reused byte-for-byte and retain their original official source URLs; newly downloaded files record their S3 source URLs. The manifest enumerates every expected file and SHA256 digest.

The browser engine exports fitted parameters directly. It does not retrain, distill or replace the Python models. Numeric parity and frozen-threshold decisions are tested against Python on generated trajectories. Browser perturbations use a deterministic JavaScript random-number generator, so their individual deletion masks/noise draws differ from the recorded NumPy benchmark. The distribution and severity definitions agree; aggregate benchmark results always come from the recorded Python experiment.

Browser visual inspection is unavailable in this environment because the required browser-control capability is absent. Numerical, DOM-contract, installed-wheel, HTTP and actual Docker checks provide the documented software evidence; they do not establish that desktop/mobile appearance has been visually inspected.

The original protocol is preserved without retrospective rewriting. Any future outcome, alert policy, model selection or stress-definition changes require a dated amendment and a separately identified experiment.
