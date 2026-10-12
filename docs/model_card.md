# Model card: Clinical AI Reliability Lab

**Project owner:** Ibrahim Khalid Baabdullah  
**Card version:** 0.1, 2026-10-12  
**Scope:** family-level research card for logistic regression, XGBoost, and a unidirectional GRU. This document is not an approval for a particular fitted model.

## Evidence status and model identity

The repository implements a retrospective research workflow and a synthetic patient replay. A working application and passing software tests are separate from evidence that a prediction model is clinically reliable.

| Evidence | Status of this card |
|---|---|
| Implemented data, feature, model, and evaluation methods | Described below and in the versioned protocol. |
| Synthetic demonstration | A software demonstration with deliberately enriched event prevalence; no clinical performance claim. |
| Reduced real-data experiment | Completed pilot-v1: 2,400 patients / 91,276 hourly rows, 200 bootstrap replicates. See the verified pilot report and failure analysis; this is not the complete-cohort study. |
| Complete public-cohort benchmark | Pending incorporation of a verified complete-cohort execution report. |
| Prospective or clinical-use validation | Not performed or established by this project. |

For any fitted artifact, identify it using its run directory, `run_manifest.json`, `selection.json`, saved configuration, source-tree hash, split manifest, and dependency record. Family names alone do not identify a model version. The authoritative run report must distinguish synthetic, reduced-cohort, and complete-cohort evidence. Populate artifact-specific results and failure analyses from that report before presenting this as a completed model evaluation.

## Intended users, population, and use

Intended users are machine-learning researchers, nursing informatics researchers, and students inspecting a retrospective prediction workflow. The target research context is adult ICU care as represented by the released PhysioNet/CinC 2019 cohorts. The platform compares model families, reproduces a challenge scoring task, and measures sensitivity to controlled observation changes and transfer from hospital A to hospital B.

The replay interface uses synthetic patient timelines. It supports inspecting scores, recorded observation availability, and mathematical feature contributions where implemented. It is not connected to a live clinical record system.

The models must not be used for patient monitoring, diagnosis, treatment recommendations, triage, discharge decisions, or reassurance that sepsis is absent. Pediatric, non-ICU, community, and present-day hospital populations have not been established as appropriate applications. No regulatory clearance or improvement in patient outcomes is claimed.

## What the score means

The training target is the supplied hourly `SepsisLabel`. It becomes positive six hours before the source reference onset and remains positive afterward. The calibrated score estimates that label under the fitted research model. It is **not** an independently validated probability of a first sepsis event in the next six hours among patients who are currently event-free.

The challenge-task analysis includes post-onset hours. A separate operational analysis evaluates alerts before inferred onset, using the explicitly defined warning window and exclusions in [the protocol](protocol.md). The official utility is a challenge-specific scoring function; its normalization does not convert it into measured clinical benefit.

## Data and inputs

The intended real-data source is PhysioNet/CinC 2019 version 1.0.0. Hospital A supplies development and internal test records; hospital B is reserved for one external transport evaluation. Refer to [data provenance](data_provenance.md) for source counts, acquisition, checksums, and licensing notices. Released records do not permit verification of person-level linkage across hospitals or repeat admissions.

Inputs comprise the 40 supplied variables: eight vital-sign variables, 26 laboratory variables, and six demographic/administrative variables, including elapsed ICU time. Identifiers, hospital, and outcome are not predictors. Values are retained in the source representation; no unit conversion or clinical outlier correction is applied automatically. The source's age value 100 denotes the oldest age category, not an exact age. Recorded `Gender` coding is not interpreted as self-described gender identity.

The ingest path checks field count and order, numeric values, binary nonreverting labels, patient-hour uniqueness, and hourly ICU-time progression. Invalid records stop the run with an error; they are not silently repaired or excluded. These structural checks cannot establish physiological plausibility or correct source timestamps.

## Features and preprocessing

| Component | Implemented behavior |
|---|---|
| Current values | Carry the last recorded value forward within a patient, including current observations. Never carry backward or between patients. |
| Six-hour mean | Mean of observed dynamic values in the current and five preceding hourly rows. |
| Six-hour trend | Difference between current and five-hours-earlier carried values, divided by five; undefined endpoints remain missing. |
| Current absence | One missingness indicator for each dynamic variable. |
| Time since recording | Hours since the last available observation, capped at 48; 48 also represents never observed. |
| Remaining missing features | Training-fitted median imputation; features absent throughout training retain a zero-filled column. |
| Scaling | Training-fitted standardization for logistic regression and GRU; no standardization for XGBoost. |

The complete representation contains 176 features. The cap applies to the time-since indicator, not to forward-filled values: an old measurement can remain carried for more than 48 hours. An externally observed variable that was entirely absent during training can behave unpredictably. Missingness can encode site-specific measurement practices rather than patient physiology.

## Development and selection

Patients in A are stratified by ever-positive label status into train/tune/calibrate/threshold/test partitions of approximately 55/15/10/10/10%, with deterministic integer allocation. Every B patient is external. The split seed is recorded; insufficient class counts cause a failure. Preprocessing is fitted on training records only. Hyperparameters are selected by tune-set average precision; a sigmoid calibrator is fitted on calibration patients; thresholds are selected by official utility on threshold patients. The final family is chosen using the threshold partition, with lower-complexity tie breaking. Test and external data do not determine these choices.

Logistic regression uses regularization and class weighting; XGBoost uses histogram trees; the GRU uses causal patient sequences, weighted binary loss, and padding masks. The actual candidate grid and fitted parameters belong in the saved run configuration and model-development report. Reduced demonstration settings are not interchangeable with the full benchmark configuration.

## Evaluation and reliability experiments

Required reporting includes average precision, AUROC, Brier score, fixed-bin calibration error, calibration bins, normalized challenge utility, eligible pre-onset event sensitivity, warning lead time, and false alert episodes per eligible non-warning patient-day. Counts of censored or otherwise ineligible events accompany the pre-onset analysis. Patient-cluster bootstrap intervals describe uncertainty conditional on the fitted model and split; they do not measure all model-development uncertainty.

The reliability analysis freezes models before hospital B evaluation and simulates observation deletion, arrival delay, and measurement noise at declared severities. Initial feature ablations retrain logistic regression; a separate analysis removes calibration for the selected family. Selective prediction uses distance from the operational threshold as a heuristic. Coverage and missed/withheld events must accompany retained-case performance; withholding has not been shown to make care safer.

The dataset lacks actual specimen and result-availability timestamps. A delay experiment shifts recorded hourly availability; it cannot reconstruct true laboratory turnaround time. Observation age in the application describes recorded availability, not necessarily physiological age.

Subgroup results use available age categories and recorded Gender codes, with unknown values separated and small groups suppressed. These limited comparisons cannot demonstrate fairness across the populations affected by a clinical system.

## Known limitations and failure modes

- Hospital B represents one transfer, not general transportability. Clinical practice, prevalence, devices, and measurement patterns can differ elsewhere or over time.
- Retrospective labels and reconstructed onset are not prospective adjudication. First-row-positive stays have left-censored onset and are excluded from the separate warning analysis.
- Longer stays contribute more training and evaluation hours. Patient bootstrap respects within-stay clustering but does not remove hour-weighting or verify independent people across records.
- Unit/source errors and physiological outliers are not automatically corrected. Artificial Gaussian noise is a controlled statistical stressor, not an estimate of real errors.
- Calibration can deteriorate after population shift, even when discrimination changes little. High classification scores can coexist with excessive alert burden.
- Feature contributions explain uncalibrated mathematical model scores; they do not show causes or recommended interventions. A validated GRU attribution is not implemented.
- Synthetic prevalence and physiology are deliberately constructed. Neither synthetic scores nor synthetic hospital differences estimate clinical performance.
- Reduced cohorts and small event counts can yield unstable thresholds and wide or undefined intervals. Report these outcomes rather than substituting zero.
- Clinical workflow integration, human response, usability, downstream harms, and prospective benefit remain untested.

## Maintenance and release requirements

No automatic retraining or hospital-B recalibration is part of the primary protocol. Any update requires a new version, a declared protocol amendment, independent evaluation, and an updated card. Before a research release, link the exact model artifacts and verified report, record known failures, and complete the manuscript's pending declarations. Load serialized Python model artifacts only from trusted local runs. Keep raw clinical records outside the repository and the public replay.

## References

- [PhysioNet/CinC 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/), DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).
- [Original challenge publication](https://doi.org/10.1097/CCM.0000000000004145).
- [Versioned project protocol](protocol.md), [data provenance](data_provenance.md), and [TRIPOD+AI reporting map](tripod_ai_mapping.md).

## Executed pilot evidence

The [pilot report](../bundles/pilot-v1/report.md) and [failure analysis](../reports/pilot_failure_analysis.md) document limited external event detection. Development-selected XGBoost detected 10/54 eligible B events (18.5%) and missed 44. This preliminary model is not suitable for clinical use. Saved model/configuration/manifest files are under `bundles/pilot-v1`; the synthetic bundle is separate.
