# Evaluation protocol v0.1

Recorded before model fitting on 2026-10-12. This is a versioned development protocol, not a registered clinical study. Amendments must be dated and identify whether outcomes had already been inspected.

## Research question and population

Which hourly ICU sepsis models retain discrimination, calibration and useful pre-onset warning performance under hospital transfer and degraded observations? Retrospective research only. No treatment decisions or patient monitoring. The public PhysioNet/CinC 2019 A/B cohorts are the target data; generated trajectories are software demonstrations only.

## Endpoints and time

1. Reproduce the supplied hourly `SepsisLabel` task and the official normalized utility. The labels already become positive six hours before onset and stay positive. Do not shift them again. Include all recorded hours in this challenge-task analysis. AUROC and average precision (reported as AP, not trapezoidal PR area), Brier score and 10 fixed-bin ECE complement utility.
2. Separately assess warning events in the six hours immediately before inferred onset, excluding post-onset observations. Infer onset as first positive row + 6 only when a zero-to-one transition is observed. Exclude first-row-positive stays from this analysis because onset is left-censored. Exclude onset beyond recorded follow-up from event sensitivity and report these counts. Never substitute post-onset detection for early warning.
3. Alert episodes use a six-hour refractory period: the first threshold crossing creates an alert; sustained high scores can create another episode after six hours. False alert episodes outside the defined warning window divided by eligible non-warning patient-hours / 24 provide the pre-onset false-alert rate. State exact denominators. Challenge hourly predictions do not use this refractory period.

## Frozen split and model selection

Seed 2019. Split patients, never rows. Hospital A: train 55%, tune 15%, calibrate 10%, threshold 10%, internal test 10%, stratified by ever-positive patient status with deterministic integer allocation. Hospital B is external test only. Require both classes in development partitions; fail explicitly when samples are insufficient. Write patient manifests and dataset hash before fitting. Patient identity is available only at the provided per-file level; cross-site identity linkage cannot be verified.

Fit imputation, scaling and error-injection scales on training patients only. Select hyperparameters by tune-set average precision. Fit sigmoid calibration on the calibration partition. Choose the alert threshold by official utility on the threshold partition, including an all-negative candidate. Select the final family by threshold-partition utility; resolve ties in favor of lower complexity (logistic, XGBoost, GRU). Report test results for every family; do not select the winner on A-test or B. This is a single split study; multiple seeds are an extension, not an unreported search.

Families: regularized logistic regression, XGBoost histogram trees, unidirectional GRU. All predictors use available current/past data only. No backward filling, centered windows, final stay length, label, hospital or patient identifier as predictors. Include current/forward-filled values, six-hour trends, missingness and elapsed hours since measurement. GRU sequences are causal, grouped by patient, with padded steps excluded from loss.

## Reliability experiments

- Hospital transfer: freeze preprocessing, model, calibrator and threshold before external evaluation.
- Deletion: independently remove 10%, 30%, 50% of observed dynamic values.
- Delay: shift dynamic observations by 1, 3, 6 hourly rows within patient, never between stays. This approximates availability delay; the dataset does not contain actual specimen/result timestamps.
- Measurement error: zero-mean Gaussian noise with SD 0.1, 0.25, 0.5 times each training-set variable SD. Preserve existing missingness, demographics and labels. These are controlled statistical stressors, not estimated real-world error distributions.
- Feature ablations: retrain the logistic baseline without trends, without missingness/age-of-measurement, and using values only; recalibrate and choose thresholds on the same assigned development partitions. This initial component study is limited to logistic regression. Also evaluate removal of calibration for the development-selected family with its own development-selected threshold.
- Selective prediction: evaluate a heuristic based on distance from the operational threshold, with cutoffs frozen using the threshold partition. Report retained hourly coverage, retained errors, withheld positive hours/events and overall event sensitivity treating withholding as no alert. Withholding is not assumed to improve safety and human-review burden remains unknown.

## Statistical reporting

Patient-cluster percentile bootstrap (seeded; production setting 1000 replicates, fast software smoke test fewer and explicitly labelled). Report 95% intervals and valid replicate counts; undefined class-dependent estimates are null, never zero. Bootstrap resamples patients with all their hours and gives repeated sampled patients distinct cluster IDs. Primary outcomes: AP, normalized utility, Brier score, pre-onset event sensitivity and false alerts per patient-day. Comparisons using separate marginal CIs are descriptive; they are not significance tests.

Report recorded Gender coding and age (<65, >=65) strata with missing/unknown separately; do not infer gender identity. Suppress estimates when fewer than 20 patients or 5 events are present, while reporting denominators. Subgroups cannot establish fairness without additional variables and evidence. Do not tune subgroups or stress severities using external results.

## Provenance and reporting

Retain downloaded archives/checksums and source notices locally; do not commit clinical records. The PhysioNet landing-page license and bundled LICENSE.txt require checking together before redistribution. Record dependency versions, configuration, data hash, split hash, source code revision, model parameters and experiment time. MLflow is local and optional; no paid account or LLM API is required.

Prepare a manuscript using TRIPOD+AI reporting domains with explicit incomplete items. Do not claim prospective validation, clinical benefit, regulatory clearance, TRIPOD+AI compliance or a finished benchmark before evidence exists. Synthetic, reduced-cohort and complete-cohort runs must be distinguishable in every report.

## Sources

Implementation clarification recorded before the first successful synthetic fit and before any real-data model fit: component ablations use logistic regression; the calibration ablation uses the development-selected family. Initial live-data verification may use a deterministic uniform sample of 1,200 patients per hospital (seed 2019), with its own frozen split manifest. This is explicitly preliminary and must not be presented as the complete cohort benchmark.

- https://physionet.org/content/challenge-2019/1.0.0/
- https://github.com/physionetchallenges/evaluation-2019
- https://www.bmj.com/content/385/bmj-2023-078378
