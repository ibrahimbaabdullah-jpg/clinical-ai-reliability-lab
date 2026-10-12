# Model card: Clinical AI Reliability Lab

**Project owner:** Ibrahim Khalid Baabdullah\
**Card version:** 0.2, 2026-10-12\
**Run:** `full-v1`; development-selected family: **XGBoost**.\
**Scope:** fitted logistic regression, XGBoost, and unidirectional GRU research models. This document does not authorize clinical use.

**Execution status:** the full-cohort retrospective benchmark completed successfully. The [final report](../bundles/full-v1/report.json) includes all three model comparisons, 18 stress evaluations, four ablations evaluated on both held-out sets, and selective-prediction analyses. The released report was checked against the completed run. This is a research software release, not evidence of readiness for patient care.

## Evidence status and model identity

The repository implements a retrospective research workflow and a synthetic patient replay. A working application and passing software tests are separate from evidence that a prediction model is clinically reliable.

| Evidence | Status of this card |
|---|---|
| Implemented data, feature, model, and evaluation methods | Described below and in the versioned protocol. |
| Synthetic demonstration | A software demonstration with deliberately enriched event prevalence; no clinical performance claim. |
| Reduced real-data experiment | Completed pilot-v1: 2,400 patients / 91,276 hourly rows, 200 bootstrap replicates. See the verified pilot report and failure analysis; this is not the complete-cohort study. |
| Complete public cohort | Completed full-v1: 40,336 records and 1,552,210 hourly rows; all prescribed primary comparisons, stress conditions, ablations, and selective analyses executed. |
| Prospective or clinical-use validation | Not performed or established by this project. |

The full-run artifact identity is recorded below. The [release bundle](../bundles/full-v1/) contains the completed report, three fitted model bundles, configuration, manifests, dependencies, and checksums. Fitted model bundles include preprocessing, calibration, and the operating threshold. Synthetic and pilot artifacts remain separately identified.

| Identity field | Recorded value |
|---|---|
| Run creation, UTC | 2026-10-12T17:37:52.416100+00:00 |
| Training code revision | `7274308bc66c7d54c3c0481d06a590a68fd2bcc2` |
| Python source-tree SHA256 | `b336c68df9b66c829172fcf8e7a9fcb63e97195afc9477defc02e09e533969f1` |
| Canonical data SHA256 | `709d62e4470467bf7aa7c5579e01538de78717faa5b8a87cfeeb1b7dfe55f05e` |
| Split-manifest SHA256 | `21390fce2a284e16ada169e7b80d9a0a5cc0d1984fa14cbdb41617ec574c0895` |
| Configuration SHA256 | `e5db7e3a5d40a707c5a15ff545388c1680c6ffee0707b7a5c166f1536a76305c` |
| Seed and primary bootstrap | Seed 2019; 1,000 patient-cluster replicates |
| Principal model dependencies | Python 3.12.14; scikit-learn 1.8.0; XGBoost 3.4.1; PyTorch 2.14.1+cpu |
| Compute | Linux x86_64, CPU only; OMP threads 2 and OpenBLAS threads 1 |

Use the accompanying `run_manifest.json`, `selection.json`, configuration, split manifest, dependency record, and final report to identify a release. The source-tree hash describes the fitted run; later documentation or deployment changes do not change its training provenance.

[Execution telemetry](../reports/full-v1-telemetry.json) recorded 2,086.077 seconds of wall time (34.8 minutes) and peak child-process resident memory of 6,857,928 KiB (approximately 6.54 GiB). These are measurements for this run and machine, not guaranteed requirements on another system. No GPU was used.

## Intended users, population, and use

Intended users are machine-learning researchers, nursing informatics researchers, and students inspecting a retrospective prediction workflow. The target research context is ICU care as represented by the released PhysioNet/CinC 2019 cohorts. The platform compares model families, reproduces a challenge scoring task, and measures sensitivity to controlled observation changes and transfer from hospital A to hospital B.

The private hosted research interface uses portable exports of the actual fitted model weights with synthetic patient timelines. It computes scores, displays recorded observation availability, and supports controlled perturbations. The synthetic replay is an out-of-distribution illustration when scored by a real-data-trained model; its scores do not provide synthetic-patient clinical evidence. It is not connected to a live clinical record system.

Browser replay provides logistic coefficient contributions in uncalibrated log-odds. Browser XGBoost and GRU contributions are unavailable; no substitute attribution is presented. The local Python XGBoost implementation supports native TreeSHAP. A validated GRU attribution is unavailable in both runtimes. Browser perturbations use a different deterministic random-number implementation from the NumPy benchmark, so their realized masks/noise are illustrative rather than identical benchmark draws.

The models must not be used for patient monitoring, diagnosis, treatment recommendations, triage, discharge decisions, or reassurance that sepsis is absent. Pediatric, non-ICU, community, and present-day hospital populations have not been established as appropriate applications. No regulatory clearance or improvement in patient outcomes is claimed.

## What the score means

The training target is the supplied hourly `SepsisLabel`. It becomes positive six hours before the source reference onset and remains positive afterward. The calibrated score estimates that label under the fitted research model. It is **not** an independently validated probability of a first sepsis event in the next six hours among patients who are currently event-free.

The challenge-task analysis includes post-onset hours. A separate operational analysis evaluates alerts before inferred onset, using the explicitly defined warning window and exclusions in [the protocol](protocol.md). The official utility is a challenge-specific scoring function; its normalization does not convert it into measured clinical benefit.

## Data and inputs

The fitted full-run source is PhysioNet/CinC 2019 version 1.0.0: 20,336 A records and 20,000 B records. Hospital A supplies development and internal test records; B provides one external transport evaluation. Refer to [data provenance](data_provenance.md) for acquisition, checksums, and licensing notices. Released records do not permit verification of person-level linkage across hospitals or repeat admissions.

**Prior external-data exposure:** 1,200 of the 20,000 B records (6%) were already included in the inspected pilot evaluation. All full-run fitting, hyperparameter, calibration, threshold, and family choices use A partitions, but the complete B cohort cannot be described as wholly unseen by the project team. The full run is not an independent replication of the pilot on entirely new external records.

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

Patients in A are stratified by ever-positive label status into train/tune/calibrate/threshold/test partitions of approximately 55/15/10/10/10%, with deterministic integer allocation. Every B patient is external in this run. Insufficient class counts cause a failure. Preprocessing is fitted on training records only. Hyperparameters are selected by tune-set average precision; a sigmoid calibrator is fitted on calibration patients; thresholds are selected by official utility on threshold patients. The final family is chosen using the threshold partition, with lower-complexity tie breaking. The full run does not use A-test or B values to determine these choices; prior pilot B exposure is disclosed above.

| Partition | Records | Hourly rows |
|---|---:|---:|
| A train | 11,185 | 435,883 |
| A tune | 3,050 | 118,820 |
| A calibrate | 2,034 | 79,280 |
| A threshold selection | 2,034 | 77,827 |
| A internal test | 2,033 | 78,405 |
| B external | 20,000 | 761,995 |

The fitted logistic model uses `C=0.01` and balanced class weights. The fitted XGBoost model has 200 depth-3 trees, learning rate 0.05, and histogram construction. The selected GRU has 32 hidden units and retains epoch 9, with causal patient sequences, weighted binary loss, and padding masks. All three full-run operating thresholds are 0.02 after separate sigmoid calibration. This threshold concerns the supplied label score; it is not a validated clinical risk threshold.

XGBoost was selected before full-run held-out evaluation because its threshold-partition utility was 0.373260, compared with 0.297102 for logistic regression and 0.330094 for GRU. The candidate grid, fitted parameters, calibrators, and training history are retained with the run. Reduced demonstration settings are not interchangeable with these fitted models.

## Verified primary performance of selected XGBoost

The following values are from the completed primary comparisons. Intervals are 95% patient-cluster percentile intervals, with 1,000 valid replicates for every displayed metric. They condition on the fitted model and split; overlapping or nonoverlapping marginal intervals are not a formal paired comparison.

| Metric | A internal test | B external |
|---|---:|---:|
| Average precision | 0.1622 [0.1284, 0.2033] | 0.0797 [0.0705, 0.0902] |
| AUROC | 0.8247 [0.7973, 0.8522] | 0.7559 [0.7427, 0.7703] |
| Normalized challenge utility | 0.4087 [0.3413, 0.4699] | 0.2744 [0.2504, 0.2993] |
| Brier score | 0.01984 [0.01711, 0.02241] | 0.01367 [0.01297, 0.01441] |
| Eligible pre-onset event sensitivity | 76.88% [70.62%, 83.34%] | 60.49% [57.41%, 63.53%] |
| False alert episodes per eligible non-warning patient-day | 1.1589 [1.0766, 1.2340] | 0.5895 [0.5674, 0.6128] |
| Median lead time among detected events, hours | 3 [3, 4] | 3 [3, 3] |

On B, the selected model detected **548 of 906 eligible events and missed 358**. Its 18,418 false alert episodes occurred over 31,241.6667 eligible non-warning patient-days, giving 0.589533 episodes/day. Of B's 1,142 ever-positive records, 223 were left-censored and 13 had inferred onset beyond follow-up, so these 236 records were excluded from the separate pre-onset analysis. They remain in challenge-task evaluation. On A-test, 123 of 160 eligible events were detected and 37 missed; 19 ever-positive records were left-censored.

Selection optimized the prespecified utility, not event sensitivity alone. The full-run B GRU detected 606/906 events with 1.1067 false episodes/day; logistic regression detected 557/906 with 0.9111 false episodes/day. These are descriptive operating-point comparisons. None establishes clinical benefit. The selected model's substantial missed-event count and continuing false alerts are material limitations.

## Evaluation and reliability experiments

Required reporting includes average precision, AUROC, Brier score, fixed-bin calibration error, calibration bins, normalized challenge utility, eligible pre-onset event sensitivity, warning lead time, and false alert episodes per eligible non-warning patient-day. Counts of censored or otherwise ineligible events accompany the pre-onset analysis. Patient-cluster bootstrap intervals describe uncertainty conditional on the fitted model and split; they do not measure all model-development uncertainty.

The reliability analysis freezes models before hospital B evaluation and simulates observation deletion, arrival delay, and measurement noise at declared severities. Initial feature ablations retrain logistic regression; a separate analysis removes calibration for the selected family. Selective prediction uses distance from the operational threshold as a heuristic. Coverage and missed/withheld events must accompany retained-case performance; withholding has not been shown to make care safer.

The completed B analyses illustrate distinct failure modes:

- With a six-hour availability delay, selected-model utility decreased from 0.2744 to 0.2121; event sensitivity was 54.75%, with 410 missed eligible events. With 50% deletion of observed values, sensitivity was 57.51% and utility 0.2505. The measured effects were not monotonic at every tested severity.
- At measurement-noise severity 0.5 training standard deviations, event sensitivity increased to 62.58% while false episodes increased to 0.7022/day. Higher detection under this perturbation came with a higher alert burden.
- Logistic values-only ablation yielded B utility 0.2191 versus 0.1605 for the full-feature logistic model, while event sensitivity decreased from 61.48% to 53.64% and false episodes decreased from 0.9111 to 0.5332/day. The full feature set did not uniformly improve every objective. Removing XGBoost calibration yielded utility 0.2693 and 0.6463 false episodes/day, compared with calibrated values of 0.2744 and 0.5895/day; these are descriptive comparisons, not significance claims.
- The largest development-fixed withholding cutoff retained 61.39% of B hours. It detected 449/906 events and missed 457, compared with 548 detections and 358 misses without withholding. All warning hours were withheld for 159 eligible events. Lower false alert burden at that cutoff did not establish greater safety.

Detailed estimates, metric intervals, and denominators are in the final report. Each stress condition uses one seeded perturbation realization; its patient-bootstrap intervals do not integrate uncertainty across repeated corruption masks or noise realizations. The ablations are limited to the stated model/component comparisons.

The dataset lacks actual specimen and result-availability timestamps. A delay experiment shifts recorded hourly availability; it cannot reconstruct true laboratory turnaround time. Observation age in the application describes recorded availability, not necessarily physiological age.

Subgroup results use available age categories and recorded Gender codes, with unknown values separated and small groups suppressed. **Subgroup and selective-prediction outputs do not have bootstrap confidence intervals in this release**; they are descriptive point estimates. These limited comparisons cannot demonstrate fairness across the populations affected by a clinical system. Primary and stress/ablation metric intervals must not be interpreted as subgroup or selective-prediction intervals.

## Known limitations and failure modes

- Hospital B represents one transfer, not general transportability. Clinical practice, prevalence, devices, and measurement patterns can differ elsewhere or over time.
- The inspected pilot overlaps 6% of the final B cohort. This limits claims that the entire external cohort was previously unseen, despite A-only full-run fitting and selection.
- Retrospective labels and reconstructed onset are not prospective adjudication. First-row-positive stays have left-censored onset and are excluded from the separate warning analysis.
- Longer stays contribute more training and evaluation hours. Patient bootstrap respects within-stay clustering but does not remove hour-weighting or verify independent people across records.
- Unit/source errors and physiological outliers are not automatically corrected. Artificial Gaussian noise is a controlled statistical stressor, not an estimate of real errors.
- Calibration can deteriorate after population shift, even when discrimination changes little. High classification scores can coexist with excessive alert burden.
- Feature contributions explain uncalibrated mathematical model scores; they do not show causes or recommended interventions. Browser attribution is logistic-only; native XGBoost TreeSHAP is local Python-only. A validated GRU attribution is not implemented.
- Synthetic prevalence and physiology are deliberately constructed. Neither synthetic scores nor synthetic hospital differences estimate clinical performance.
- Reduced cohorts and small event counts can yield unstable thresholds and wide or undefined intervals. Report these outcomes rather than substituting zero.
- Clinical workflow integration, human response, usability, downstream harms, and prospective benefit remain untested.

## Maintenance and release requirements

No automatic retraining or hospital-B recalibration is part of the primary protocol. Any update requires a new version, a declared protocol amendment, independent evaluation, and an updated card. The complete retrospective run and research artifacts are released under `bundles/full-v1`; manuscript authorship, institutional, funding, conflict, and submission declarations require the responsible author's confirmation where outstanding. Load serialized Python model artifacts only from trusted local runs. Keep raw clinical records outside the repository and all hosted replay. The hosted research demonstration remains private; no public clinical deployment is asserted.

## References

- [PhysioNet/CinC 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/), DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).
- [Original challenge publication](https://doi.org/10.1097/CCM.0000000000004145).
- [Versioned project protocol](protocol.md), [data provenance](data_provenance.md), and [TRIPOD+AI reporting map](tripod_ai_mapping.md).

## Executed pilot evidence

The [pilot report](../bundles/pilot-v1/report.md) and [failure analysis](../reports/pilot_failure_analysis.md) document limited external event detection. Development-selected XGBoost detected 10/54 eligible B events (18.5%) and missed 44. This preliminary model is not suitable for clinical use. Saved model/configuration/manifest files are under `bundles/pilot-v1`; the synthetic bundle is separate.
