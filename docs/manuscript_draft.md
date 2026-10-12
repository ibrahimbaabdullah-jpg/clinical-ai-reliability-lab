# Clinical AI Reliability Lab: development and external evaluation of ICU sepsis prediction models under observation degradation

**Completed retrospective analysis — research manuscript draft**

**Project owner and draft attribution:** Ibrahim Khalid Baabdullah

**Version:** 0.2, 2026-10-12

**Affiliation, additional authors, corresponding-author details, and contributions:** pending author confirmation; none are inferred here.

This draft reports the completed `full-v1` retrospective analysis of the complete public cohort, with results traceable to its [report](../bundles/full-v1/report.json), [run manifest](../bundles/full-v1/run_manifest.json), configuration and fitted models. It remains a manuscript draft because author-specific declarations and final author review are unresolved. It is not a registered study, a claim of full TRIPOD+AI adherence or a clinical validation for deployment. The earlier reduced pilot is documented separately in [its report](../reports/pilot.md) and [failure analysis](../reports/pilot_failure_analysis.md); synthetic replay is not used as clinical evidence.

## Structured abstract

**Objective:** To compare clinical prediction models under hospital transfer, missing observations, delayed availability, and measurement noise, while distinguishing challenge-task classification from strictly pre-onset sepsis warning.

**Design and setting:** Retrospective model development and external evaluation using the two public ICU cohorts from the PhysioNet/Computing in Cardiology Challenge 2019. Hospital A supplies separate training, tuning, calibration, threshold-selection, and internal-test partitions. Hospital B is reserved for external evaluation.

**Methods:** The platform compares regularized logistic regression, XGBoost, and a unidirectional gated recurrent unit network. Features and sequences use current and past observations only. Primary evaluation measures are average precision, normalized challenge utility, Brier score, pre-onset event sensitivity, and false alert episodes per eligible non-warning patient-day. Patient-cluster bootstrap intervals accompany performance estimates. Reliability experiments include controlled observation degradation, component ablation, and selective prediction with explicit coverage and withheld-event reporting.

**Results:** All 40,336 records (1,552,210 hours; 2,932 ever-positive records) were analyzed. Development selected XGBoost. In hospital B, AP was 0.080 (95% CI 0.071–0.090); 548/906 eligible events were detected (60.5%; 95% CI 57.4%–63.5%), with 0.590 false alerts per eligible non-warning patient-day (95% CI 0.567–0.613). Six-hour observation delay reduced B detections to 496. Withholding could lower retained-case error while missing additional events. The full B analysis overlaps 1,200 previously inspected pilot records; the protocol and fitting/selection boundaries were unchanged.

**Conclusions:** Model rankings and apparent improvements depended on the endpoint. Hospital transfer and delayed observations reduced performance, while lower retained error under withholding concealed additional missed events. The executed study supports transparent reporting of calibration, event detection, alert burden and coverage together. It does not establish clinical benefit or readiness for patient care.

## Introduction: context, intended use, and objectives

An hourly classification score alone does not describe the usefulness of an early-warning model. Timing, calibration, missed events, and alert burden also affect interpretation. A model can rely on local observation practices and lose reliability when measurements are missing or care settings change. This project uses a defined retrospective task to make those dependencies inspectable.

The intended users are clinical informatics and machine-learning researchers. The platform supports research and education, not diagnosis, patient monitoring, or treatment recommendations. Its interactive replay uses generated patient trajectories. The target clinical context for the retrospective study is ICU care represented by the released challenge cohorts; applicability to other populations is not assumed.

The original challenge supplied a Weibull-Cox baseline and demonstrated that algorithm rankings depended on the metric and hospital [2]. This study uses a separate A-to-B development/evaluation design, a regularized logistic baseline, boosted trees and a causal recurrent model. It reproduces the challenge target and utility calculation, not the original baseline algorithm or hidden-test leaderboard. The hidden third hospital is unavailable here, so results cannot be compared directly with published challenge rankings.

The primary objective is to compare three model families using a protocol that keeps hospital B out of all model-development decisions. Secondary objectives are to quantify sensitivity to controlled data changes, examine the contributions of feature components and calibration, and describe selective-withholding tradeoffs among coverage, classification error and event detection. Withholding is not presumed beneficial; no claim of algorithmic novelty is made.

## Methods

### Study design, source population, and cohort assembly

The source is PhysioNet/CinC 2019 version 1.0.0, which releases hospital A and B patient records [1,2]. The target is all available public records; no a priori sample-size calculation was performed. Adequacy is described using patient/event counts and confidence-interval precision, rather than treating hourly rows as independent patients. The source publication identifies hospital A as Beth Israel Deaconess Medical Center and hospital B as Emory University Hospital. It describes collection during the preceding decade without providing exact site-specific calendar dates. Source curation excluded stays with fewer than eight hourly windows and sepsis onset less than four hours after ICU admission; records ended at discharge and were capped at two weeks [2]. These are source-reported rules, not filters reapplied by this project. An explicit adult-only eligibility rule was not identified in the cited release documentation. The released fields do not permit independent reconstruction of treatment or outcome adjudication.

Data are acquired from official sources with recorded URLs, file hashes, retrieval time, and a manifest identifying complete or reduced scope. The feasibility pilot selected 1,200 filenames per hospital uniformly without replacement, seed 2019, without consulting labels. Its results were inspected before the full-cohort experiment, which includes those same records. Hospital B was excluded from all fitting and selection, and the previously committed full-cohort protocol/configuration was retained; the external analysis is not a wholly new, never-inspected confirmatory validation. See [execution notes](execution_notes.md). Such a sample is a distinct reduced-cohort experiment, not completion of the full public benchmark. Synthetic trajectories support software tests and public demonstration only.

Each supplied file is treated as one released patient record. Cross-hospital person linkage and repeated admissions cannot be independently identified. Ingestion requires the ordered challenge fields, numeric or missing predictor values, nonreverting binary labels, and consecutive hourly ICU time. Malformed input stops the run with an identified error; it is not silently repaired or dropped. The participant flow below accounts for acquisition, analyzed records, validation and every partition.

### Outcomes and reference time

The source pairs suspected infection with a two-point SOFA increase within 24 hours. Antibiotic-first cases require cultures within 24 hours; culture-first cases require antibiotics within 72 hours, with at least 72 consecutive hours of antibiotic administration. Reference onset is the earlier qualifying time when SOFA deterioration falls between 24 hours before and 12 hours after suspicion [1]. We use the supplied labels rather than reconstructing this definition from unavailable treatment fields.

The primary classification target is the source `SepsisLabel` without an additional shift. Its positive interval starts six hours before the challenge reference onset and continues afterward [1]. All recorded hours enter the challenge-task analysis. Consequently, model output is a score for the supplied label, not a validated incident-event probability in a prospective event-free population.

For a separate warning analysis, onset is inferred as the first positive hourly row plus six only when an observed zero-to-one transition is present. Stays positive in the first row are left-censored and excluded from this analysis. Cases with inferred onset later than the final observed hour are also excluded. Their counts remain reported. All remain eligible for the challenge-task analysis.

A valid pre-onset warning is an alert in the interval from six hours before inferred onset up to, but not including, onset. Post-onset scores never count as early warning. An above-threshold hour emits an alert episode when no alert occurred in the preceding six hours; sustained elevated scores can therefore generate another alert after the refractory interval. False episodes outside the warning window are divided by eligible non-warning patient-hours divided by 24. The denominator includes never-septic hours and hours earlier than the warning window in eligible septic stays; it excludes post-onset time and the censored septic cases described above. Warning lead time is evaluated among detected eligible events. These operational definitions are separate from the original hourly challenge utility.

The study uses supplied labels rather than re-adjudicating outcomes. It cannot independently verify clinical diagnosis, source measurement timing, or the consistency of label ascertainment across demographic groups.

### Predictors, temporal features, and missing values

All 40 source predictors are candidates; identifiers, hospital labels, and outcomes are excluded. Features are built within patient and in chronological order. The complete tabular representation comprises 40 current/forward-filled values and four additional features for each of the 34 dynamic variables: a six-row observed mean, a carried-value endpoint slope, current missingness, and hours since last recorded availability.

The six-row window contains the current and previous five hourly observations. The slope is the carried value now minus the carried value five hours earlier, divided by five. It remains missing when an endpoint is unavailable. Time since the last available measurement is capped at 48 hours; never-measured variables receive the same cap. Carried values themselves have no 48-hour expiry. No backward interpolation, centered window, final length of stay, or information from later rows is used.

Remaining missing values are median-imputed using training rows only. Features entirely absent in training are retained with zero imputation. Logistic regression and GRU inputs are standardized using training statistics; XGBoost uses imputed unstandardized features. The project retains source numerical values without automatic unit conversion or outlier clipping. Per-site missingness, demographic distributions and feature summaries are reported below and in the aggregate profile. Chronological code cannot establish whether source laboratory results were available at the corresponding clinical time, because specimen/result timestamps are absent.

### Partitioning, development, and model selection

Hospital A patients are stratified by ever-positive status into five disjoint sets: training 55%, tuning 15%, calibration 10%, threshold selection 10%, and internal test 10%. Integer allocations are deterministic. Seed 2019 is used and each assigned patient retains all recorded hours. Hospital B is external only. At least ten positive and ten negative A patients are required to populate the development partitions; insufficient cohorts fail explicitly. Patient manifests and data hashes are saved before fitting.

Model families are regularized logistic regression, XGBoost histogram trees, and a unidirectional GRU. Logistic regression uses balanced class weights. The GRU weights positive loss using the training-hour negative-to-positive ratio, processes complete patient sequences, and masks padded positions. XGBoost uses its recorded configuration; the current benchmark configuration does not request an additional positive-class weight. A separate calibration stage addresses probability distortion from fitting and weighting but cannot guarantee calibration after transfer.

The executed benchmark configuration specifies logistic inverse-regularization values 0.01, 0.1, 1, and 10; XGBoost candidates with 200/depth-3 or 300/depth-5 trees at learning rate 0.05; and GRU hidden sizes 16 and 32 with up to 20 epochs, patience 3, batch size 64, and learning rate 0.001. The GRU retains its best tuning epoch. The saved configuration records these settings; the smaller demonstration configurations are not used for this analysis.

Hyperparameters are selected by tuning-set average precision. Sigmoid calibration of model log-odds is fitted using the calibration partition. Each operating threshold maximizes original normalized utility over a prespecified grid on the threshold-selection partition; the grid includes an all-negative candidate, and ties choose the higher threshold. The final family is selected on that partition's utility; ties favor logistic regression, then XGBoost, then GRU. Performance for every family is reported on A-test and B without choosing a winner from those results. Preprocessing, calibration, thresholds, and model weights remain frozen for external evaluation. No model updating on B is part of this protocol.

### Performance estimation and uncertainty

Hourly measures are average precision (AP), AUROC, Brier score, fixed 10-bin expected calibration error, calibration-bin summaries, and normalized challenge utility. AP is not labeled as trapezoidal precision–recall area. The original utility implementation and its pinned source provenance are retained in the repository. Utility is normalized at cohort level, not averaged as per-patient normalized scores.

The main operational measures are eligible event sensitivity, false alert episodes per eligible non-warning patient-day, and detected-event warning lead time. Exact event, alert, hour and exclusion denominators accompany the estimates. Undefined quantities are represented as missing, not zero.

Patient-cluster percentile bootstrap intervals use 1,000 replicates. All hours of each resampled patient remain together. The report records valid replicate counts. Intervals condition on a fitted model and partition and do not incorporate the entire model-selection process. Comparisons of marginal intervals are descriptive, not formal tests of paired model differences. Longer stays contribute more hourly observations; clustering does not remove this weighting.

### Reliability experiments

The development-selected family is evaluated after independently deleting 10%, 30%, or 50% of observed dynamic values; shifting observations by 1, 3, or 6 hourly rows within patients; and adding zero-mean Gaussian measurement noise with standard deviation 0.1, 0.25, or 0.5 times each variable's training-set standard deviation. Missing cells remain missing under noise; labels and administrative fields are unchanged. Perturbations precede causal feature construction. Each severity uses one fixed seeded corruption realization; patient-bootstrap intervals condition on that realization and do not quantify variation across repeated corruption draws, model refits or data splits. These scenarios are controlled experiments, not estimated clinical error rates or actual laboratory delays.

Logistic regression is retrained without trends, without explicit missingness/time-since features, and using values only. Every component ablation repeats tuning, calibration, and threshold selection in the original assigned partitions. Removing calibration is additionally evaluated for the selected family using a separately selected development threshold. Thus this initial feature-ablation experiment is restricted to logistic regression and is not represented as an exhaustive ablation of all architectures.

Selective prediction uses distance from the operating threshold as a heuristic. Each family's cutoffs are zero and the 10th, 25th and 50th percentiles of absolute probability-to-threshold distance in the A threshold partition, fixed without external outcomes. Reporting includes retained hourly coverage, retained error, withheld positive-label hours, eligible events with any or all warning-window hours withheld, and overall event sensitivity when withholding produces no alert. Selective-policy estimates have no confidence intervals in this implementation. A higher retained-case metric alone does not establish a safer system; clinical review workload is unknown.

### Subgroups and responsible interpretation

Exploratory summaries use recorded Gender codes and age below 65 versus at least 65, with unknown values reported separately. Estimates are suppressed for groups with fewer than 20 patients or five ever-positive patients while denominators are retained. Event sensitivity and lead time are additionally suppressed when fewer than five eligible pre-onset events remain. Subgroup point estimates do not have bootstrap confidence intervals in this implementation; the subgroup-uncertainty component of TRIPOD+AI item 23a remains incomplete. These variables do not capture all dimensions of inequality and cannot establish fairness. Differences in disease prevalence, missingness, label ascertainment, and clinical setting must be considered before interpreting a subgroup contrast.

### Reproducibility, ethics, and reporting

Execution records include patient splits, data/configuration/protocol hashes, source revision and source-tree hash, dependency versions, model parameters, and timestamps. Parquet supports analytical storage; optional local MLflow records experiments. Model bundles preserve preprocessing and calibration. A containerized FastAPI application presents synthetic replay and research results. Feature contributions for supported tabular models describe uncalibrated mathematical scores, not causal mechanisms or treatment effects.

The methods are documented in [protocol.md](protocol.md). The protocol is versioned, not registered; no preregistration is claimed. Any amendments must identify their timing relative to inspected results. This draft uses project-specific TRIPOD+AI reporting domains [3] and [the reporting map](tripod_ai_mapping.md); remaining items prevent a claim of complete adherence.

No project-specific ethics approval, waiver, exemption, or institutional determination is claimed. The responsible author must document the applicable determination before submission if required. No patient or public involvement is documented. The replay interface does not accept real patient timelines.

## Results

### Cohort assembly and source differences

Run `full-v1` analyzed all 40,336 verified source files (255,057,484 bytes): 20,336 records from A and 20,000 from B, totaling 1,552,210 recorded hours. Every acquired record passed the implemented schema checks and entered the hourly analysis; no project-level record exclusion or silent repair was applied. There were 2,932 ever-positive records. The source curation rules described above were already reflected in the release.

| Population/partition | Records | Hourly rows | Ever-positive records | Eligible pre-onset events | First-row-positive exclusions | Onset-after-follow-up exclusions |
|---|---:|---:|---:|---:|---:|---:|
| Hospital A, total | 20,336 | 790,215 | 1,790 | 1,585 | 203 | 2 |
| A training | 11,185 | 435,883 | 985 | 868 | 115 | 2 |
| A tuning | 3,050 | 118,820 | 268 | 239 | 29 | 0 |
| A calibration | 2,034 | 79,280 | 179 | 165 | 14 | 0 |
| A threshold selection | 2,034 | 77,827 | 179 | 153 | 26 | 0 |
| A internal test | 2,033 | 78,405 | 179 | 160 | 19 | 0 |
| Hospital B, external | 20,000 | 761,995 | 1,142 | 906 | 223 | 13 |
| Total A + B | 40,336 | 1,552,210 | 2,932 | 2,491 | 426 | 15 |

A partition rows subdivide the A-total row; they are not additional records. Warning-analysis exclusions are disjoint and do not remove records from the challenge task. Counts were calculated from the frozen split manifest and source labels, independently of predictions.

Ever-positive prevalence was 8.80% in A and 5.71% in B. Median recorded stay was 39 hours [IQR 25–47] and 38 [23–47], respectively; both ranged from 8 to 336 hours. Median recorded age was 64.67 [52.29–75.66] versus 62 [50–72]. B contained 22 records with source Age below 18, which were retained under the all-released-records protocol. This is not an adult-only analysis. A had 8,502/11,834 records with recorded Gender codes 0/1; B had 9,268/10,732. No age or Gender values were missing. Source Age=100 occurred in 392 B records and denotes the ≥90 category, not exact age.

Observation patterns differed materially. Hourly DBP missingness was 48.13% in A versus 13.94% in B; respiratory-rate missingness was 9.78% versus 21.14%; lactate missingness was 96.57% versus 98.12%. EtCO2 was absent in every A hour but observed in 57,636 B hours. Thus some external input availability had no development counterpart. Complete patient-weighted demographics and all 40 predictors' missingness/quantiles are in the [aggregate cohort profile](../reports/full-v1-cohort/cohort_profile.md).

### Model development and frozen operating points

All three model families were fitted using their prespecified candidates. Logistic regression selected C=0.01: tuning AP was 0.074741, compared with 0.074717, 0.074722 and 0.074720 for C=0.1, 1 and 10. XGBoost selected 200 trees/depth 3, tuning AP 0.116398, rather than 300 trees/depth 5, AP 0.097927. Both used learning rate 0.05.

The GRU selected 32 hidden units with best tuning AP 0.091954 at epoch 9. Training stopped after epoch 12 and restored that checkpoint. The 16-unit candidate's best AP was 0.084762 at epoch 7. The selected GRU used batch size 64, learning rate 0.001 and positive-loss weight 45.370532; both candidates used the predefined 20-epoch maximum and patience 3.

| Family | Selected tuning AP | Frozen calibrated threshold | A threshold-partition utility |
|---|---:|---:|---:|
| Logistic regression | 0.074741 | 0.02 | 0.297102 |
| XGBoost | 0.116398 | 0.02 | 0.373260 |
| GRU | 0.091954 | 0.02 | 0.330094 |

XGBoost was selected before either held-out population was evaluated. The comparison below retains every family; B results did not change the selection, weights, calibration or operating points.

For the fitted calibration transform `sigmoid(a × logit(p) + b)`, the slope/intercept pairs were 0.845254/−3.772286 for logistic regression, 1.078381/0.228169 for XGBoost and 0.546675/−3.406748 for the GRU. Raw probabilities are clipped to [10⁻⁷, 1−10⁻⁷] before the logit. Exact coefficients, preprocessing statistics and model weights are retained in the fitted bundles.

Run completion was recorded with exit code 0. The [execution telemetry](../reports/full-v1-telemetry.json) records 2,086.077 seconds wall time and peak child-process RSS of 6,857,928 KiB (approximately 6.54 GiB) in a shared x86_64 Linux environment, with OMP threads set to two and BLAS threads to one. This is a reproducibility record, not a dedicated-hardware speed benchmark. The saved environment includes Python 3.12.14, scikit-learn 1.8.0, XGBoost 3.4.1 and CPU PyTorch 2.14.1.

### Primary performance

Values are point estimates with 95% patient-cluster percentile confidence intervals. Each displayed interval used 1,000 valid replicates. AP is hourly average precision; event sensitivity is the separate eligible pre-onset endpoint. False alerts use eligible non-warning patient-days.

| Model | Evaluation population | AP (95% CI) | Utility (95% CI) | Brier (95% CI) | Event sensitivity (95% CI) | False alerts/patient-day (95% CI) |
|---|---|---|---|---|---|---|
| Logistic regression | A internal test | 0.103 (0.088–0.131) | 0.397 (0.328–0.458) | 0.0216 (0.0184–0.0251) | 0.838 (0.776–0.897) | 1.493 (1.406–1.572) |
| Logistic regression | B external | 0.053 (0.048–0.061) | 0.161 (0.133–0.187) | 0.0143 (0.0135–0.0151) | 0.615 (0.583–0.647) | 0.911 (0.889–0.934) |
| XGBoost | A internal test | 0.162 (0.128–0.203) | 0.409 (0.341–0.470) | 0.0198 (0.0171–0.0224) | 0.769 (0.706–0.833) | 1.159 (1.077–1.234) |
| XGBoost | B external | 0.080 (0.071–0.090) | 0.274 (0.250–0.299) | 0.0137 (0.0130–0.0144) | 0.605 (0.574–0.635) | 0.590 (0.567–0.613) |
| GRU | A internal test | 0.104 (0.085–0.138) | 0.328 (0.251–0.389) | 0.0205 (0.0178–0.0231) | 0.775 (0.705–0.837) | 1.593 (1.512–1.662) |
| GRU | B external | 0.052 (0.048–0.058) | 0.160 (0.131–0.189) | 0.0137 (0.0130–0.0144) | 0.669 (0.637–0.698) | 1.107 (1.082–1.128) |

[Figure 1](../reports/figures/full-v1/01_primary_performance.png) summarizes primary performance with uncertainty; [Figure 2](../reports/figures/full-v1/02_external_calibration.png) shows B calibration bins and score counts.

The development-selected XGBoost model's AP decreased from 0.162 in A-test to 0.080 in B, and utility from 0.409 to 0.274. It detected 123/160 eligible A-test events and 548/906 B events. Logistic regression detected more A-test events and the GRU detected more B events, but both generated more false episodes than XGBoost at their own development-selected operating points. These are descriptive comparisons, not paired significance tests or a post hoc model-selection rule.

| Model | Population | Detected/eligible events | Missed events | False alert episodes | Median lead time among detected events, hours |
|---|---|---:|---:|---:|---:|
| Logistic regression | A test | 134/160 | 26 | 4,771 | 3 |
| Logistic regression | B external | 557/906 | 349 | 28,463 | 3 |
| XGBoost | A test | 123/160 | 37 | 3,703 | 3 |
| XGBoost | B external | 548/906 | 358 | 18,418 | 3 |
| GRU | A test | 124/160 | 36 | 5,091 | 3 |
| GRU | B external | 606/906 | 300 | 34,574 | 3 |

The false-alert exposure was 76,688 eligible hours (3,195.333 patient-days) in A-test and 749,800 hours (31,241.667 patient-days) in B. The denominators include early pre-warning time in eligible positive stays as specified in Methods; they are not just never-positive ICU stays. Lead time is conditional on a detected event and does not describe the missed events.

Hourly label prevalence was 2.190% in A-test and 1.415% in B. XGBoost's fixed-bin ECE was 0.001644 and 0.003231; GRU ECE was 0.001410 and 0.000926. These small aggregate errors should not be read as evidence of safe decisions: calibration bins are dominated by low-probability hours, and B XGBoost still missed 358 eligible events. The Brier score's decrease in B also occurs alongside lower label prevalence and does not negate poorer event detection.

### Observation-quality stress tests

The frozen XGBoost model was evaluated in all 18 prespecified perturbed conditions. The table contains point estimates; patient-bootstrap intervals and unchanged-versus-perturbed views are retained in the complete report and [Figure 3](../reports/figures/full-v1/03_observation_stress.png). No model or threshold was refitted for these tests.

| Population | Condition | AP | Utility | Brier | Detected/eligible events | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|
| A test | Unchanged | 0.162 | 0.409 | 0.0198 | 123/160 | 1.159 |
| A test | Delete 10% | 0.161 | 0.401 | 0.0198 | 121/160 | 1.144 |
| A test | Delete 30% | 0.153 | 0.392 | 0.0200 | 119/160 | 1.093 |
| A test | Delete 50% | 0.138 | 0.407 | 0.0202 | 115/160 | 1.049 |
| A test | Delay 1 h | 0.156 | 0.407 | 0.0199 | 120/160 | 1.153 |
| A test | Delay 3 h | 0.141 | 0.394 | 0.0201 | 119/160 | 1.146 |
| A test | Delay 6 h | 0.120 | 0.372 | 0.0203 | 119/160 | 1.127 |
| A test | Noise 0.1 × training SD | 0.163 | 0.405 | 0.0198 | 119/160 | 1.160 |
| A test | Noise 0.25 × training SD | 0.143 | 0.407 | 0.0200 | 122/160 | 1.223 |
| A test | Noise 0.5 × training SD | 0.143 | 0.401 | 0.0202 | 128/160 | 1.340 |
| B external | Unchanged | 0.080 | 0.274 | 0.0137 | 548/906 | 0.590 |
| B external | Delete 10% | 0.078 | 0.268 | 0.0137 | 533/906 | 0.588 |
| B external | Delete 30% | 0.075 | 0.259 | 0.0137 | 516/906 | 0.583 |
| B external | Delete 50% | 0.069 | 0.250 | 0.0138 | 521/906 | 0.581 |
| B external | Delay 1 h | 0.077 | 0.266 | 0.0137 | 530/906 | 0.591 |
| B external | Delay 3 h | 0.071 | 0.245 | 0.0138 | 499/906 | 0.592 |
| B external | Delay 6 h | 0.062 | 0.212 | 0.0139 | 496/906 | 0.614 |
| B external | Noise 0.1 × training SD | 0.078 | 0.276 | 0.0137 | 553/906 | 0.599 |
| B external | Noise 0.25 × training SD | 0.078 | 0.274 | 0.0138 | 553/906 | 0.626 |
| B external | Noise 0.5 × training SD | 0.075 | 0.265 | 0.0140 | 567/906 | 0.702 |

A six-hour availability delay reduced B AP from 0.080 to 0.062, utility from 0.274 to 0.212, and detected events from 548 to 496, while false alerts rose from 0.590 to 0.614 per eligible patient-day. Deleting 50% of observations reduced B detections to 521. Under the largest noise condition, detections instead increased to 567 while false alerts rose to 0.702 and Brier error increased. Higher sensitivity under noise therefore did not imply improved reliability. The response was not monotonic for every metric/severity; these are fixed stochastic realizations, not formal estimates of corruption-response variability.

### Component ablations

All four prespecified ablations completed. Feature ablations refitted logistic regression with their own A-only tuning, calibration and threshold selection; the no-calibration ablation used the selected XGBoost model with a separately selected A threshold. These are complete-pipeline comparisons, not isolated coefficient interventions. Point estimates are shown; 1,000-replicate patient-bootstrap intervals are in the [full report](../bundles/full-v1/report.json).

| Variant | Population | Threshold | AP | Utility | Brier | Detected/eligible events | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic, no trends | A test | 0.02 | 0.097 | 0.400 | 0.0218 | 134/160 | 1.493 |
| Logistic, no trends | B external | 0.02 | 0.055 | 0.198 | 0.0142 | 531/906 | 0.705 |
| Logistic, no missingness/time-since | A test | 0.03 | 0.090 | 0.344 | 0.0216 | 100/160 | 0.816 |
| Logistic, no missingness/time-since | B external | 0.03 | 0.051 | 0.187 | 0.0144 | 487/906 | 0.666 |
| Logistic, values only | A test | 0.03 | 0.087 | 0.343 | 0.0219 | 100/160 | 0.824 |
| Logistic, values only | B external | 0.03 | 0.054 | 0.219 | 0.0143 | 486/906 | 0.533 |
| XGBoost, no calibration | A test | 0.02 | 0.162 | 0.403 | 0.0199 | 125/160 | 1.247 |
| XGBoost, no calibration | B external | 0.02 | 0.080 | 0.269 | 0.0136 | 561/906 | 0.646 |

Removing explicit missingness/time-since features or using values only reduced A-test detections from the full logistic model's 134 to 100. In B, the values-only variant had higher utility (0.219 versus 0.161) and lower false-alert burden (0.533 versus 0.911), but detected fewer events (486 versus 557). Removing trends yielded 531 B detections. These findings do not establish that any component is uniformly beneficial across populations or endpoints; the ablations were not used to replace the frozen primary model.

Removing XGBoost calibration preserved AP, as expected for this monotonic transform, and yielded lower B Brier error (0.013593 versus 0.013671) and ECE (0.001770 versus 0.003231). It also increased B detections from 548 to 561 and false alerts from 0.590 to 0.646, with slightly lower utility (0.269 versus 0.274). Calibration therefore did not improve every metric after transfer. No paired hypothesis test was performed for these differences.

### Selective prediction

XGBoost's B policies show the tradeoff between withholding and missed events. Cutoffs below are absolute distances from the frozen 0.02 threshold. The any/all columns count eligible events with at least one/all warning-window hours withheld; they are not disjoint categories. All 906 eligible events stay in the event-detection denominator.

| Distance cutoff | Hourly coverage | Retained classification error | Detected/eligible events | Withheld positive-label hours | Events with any/all warning hours withheld | False alerts/patient-day |
|---|---:|---:|---:|---:|---:|---:|
| 0.000000 | 100.0% | 11.5% | 548/906 | 0 | 0/0 | 0.590 |
| 0.003787 | 94.9% | 10.0% | 514/906 | 892 | 196/1 | 0.475 |
| 0.008062 | 85.5% | 9.5% | 479/906 | 2,165 | 319/28 | 0.398 |
| 0.012976 | 61.4% | 11.1% | 449/906 | 4,244 | 458/159 | 0.328 |

At cutoff 0.008062, B retained 85.5% of hours and classification error among retained hours decreased from 11.5% to 9.5%, but detection decreased from 548/906 to 479/906. At the largest cutoff, 159 events had their entire warning window withheld and only 449/906 events were detected. The most restrictive A-test policy retained 52.2% of hours, detected 102/160 events versus 123/160 without withholding, and reduced false alerts from 1.159 to 0.653 per eligible patient-day.

The direction was similar for other families at their most restrictive B cutoffs: logistic regression detected 435 events versus 557 without withholding, and the GRU detected 469 versus 606. At those settings, 162 and 231 events, respectively, had all warning-window hours withheld. These results do not establish a safety benefit, and no clinician review/response mechanism was evaluated. Selective-policy estimates are descriptive and have no confidence intervals. All-family policy values and coverage are available in the report; [Figure 4](../reports/figures/full-v1/04_selective_withholding.png) illustrates selected-model hospital B results.

### Exploratory subgroups

The table reports selected-model point estimates with explicit patient/event denominators. No displayed subgroup is below the suppression limits. Missing/other age and Gender groups have zero records and remain represented as suppressed empty groups in the JSON report.

| Population | Group | Records | Ever-positive records | AP | Detected/eligible events | False alerts/patient-day |
|---|---|---:|---:|---:|---:|---:|
| A test | Age <65 | 1,032 | 92 | 0.165 | 68/82 | 1.174 |
| A test | Age ≥65 | 1,001 | 87 | 0.165 | 55/78 | 1.144 |
| A test | Gender code 0 | 851 | 72 | 0.148 | 47/65 | 1.096 |
| A test | Gender code 1 | 1,182 | 107 | 0.170 | 76/95 | 1.205 |
| B external | Age <65 | 11,096 | 627 | 0.069 | 290/482 | 0.579 |
| B external | Age ≥65 | 8,904 | 515 | 0.098 | 258/424 | 0.602 |
| B external | Gender code 0 | 9,268 | 495 | 0.085 | 237/391 | 0.554 |
| B external | Gender code 1 | 10,732 | 647 | 0.077 | 311/515 | 0.619 |

B event sensitivity was approximately 60% in each recorded age/Gender stratum, while subgroup AP and alert burden differed. This does not demonstrate parity or fairness: subgroup confidence intervals are absent, only limited demographic variables are available, and differences in prevalence and measurement processes can influence these comparisons. The implementation reports all three families' subgroup results without using them to tune models.

## Discussion

The predefined development rule selected XGBoost, which also had the highest observed AP and challenge utility in both evaluation populations. The GRU's greater architectural complexity did not yield better performance on those endpoints in this experiment. Event sensitivity gave a different comparison: logistic regression detected more A-test events and the GRU more B events, with greater false-alert burden at their respective frozen operating points. No single metric or model ranking fully captured that tradeoff.

Transfer from A to B reduced XGBoost AP, utility and pre-onset sensitivity. The model still missed 358 of 906 eligible B events while producing 18,418 false episodes over 31,241.667 eligible non-warning patient-days. No prespecified clinical acceptability target was evaluated, and these estimates do not establish a suitable operating point for care. Lower B label prevalence affects AP and Brier comparisons; a smaller Brier score in B cannot be interpreted as improved transfer performance by itself. Positive challenge utility is a property of the challenge scoring policy, not measured patient benefit.

Feature ablations also changed the tradeoff: a simpler logistic representation reduced B false alerts and increased utility while missing more events. Removing XGBoost calibration reduced B Brier error/ECE but increased alert burden. Thus neither added feature complexity nor a calibration stage was uniformly beneficial across all reported endpoints.

The observation experiments illustrate why sensitivity must be interpreted alongside burden. Delaying recorded availability reduced early detections; adding noise sometimes increased detections and false alerts simultaneously. Site-level differences in observation availability, including a variable absent throughout A, make the source shift concrete, but this study does not identify which differences caused the transfer loss. Controlled corruptions are not estimates of actual error mechanisms, and isolated point-estimate improvements under perturbation do not prove robustness.

Selective withholding reduced retained-case classification error at some cutoffs while reducing overall event detection. Reporting only retained error would obscure the resulting missed events and fully withheld warning windows. A review pathway might change that tradeoff, but no such pathway was simulated or evaluated. These findings support reporting coverage and withheld-event counts together rather than treating distance from the threshold as evidence of safety.

The design separates the supplied hourly-label target from strictly pre-onset warnings. It also freezes preprocessing, calibration, operating points and family selection before held-out evaluation. Nevertheless, the 1,200 B records inspected in the pilot are included in the full analysis. Keeping the previously committed protocol unchanged prevents post-pilot endpoint/model-selection changes in this run; it does not turn the B evaluation into wholly unseen confirmatory validation.

Additional limitations include one external hospital, uncertain linkage of repeated people, unavailable exact source collection dates, source outcome assumptions, absent specimen/result-availability timestamps, source-unit ambiguity, retained physiological outliers, and limited demographic coverage. Longer stays contribute more training/evaluation hours. The candidate grids are limited, and the study uses one split and training seed. Bootstrap intervals condition on fitted models and one corruption realization and omit model-selection, retraining and split uncertainty. Subgroup and selective-policy intervals are absent. The data and metrics do not establish equal performance across affected populations or prospective clinical benefit.

Future work should evaluate independently collected cohorts, repeated development/corruption realizations, subgroup uncertainty, measurement/label provenance and the stability of operating thresholds. Prospective workflow evaluation would need to establish how people respond to alerts and withheld predictions, including burdens and harms. The synthetic dashboard demonstrates executable model behavior; it is not evidence that clinical staff can safely act on these models.

## Declarations and availability

- **Authorship and affiliations:** Ibrahim Khalid Baabdullah is identified as project owner. Final authorship, contributions, affiliations, and contact details require confirmation.
- **Funding and conflicts of interest:** pending author declarations; no statements are fabricated here.
- **Ethics and consent:** applicable institutional determination pending if required. No approval, exemption, or waiver is asserted.
- **Registration:** not registered; the versioned development protocol is in this repository.
- **Patient and public involvement:** none documented in this project.
- **Data:** obtain the released data from PhysioNet. Source records are excluded from the repository. The landing-page/bundled-license discrepancy is recorded in [data_provenance.md](data_provenance.md); project software terms do not relicense the dataset.
- **Code and models:** exact fitted models, preprocessing/calibration, saved configuration, environment, report and checksums are available to repository users in [`bundles/full-v1`](../bundles/full-v1/). The training source revision is `7274308bc66c7d54c3c0481d06a590a68fd2bcc2`; release version 0.2.0 was assigned after execution. Access remains subject to repository/Site sharing settings; no unrestricted public release is implied. Load serialized Python models only from trusted sources. Raw clinical records and row-level predictions are not part of the shared model bundle.
- **AI assistance:** AI tools assisted software implementation and manuscript drafting. The responsible author must review the code, analyses, references, and final claims and supply any journal-required details of that assistance.

## References

1. Reyna M, Josef C, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*. Version 1.0.0. PhysioNet; 2019. [doi:10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).
2. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*. 2020;48(2):210–217. [doi:10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145).
3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378). [Official checklist](https://www.tripod-statement.org/wp-content/uploads/2019/12/TRIPODAI_checklist.pdf).
