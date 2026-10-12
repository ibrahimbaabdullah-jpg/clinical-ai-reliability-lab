# Clinical AI Reliability Lab: development and external evaluation of ICU sepsis prediction models under observation degradation

**Research protocol and software manuscript draft — complete-cohort results pending**  
**Project owner and draft attribution:** Ibrahim Khalid Baabdullah  
**Version:** 0.1, 2026-10-12  
**Affiliation, additional authors, corresponding-author details, and contributions:** pending author confirmation; none are inferred here.

This document describes the implemented study workflow and the planned complete-cohort analysis. It is not a completed results paper, a registered protocol, or a claim of full TRIPOD+AI adherence. The executed reduced real-data pilot is documented separately in [the pilot report](../reports/pilot.md) and [failure analysis](../reports/pilot_failure_analysis.md). The Results placeholders below are for the still-pending complete-cohort study. Replace pending fields only with values traceable to the verified execution report and its manifests. Do not combine synthetic and clinical results.

## Structured abstract

**Objective:** To compare clinical prediction models under hospital transfer, missing observations, delayed availability, and measurement noise, while distinguishing challenge-task classification from strictly pre-onset sepsis warning.

**Design and setting:** Retrospective model development and planned external evaluation using the two public ICU cohorts from the PhysioNet/Computing in Cardiology Challenge 2019. Hospital A supplies separate training, tuning, calibration, threshold-selection, and internal-test partitions. Hospital B is reserved for external evaluation.

**Methods:** The platform compares regularized logistic regression, XGBoost, and a unidirectional gated recurrent unit network. Features and sequences use current and past observations only. Primary evaluation measures are average precision, normalized challenge utility, Brier score, pre-onset event sensitivity, and false alert episodes per eligible non-warning patient-day. Patient-cluster bootstrap intervals accompany performance estimates. Reliability experiments include controlled observation degradation, component ablation, and selective prediction with explicit coverage and withheld-event reporting.

**Results:** Complete-cohort results are pending. This draft contains no numerical clinical performance claims. A runnable pipeline or synthetic demonstration does not establish clinical reliability. Any reduced-cohort experiment requires its own cohort counts, configuration, uncertainty, and preliminary designation.

**Conclusions:** The software provides an auditable framework for a retrospective reliability study. Which model performs best, whether withholding helps, and whether findings transfer beyond the evaluated hospital remain empirical questions. Clinical benefit and readiness for patient care have not been established.

## Introduction: context, intended use, and objectives

An hourly classification score alone does not describe the usefulness of an early-warning model. Timing, calibration, missed events, and alert burden also affect interpretation. A model can rely on local observation practices and lose reliability when measurements are missing or care settings change. This project uses a defined retrospective task to make those dependencies inspectable.

The intended users are clinical informatics and machine-learning researchers. The platform supports research and education, not diagnosis, patient monitoring, or treatment recommendations. Its interactive replay uses generated patient trajectories. The target clinical context for the retrospective study is ICU care represented by the released challenge cohorts; applicability to other populations is not assumed.

The primary objective is to compare three model families using a protocol that keeps hospital B out of all model-development decisions. Secondary objectives are to quantify sensitivity to controlled data changes, examine the contributions of feature components and calibration, and investigate whether selective withholding changes errors at an acceptable coverage level. Withholding is not presumed beneficial. A final manuscript must add a focused comparison with relevant existing clinical sepsis models; no claim of novelty or superiority is made in this draft.

## Methods

### Study design, source population, and cohort assembly

The source is PhysioNet/CinC 2019 version 1.0.0, which releases hospital A and B patient records [1,2]. The available public cohort is the target study population, rather than a sample-size estimate based on independent hourly observations. The source publication identifies hospital A as Beth Israel Deaconess Medical Center and hospital B as Emory University Hospital. It describes collection during the preceding decade without providing exact site-specific calendar dates. Source curation excluded stays with fewer than eight hourly windows and sepsis onset less than four hours after ICU admission; records ended at discharge and were capped at two weeks [2]. These are source-reported rules, not filters reapplied by this project. An explicit adult-only eligibility rule was not identified in the cited release documentation. The released fields do not permit independent reconstruction of treatment or outcome adjudication.

Data are acquired from official sources with recorded URLs, file hashes, retrieval time, and a manifest identifying complete or reduced scope. The feasibility pilot selected 1,200 filenames per hospital uniformly without replacement, seed 2019, without consulting labels. Its results were inspected before the full-cohort experiment, which includes those same records. Hospital B was excluded from all fitting and selection, and the previously committed full-cohort protocol/configuration was retained; the external analysis is not a wholly new, never-inspected confirmatory validation. See [execution notes](execution_notes.md). Such a sample is a distinct reduced-cohort experiment, not completion of the full public benchmark. Synthetic trajectories support software tests and public demonstration only.

Each supplied file is treated as one released patient record. Cross-hospital person linkage and repeated admissions cannot be independently identified. Ingestion requires the ordered challenge fields, numeric or missing predictor values, nonreverting binary labels, and consecutive hourly ICU time. Malformed input stops the run with an identified error; it is not silently repaired or dropped. The final participant flow must account for downloads, analyzed records, validation failures, and every partition.

### Outcomes and reference time

The primary classification target is the source `SepsisLabel` without an additional shift. Its positive interval starts six hours before the challenge reference onset and continues afterward [1]. All recorded hours enter the challenge-task analysis. Consequently, model output is a score for the supplied label, not a validated incident-event probability in a prospective event-free population.

For a separate warning analysis, onset is inferred as the first positive hourly row plus six only when an observed zero-to-one transition is present. Stays positive in the first row are left-censored and excluded from this analysis. Cases with inferred onset later than the final observed hour are also excluded. Their counts remain reported. All remain eligible for the challenge-task analysis.

A valid pre-onset warning is an alert in the interval from six hours before inferred onset up to, but not including, onset. Post-onset scores never count as early warning. Threshold crossings emit alert episodes subject to a six-hour refractory interval; sustained elevated scores can generate a further alert once six hours have elapsed. False episodes outside the warning window are divided by eligible non-warning patient-hours divided by 24. The denominator includes never-septic hours and hours earlier than the warning window in eligible septic stays; it excludes post-onset time and the censored septic cases described above. Warning lead time is evaluated among detected eligible events. These operational definitions are separate from the original hourly challenge utility.

The study uses supplied labels rather than re-adjudicating outcomes. It cannot independently verify clinical diagnosis, source measurement timing, or the consistency of label ascertainment across demographic groups.

### Predictors, temporal features, and missing values

All 40 source predictors are candidates; identifiers, hospital labels, and outcomes are excluded. Features are built within patient and in chronological order. The complete tabular representation comprises 40 current/forward-filled values and four additional features for each of the 34 dynamic variables: a six-row observed mean, a carried-value endpoint slope, current missingness, and hours since last recorded availability.

The six-row window contains the current and previous five hourly observations. The slope is the carried value now minus the carried value five hours earlier, divided by five. It remains missing when an endpoint is unavailable. Time since the last available measurement is capped at 48 hours; never-measured variables receive the same cap. Carried values themselves have no 48-hour expiry. No backward interpolation, centered window, final length of stay, or information from later rows is used.

Remaining missing values are median-imputed using training rows only. Features entirely absent in training are retained with zero imputation. Logistic regression and GRU inputs are standardized using training statistics; XGBoost uses imputed unstandardized features. The project retains source numerical values without automatic unit conversion or outlier clipping. Actual per-site missingness, demographic distributions, and feature summaries remain required results. Chronological code cannot establish whether source laboratory results were available at the corresponding clinical time, because specimen/result timestamps are absent.

### Partitioning, development, and model selection

Hospital A patients are stratified by ever-positive status into five disjoint sets: training 55%, tuning 15%, calibration 10%, threshold selection 10%, and internal test 10%. Integer allocations are deterministic. Seed 2019 is used and each assigned patient retains all recorded hours. Hospital B is external only. At least ten positive and ten negative A patients are required to populate the development partitions; insufficient cohorts fail explicitly. Patient manifests and data hashes are saved before fitting.

Model families are regularized logistic regression, XGBoost histogram trees, and a unidirectional GRU. Logistic regression uses balanced class weights. The GRU weights positive loss using the training-hour negative-to-positive ratio, processes complete patient sequences, and masks padded positions. XGBoost uses its recorded configuration; the current benchmark configuration does not request an additional positive-class weight. A separate calibration stage addresses probability distortion from fitting and weighting but cannot guarantee calibration after transfer.

The benchmark configuration specifies logistic inverse-regularization values 0.01, 0.1, 1, and 10; XGBoost candidates with 200/depth-3 or 300/depth-5 trees at learning rate 0.05; and GRU hidden sizes 16 and 32 with up to 20 epochs, patience 3, batch size 64, and learning rate 0.001. The GRU retains its best tuning epoch. Actual run settings and any changes must be copied from the saved configuration, rather than inferred from these defaults. Demonstration configurations are deliberately smaller.

Hyperparameters are selected by tuning-set average precision. Sigmoid calibration of model log-odds is fitted using the calibration partition. Each threshold maximizes original normalized utility on the threshold-selection partition, including an all-negative candidate. The final family is selected on that partition's utility; ties favor logistic regression, then XGBoost, then GRU. Performance for every family is reported on A-test and B without choosing a winner from those results. Preprocessing, calibration, thresholds, and model weights remain frozen for external evaluation. No model updating on B is part of this protocol.

### Performance estimation and uncertainty

Hourly measures are average precision (AP), AUROC, Brier score, fixed 10-bin expected calibration error, calibration-bin summaries, and normalized challenge utility. AP is not labeled as trapezoidal precision–recall area. The original utility implementation and its pinned source provenance are retained in the repository. Utility is normalized at cohort level, not averaged as per-patient normalized scores.

The main operational measures are eligible event sensitivity, false alert episodes per eligible non-warning patient-day, and detected-event warning lead time. Exact event, alert, hour, and exclusion denominators must accompany estimates. Undefined quantities are represented as missing, not zero.

Patient-cluster percentile bootstrap intervals use 1,000 replicates in the benchmark configuration. All hours of each resampled patient remain together. The report records valid replicate counts. Faster software runs use fewer replicates and must say so. Intervals condition on a fitted model and partition and do not incorporate the entire model-selection process. Comparisons of marginal intervals are descriptive, not formal tests of paired model differences. Longer stays contribute more hourly observations; clustering does not remove this weighting.

### Reliability experiments

The development-selected family is evaluated after independently deleting 10%, 30%, or 50% of observed dynamic values; shifting observations by 1, 3, or 6 hourly rows within patients; and adding zero-mean Gaussian measurement noise with standard deviation 0.1, 0.25, or 0.5 times each variable's training-set standard deviation. Missing cells remain missing under noise; labels and administrative fields are unchanged. Perturbations precede causal feature construction. These scenarios are controlled experiments, not estimated clinical error rates or actual laboratory delays.

Logistic regression is retrained without trends, without explicit missingness/time-since features, and using values only. Every component ablation repeats tuning, calibration, and threshold selection in the original assigned partitions. Removing calibration is additionally evaluated for the selected family using a separately selected development threshold. Thus this initial feature-ablation experiment is restricted to logistic regression and is not represented as an exhaustive ablation of all architectures.

Selective prediction uses distance from the operating threshold as a heuristic. Cutoffs are fixed from development threshold-partition distances, not external outcomes. Results must include retained hourly coverage, retained error, withheld positive-label hours and events, and overall event sensitivity when withholding produces no alert. A higher retained-case metric alone does not establish a safer system; clinical review workload is unknown.

### Subgroups and responsible interpretation

Exploratory summaries use recorded Gender codes and age below 65 versus at least 65, with unknown values reported separately. Estimates are suppressed for groups with fewer than 20 patients or five events while denominators are retained. These variables do not capture all dimensions of inequality and cannot establish fairness. Differences in disease prevalence, missingness, label ascertainment, and clinical setting must be considered before interpreting a subgroup contrast.

### Reproducibility, ethics, and reporting

Execution records include patient splits, data/configuration/protocol hashes, source revision and source-tree hash, dependency versions, model parameters, and timestamps. Parquet supports analytical storage; optional local MLflow records experiments. Model bundles preserve preprocessing and calibration. A containerized FastAPI application presents synthetic replay and research results. Feature contributions for supported tabular models describe uncalibrated mathematical scores, not causal mechanisms or treatment effects.

The methods are documented in [protocol.md](protocol.md). The protocol is versioned, not registered; no preregistration is claimed. Any amendments must identify their timing relative to inspected results. This draft uses project-specific TRIPOD+AI reporting domains [3] and [the reporting map](tripod_ai_mapping.md); remaining items prevent a claim of complete adherence.

No project-specific ethics approval, waiver, exemption, or institutional determination is claimed. The responsible author must document the applicable determination before submission if required. No patient or public involvement is documented. The public interface does not accept real patient timelines.

## Results: pending evidence

### Cohort and model-development flow

**PENDING:** verified run ID, complete/reduced designation, total and per-site patient counts, hourly records, positive patients, eligible pre-onset events, follow-up distribution, missingness, validation failures, excluded-event counts, and counts for every partition.

**PENDING:** fitted model versions, hyperparameter comparisons, actual epochs, selected calibration parameters, thresholds, selection criterion values, training warnings, and any failed model fits. Failures must be retained in the report rather than disappearing from the comparison.

### Primary performance table

| Model | Evaluation population | AP (95% CI) | Utility (95% CI) | Brier (95% CI) | Event sensitivity (95% CI) | False alerts/patient-day (95% CI) |
|---|---|---|---|---|---|---|
| Logistic regression | A internal test | PENDING | PENDING | PENDING | PENDING | PENDING |
| Logistic regression | B external | PENDING | PENDING | PENDING | PENDING | PENDING |
| XGBoost | A internal test | PENDING | PENDING | PENDING | PENDING | PENDING |
| XGBoost | B external | PENDING | PENDING | PENDING | PENDING | PENDING |
| GRU | A internal test | PENDING | PENDING | PENDING | PENDING | PENDING |
| GRU | B external | PENDING | PENDING | PENDING | PENDING | PENDING |

Each event metric requires the associated eligible-event and exposure denominators. This table must not be populated with synthetic outputs as if they were clinical results.

### Reliability, subgroups, and failure analysis

**PENDING:** perturbation-response plots with unchanged-model comparisons; feature/calibration ablations; coverage versus error and withheld-event tables; calibration plots; subgroup denominators and suppressed estimates; threshold instability; and representative failure patterns. Any reduced-cohort figures require a prominent preliminary designation. A complete-cohort analysis remains pending until its own acquisition and execution manifests are verified.

## Discussion

No result-based conclusion about the preferred model, transfer degradation, useful withholding, or clinical value can be written before results are available. The principal current contribution is a reproducible framework that separates label-task scoring from pre-onset alerts and connects model evaluation with explicit observation-quality experiments.

The eventual discussion should assess whether changes in calibration, event detection, and alert burden support the same interpretation as discrimination metrics. It should explain simpler-model wins or failures without treating model complexity as a goal. Correlated features and contribution plots cannot establish causal explanations.

Limitations include one external hospital, uncertain linkage of repeated people, source outcome assumptions, absent actual availability timestamps, source-unit ambiguity, retained physiological outliers, limited demographic coverage, hour-weighted learning, single-split model development, and uncertainty not fully covering model selection. Synthetic stressors do not capture all real observation processes. Reduced cohorts can produce unstable choices and imprecise event estimates. Even a strong complete-cohort retrospective result would not establish prospective clinical benefit.

Future work requires independent cohorts, evaluation of measurement and label provenance, additional sources of uncertainty, workflow-aware usability testing, and appropriately governed prospective assessment before clinical application. A synthetic dashboard is an educational interface, not evidence that staff can safely act on the models.

## Declarations and availability

- **Authorship and affiliations:** Ibrahim Khalid Baabdullah is identified as project owner. Final authorship, contributions, affiliations, and contact details require confirmation.
- **Funding and conflicts of interest:** pending author declarations; no statements are fabricated here.
- **Ethics and consent:** applicable institutional determination pending if required. No approval, exemption, or waiver is asserted.
- **Registration:** not registered; the versioned development protocol is in this repository.
- **Patient and public involvement:** none documented in this project.
- **Data:** obtain the released data from PhysioNet. Source records are excluded from the repository. The landing-page/bundled-license discrepancy is recorded in [data_provenance.md](data_provenance.md); project software terms do not relicense the dataset.
- **Code and models:** repository and model-artifact access should be stated at submission using actual release URLs and version identifiers. Repository existence does not imply current public visibility. Fitted artifact and complete-benchmark availability remain to be documented from verified runs.
- **AI assistance:** AI tools assisted software implementation and manuscript drafting. The responsible author must review the code, analyses, references, and final claims and supply any journal-required details of that assistance.

## References

1. Reyna M, Josef C, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*. Version 1.0.0. PhysioNet; 2019. [doi:10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).
2. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*. 2020;48(2):210–217. [doi:10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145).
3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378). [Official checklist](https://www.tripod-statement.org/wp-content/uploads/2019/12/TRIPODAI_checklist.pdf).
