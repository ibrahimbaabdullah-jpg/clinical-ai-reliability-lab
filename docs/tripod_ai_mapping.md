# TRIPOD+AI reporting map

This is a **working reporting map**, not a claim of TRIPOD+AI adherence, study registration, ethics approval or completed clinical validation. A protocol and runnable software do not replace empirical results. The final manuscript must identify its analyzed cohort, actual model versions and missing items explicitly.

Reference: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378). The [official checklist](https://www.tripod-statement.org/wp-content/uploads/2019/12/TRIPODAI_checklist.pdf) is content-versioned 11 January 2024; the older year in its URL is not the checklist version.

The rows below are project-specific tracking descriptions, not a reproduction of the official checklist. Consult the linked checklist and its subitems when preparing a submission. “Planned” never means completed.

| Item | Project evidence and manuscript location | Remaining work |
|---|---|---|
| 1: Study identification | README research question; manuscript title | Identify development plus external evaluation, ICU population and sepsis outcome in final title. |
| 2: Abstract | Manuscript abstract planned | Write after results; identify cohort sizes, event counts, model selection, uncertainty and limitations. |
| 3: Context and intended use | README; protocol research question | Compare relevant existing models; explain research users, care context and unavailable inequality information. |
| 4: Objectives | Protocol endpoints and reliability experiments | Keep primary and exploratory objectives distinct; record amendments. |
| 5: Data sources | [Data provenance](data_provenance.md) | Verify original collection periods and representativeness from source publications; do not invent dates. |
| 6: Participants and setting | Protocol population and A/B design | Produce actual cohort flow, exclusions and site descriptions; document unavailable treatment detail. |
| 7: Data preparation | Causal preprocessing and schema checks specified in protocol | Link the final implementation and report actual validation failures and handling by subgroup where possible. |
| 8: Outcome | Protocol hourly-label task and separate pre-onset endpoint | Report label limitations, left-censoring and beyond-followup counts; do not imply adjudicated diagnoses. |
| 9: Predictors | Protocol causal feature specification | Export exact predictor definitions, temporal windows, units and unavailable measurement detail. |
| 10: Study size | Fixed public A/B cohorts | Report patients and events in every partition; assess precision and adequacy rather than asserting power from row count. |
| 11: Missingness | Protocol preprocessing and stress tests | Report observed missingness by feature/site and exact fitted imputation parameters. |
| 12: Analysis | Protocol splits, tuning, calibration, metrics and bootstrap | Link code/configuration and actual execution record; document deviations and any model updating. |
| 13: Imbalance | Model training configuration and calibration planned | State actual weighting/sampling choices and effects on probability calibration. |
| 14: Fairness | Protocol recorded-Gender and age strata | Report denominators, suppressed estimates and uncertainty; acknowledge limited demographic coverage. |
| 15: Outputs and thresholds | Protocol A-only threshold selection | Publish selected thresholds and selection rationale per trained family. |
| 16: Development/evaluation differences | Hospital A development; hospital B external evaluation | Report site-wise population, missingness and predictor distributions. |
| 17: Ethics | Retrospective public-data research status | Report the actual applicable institutional determination if obtained; no approval or exemption is claimed here. |
| 18: Open science | Git history, versioned protocol, source citations and reproducibility commands | State actual funding/conflicts, registration status, access conditions and model artifact availability. |
| 19: Patient/public involvement | No involvement documented | State that directly unless participation actually occurs; do not invent consultation. |
| 20: Participants in results | Cohort flow and descriptive tables planned | Fill using the actual analyzed cohort, events, followup and missingness by hospital and relevant groups. |
| 21: Model-development results | Split/tuning manifests planned | Report actual candidate counts, patient/event counts and hyperparameter selection for all families. |
| 22: Model specification | Serialized preprocessing/model artifacts and inference API planned | Publish exact models, dependencies, parameters and access restrictions sufficient for independent predictions. |
| 23: Performance | Benchmark and failure analysis planned | Report discrimination, calibration, utility, alert burden, event sensitivity and patient-bootstrap intervals. |
| 24: Updating | No hospital B updating in primary protocol | State this explicitly; any later recalibration needs separate data and a declared analysis. |
| 25: Interpretation | Discussion planned | Interpret completed findings without claiming clinical benefit from retrospective metrics. |
| 26: Limitations | Protocol and provenance limits | Add observed failures, sampling uncertainty, site dependence, outcome limitations and prospective validation gaps. |
| 27: Practical use and next steps | Synthetic replay, model card and research-use limits | Describe failed/missing inputs, user expertise and required prospective/workflow evaluation before clinical use. |

## Evidence required before a completed benchmark is claimed

- A complete real-data acquisition manifest and transparent cohort scope.
- Frozen patient partitions and development-only fitting of transforms, calibration and operating thresholds.
- Executed comparisons of all prespecified model families, with failures reported rather than omitted.
- Results under hospital transfer and all prespecified reliability experiments, with patient-level uncertainty and event/alert denominators.
- Actual software/dependency versions, model artifacts, model cards and reproducible commands.
- A manuscript whose tables and claims agree with the generated reports; unresolved items remain explicitly marked.

Smoke-test and synthetic results can demonstrate that software executes. They cannot substitute for the full-cohort study or establish clinical reliability.
