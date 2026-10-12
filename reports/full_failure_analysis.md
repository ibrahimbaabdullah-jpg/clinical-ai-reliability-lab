# Full-cohort failure analysis

The frozen run covers all 40,336 public A/B patients and 1,552,210 patient-hours. XGBoost was selected using hospital A’s threshold partition: utility 0.373260, compared with 0.330094 for GRU and 0.297102 for logistic regression. Model fitting, hyperparameter selection, calibration, thresholds and family selection used A only. All three calibrated thresholds are 0.02. No model was retuned on these results.

## External performance and missed events

Hospital B contains 20,000 patients and 761,995 hours. Of 1,142 patients with positive challenge labels, 223 have first-row-positive labels and 13 have inferred onset beyond follow-up. They remain in hourly challenge scoring but are excluded from the operational endpoint and its person-time denominator, leaving **906 eligible events**; 236/1,142 labelled septic stays (20.67%) are excluded. Warning episodes must occur in `[inferred onset−6, inferred onset)` and observe a six-hour cooldown.

Values in parentheses are 95% patient-cluster percentile intervals, with 1,000 valid replicates per estimate, conditional on the fitted models and chosen split. Displayed values are rounded; the [complete aggregate JSON](../bundles/full-v1/report.json) retains full precision. Average precision (AP), utility and Brier use the original hourly labels, including post-onset rows. Event sensitivity is a separate pre-onset endpoint.

| Model | AP (95% CI) | Utility (95% CI) | Brier (95% CI) | Event sensitivity (95% CI) | False alerts/day (95% CI) |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.053377 (0.048040–0.060812) | 0.160530 (0.133224–0.186977) | 0.014255 (0.013477–0.015067) | 557/906; 61.48% (58.28–64.65) | 0.911059 (0.888915–0.933817) |
| XGBoost (selected on A) | 0.079712 (0.070516–0.090164) | 0.274438 (0.250358–0.299296) | 0.013671 (0.012973–0.014412) | 548/906; 60.49% (57.41–63.53) | 0.589533 (0.567376–0.612834) |
| GRU | 0.052304 (0.047827–0.057727) | 0.160270 (0.131313–0.189316) | 0.013689 (0.013004–0.014423) | 606/906; 66.89% (63.72–69.79) | 1.106663 (1.082280–1.128031) |

XGBoost misses **358/906 events**, logistic regression 349, and GRU 300. Their false-alert counts are respectively **18,418**, 28,463 and 34,574 over the same **749,800 eligible non-warning hours (31,241⅔ patient-days)**. This denominator includes all never-septic hours and eligible septic hours before the warning window, excluding post-onset hours. In never-septic patients alone, XGBoost produces 13,309 episodes over 694,549 hours (0.459890/day). Its median lead time is three hours **among the 548 detected events**, not among all events. GRU detects more events at its frozen threshold but generates substantially more alarms; the results do not establish an acceptable clinical operating point.

## Hospital transfer and calibration

For XGBoost, A-test to B AP changes from 0.162169 to 0.079712, utility from 0.408684 to 0.274438, and sensitivity from 123/160 (76.875%) to 548/906 (60.486%). False-alert rates change from 1.158878/day to 0.589533/day. A-test has 2,033 patients; these transfer differences do not identify their causes.

Brier falls from 0.019844 on A-test to 0.013671 on B while positive-label prevalence also falls from 2.190% to 1.415%; this alone does not demonstrate better calibration. XGBoost expected calibration error (ECE) increases from 0.001644 to 0.003231. Within B’s 16,175 hours scored in `[0.1, 0.2)`, mean predicted probability is 13.765%, versus 8.866% observed positive labels. A small global ECE can obscure overprediction within this score range.

## Controlled perturbations and component ablations

The table shows selected XGBoost on B at each maximum prespecified severity; all models and thresholds remain frozen. Full marginal intervals and all severities are in the aggregate JSON.

| Condition | AP | Utility | Detected / 906 | False alerts/day |
|---|---:|---:|---:|---:|
| Baseline | 0.079712 | 0.274438 | 548 | 0.589533 |
| 50% observed-value deletion | 0.068543 | 0.250488 | 521 | 0.581435 |
| Six-hour availability delay | 0.062230 | 0.212145 | 496 | 0.613636 |
| Noise SD = 0.5 × training SD | 0.075442 | 0.265121 | 567 | 0.702171 |

Six-hour delay increases missed events from 358 to 410. Maximum noise detects more events while producing 21,937 false episodes and worsening AP, utility and Brier (0.013989). Thus increased sensitivity does not itself show improvement. Deletion sensitivity is non-monotonic across severities. These simulated errors and one perturbation seed do not estimate real clinical error rates.

Feature ablations concern **logistic regression**, with A-only refitting, recalibration and threshold selection:

| Feature variant | B AP | B utility | Detected / 906 | False alerts/day |
|---|---:|---:|---:|---:|
| Full features | 0.053377 | 0.160530 | 557 | 0.911059 |
| Without trends | 0.054628 | 0.198013 | 531 | 0.705276 |
| Without explicit missingness/measurement age | 0.051377 | 0.186784 | 487 | 0.666450 |
| Values only | 0.053916 | 0.219132 | 486 | 0.533198 |

Values-only improves external utility but misses 420 events versus 349 with full features. Components therefore do not uniformly improve all outcomes. Removing **XGBoost calibration**, with a separately A-selected threshold, leaves AP unchanged, lowers Brier to 0.013593 and ECE to 0.001770, but lowers utility to 0.269253 and raises false alerts to 0.646284/day (561/906 detected). Calibration learned on A does not uniformly improve B reliability.

## Selective withholding trades alarms for additional missed events

At the most restrictive prespecified XGBoost cutoff, retained B coverage is 467,802/761,995 hours (61.392%). Retained Brier changes from 0.013671 at full coverage to 0.013351, while detection falls from 548/906 to 449/906 (49.558%) and missed events rise from 358 to **457**. False alerts decrease to 0.328344/day. The analysis withholds 294,193 hours, including 4,244 positive-labelled hours; those labels include post-onset time. At least one warning hour is withheld for 458 eligible events, and **all six warning hours are withheld for 159**. These results do not establish safer predictions: threshold distance is a heuristic, and evaluation retains all 906 events while treating withholding as no alert. At 85.513% coverage, retained Brier is lower (0.012734), so more withholding does not monotonically improve it. Human-review outcomes are unmeasured.

## Subgroups and evidence boundaries

For XGBoost on B, age <65 comprises 11,096 patients with detection 290/482 (60.166%); age ≥65 comprises 8,904 with 258/424 (60.849%). Recorded Gender code 0 comprises 9,268 patients with 237/391 (60.614%); code 1 comprises 10,732 with 311/515 (60.388%). No subgroup confidence intervals or interaction tests were calculated. Missing/other strata contain no patients and are suppressed. Similar point sensitivities cannot establish equivalence or fairness, and these limited recorded variables do not cover wider inequities.

All 1,200 B patients examined in the earlier pilot recur in the full B cohort (6%). Thus B was held out of full-run fitting and selection but was **not entirely previously unseen**, and the full run is not an independent replication of the pilot. This is one split, one hospital transfer, retrospective reconstructed onset and simulated data corruption. Comparisons of marginal intervals are not paired significance tests. Aggregate findings cannot identify individual failure mechanisms. Challenge utility is not measured clinical benefit. No prospective benefit, clinical acceptability or deployment readiness has been established.
