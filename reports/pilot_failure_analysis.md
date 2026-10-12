# Pilot failure analysis

This is a deterministic reduced-cohort experiment, not the final 40,336-patient study. No hyperparameters or thresholds were changed after observing these results.

## Hospital transfer exposes limited warning performance

XGBoost was selected using hospital A's threshold partition (utility 0.3632), before test evaluation. Its utility decreased from 0.2978 on A-test to 0.1025 on B. Its pre-onset event sensitivity was 3/9 on A-test and 10/54 on B. B sensitivity was 18.5%, with a 200-replicate patient-bootstrap 95% interval of approximately 8.9–27.9%. It missed 44 of 54 eligible B events. The small A-test event denominator makes that comparison imprecise.

Logistic regression detected 23/54 eligible B events (42.6%; 95% interval approximately 29.4–55.8%), but produced 0.752 false alert episodes per eligible patient-day versus 0.085 for XGBoost. The comparison illustrates a tradeoff; it does not establish that either chosen threshold is clinically acceptable. Thresholds were optimized for the original challenge utility, not for a clinical minimum-sensitivity target.

All 13 first-row-positive B patients were excluded only from the inferred pre-onset endpoint because onset is left-censored; all remained in challenge scoring. No B event had inferred onset beyond available follow-up in this sample.

## Discrimination and perturbations

Average precision on B was 0.0272 for logistic regression, 0.0578 for XGBoost and 0.0291 for the five-epoch GRU pilot. These scores describe the supplied shifted hourly labels, including post-onset rows. They are not pure pre-onset event discrimination.

For the selected XGBoost, six-hour measurement delay reduced B average precision from 0.0578 to approximately 0.0441. Removing 50% of available observations reduced it to approximately 0.0500. Some noise levels increased AP slightly; this is not evidence that measurement error is helpful. The experiment reports marginal bootstrap intervals and has not performed paired significance testing.

## Implications for the full study

Retain the frozen benchmark configuration; do not tune on B. Execute the full cohort with the production tuning grids and 1,000 bootstrap replicates, report uncertainty and subgroup counts, and distinguish any protocol amendment from the initial protocol. Consider any alternative alert objective as a separately prespecified study, rather than retroactively optimizing B sensitivity. No prospective clinician workflow, patient benefit or clinical suitability has been demonstrated.
