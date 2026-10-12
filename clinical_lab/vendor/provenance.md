# PhysioNet Challenge 2019 scoring code

`evaluate_sepsis_score.py` is vendored without modification from:
https://github.com/physionetchallenges/evaluation-2019/blob/4edb43177ee7d89dc86ab211705753c2d27b71e4/evaluate_sepsis_score.py

- Commit: `4edb43177ee7d89dc86ab211705753c2d27b71e4`
- SHA-256: `26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5`
- License: BSD 2-Clause; original `LICENSE.txt` included alongside this file.

The lab calls `compute_prediction_utility` for each patient's full hourly
trajectory. Normalization uses the sums of observed, best, and inaction utility
across the cohort, matching the official scorer. Undefined cohort normalization
(no potentially useful positive prediction) is returned as JSON `null`.
The operational pre-onset warning analysis is a separate, explicitly defined
endpoint and does not alter challenge scoring.
