"""Leakage-aware challenge and pre-onset evaluation.

Original labels are used without further shifting for all hourly metrics.  The
separate operational analysis infers onset = first positive row + 6, excludes
left-censored/onset-after-follow-up cases, and only counts observed pre-onset
hours.  It is a retrospective research endpoint, not a clinical alarm claim.
"""
from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import pandas as pd

from .vendor.evaluate_sepsis_score import compute_prediction_utility

SCORER_COMMIT = "4edb43177ee7d89dc86ab211705753c2d27b71e4"
WARNING_HOURS = 6
COOLDOWN_HOURS = 6
PRIMARY_KEYS = ("auroc", "average_precision", "brier", "ece", "challenge_utility", "event_sensitivity", "false_alerts_per_patient_day", "median_lead_time_hours")


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _validated(df: pd.DataFrame, probabilities: Iterable[float]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {"patient_id", "hospital", "hour", "SepsisLabel"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing evaluation columns: {sorted(required-set(df.columns))}")
    if df.empty:
        raise ValueError("Evaluation requires at least one patient-hour")
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) != len(df) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be one finite value in [0,1] per row")
    if df[list(required)].isna().any().any() or not df.SepsisLabel.isin([0, 1]).all():
        raise ValueError("Metadata must be complete and SepsisLabel binary")
    # Evaluation needs metadata and subgroup fields, never the full feature
    # table. Avoid copying 40 clinical columns in each stress/bootstrap call.
    columns = ["patient_id", "hospital", "hour", "SepsisLabel"]
    columns += [column for column in ("Age", "Gender") if column in df]
    frame = df.loc[:, columns].reset_index(drop=True)
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(frame[["hospital", "patient_id"]]), sort=False)
    grouped = frame.groupby(groups, sort=False)
    if not np.array_equal(frame.hour.to_numpy(), grouped.cumcount().to_numpy()):
        raise ValueError("Each patient must have ordered contiguous zero-based hours")
    if grouped.SepsisLabel.diff().lt(0).any():
        raise ValueError("Challenge labels must stay positive after their first positive hour")
    return frame, p, groups


def _check_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or threshold < 0 or threshold > np.nextafter(1.0, 2.0):
        raise ValueError("Threshold must be in [0,1], or nextafter(1,2) for all-negative predictions")


def alert_hours(hours: np.ndarray, predictions: np.ndarray, cooldown: int = COOLDOWN_HOURS) -> np.ndarray:
    """Emit while high, with at least `cooldown` hours between emitted alarms."""
    if cooldown < 1:
        raise ValueError("Cooldown must be positive")
    emitted: list[int] = []
    for hour in np.asarray(hours)[np.asarray(predictions, dtype=bool)]:
        if not emitted or hour - emitted[-1] >= cooldown:
            emitted.append(int(hour))
    return np.asarray(emitted, dtype=int)


def _utility_triplet(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    # This window and the cohort-level normalization exactly reproduce the pinned
    # official evaluate_sepsis_score; utility is never averaged patient by patient.
    best = np.zeros(len(labels), dtype=int)
    if labels.any():
        onset = int(np.argmax(labels)) + 6
        best[max(0, onset - 12): min(onset + 4, len(labels))] = 1
    return tuple(float(compute_prediction_utility(labels, values, check_errors=False))
                 for values in (predictions, best, np.zeros(len(labels), dtype=int)))


def _patient_summaries(frame: pd.DataFrame, predictions: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_labels = frame.SepsisLabel.to_numpy(dtype=int)
    all_hours = frame.hour.to_numpy(dtype=int)
    positions = pd.Series(np.arange(len(frame))).groupby(groups, sort=False).indices
    for group_id, idx in positions.items():
        labels = all_labels[idx]
        hourly = predictions[idx]
        observed, best, inaction = _utility_triplet(labels, hourly)
        row: dict[str, Any] = dict(group=int(group_id), observed_utility=observed,
            best_utility=best, inaction_utility=inaction, eligible=0, detected=0,
            false_alerts=0, sepsis_free_hours=0, nonseptic_false_alerts=0,
            nonseptic_hours=0, lead_time=np.nan, excluded_left_censored=0,
            excluded_onset_after_followup=0, septic=int(labels.any()))
        hours = all_hours[idx]
        if not labels.any():
            alarms = alert_hours(hours, hourly)
            row.update(false_alerts=len(alarms), sepsis_free_hours=len(hours),
                nonseptic_false_alerts=len(alarms), nonseptic_hours=len(hours))
        elif labels[0] == 1:
            row["excluded_left_censored"] = 1
        else:
            onset = int(hours[np.flatnonzero(labels)[0]]) + 6
            if onset > int(hours[-1]):
                row["excluded_onset_after_followup"] = 1
            else:
                # Post-onset rows never enter operational event/alert analysis.
                before = hours < onset
                alarms = alert_hours(hours[before], hourly[before])
                window_start = onset - WARNING_HOURS
                valid = alarms[(alarms >= window_start) & (alarms < onset)]
                row.update(eligible=1, detected=int(len(valid) > 0),
                    false_alerts=int((alarms < window_start).sum()),
                    sepsis_free_hours=int((hours < window_start).sum()),
                    lead_time=float(onset-valid[0]) if len(valid) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def _operational(stats: pd.DataFrame, counts: np.ndarray | None = None,
                 _columns: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    weights = np.ones(len(stats), dtype=int) if counts is None else np.asarray(counts, dtype=int)
    columns = _columns if _columns is not None else {
        name: stats[name].to_numpy(dtype=float) for name in stats.columns
    }
    def total(name: str) -> float:
        return float(np.dot(weights, columns[name]))
    leads = np.repeat(columns["lead_time"], weights)
    leads = leads[np.isfinite(leads)]
    eligible, detected = total("eligible"), total("detected")
    observed, best, inaction = (total(name) for name in ("observed_utility", "best_utility", "inaction_utility"))
    return dict(challenge_utility=_ratio(observed-inaction, best-inaction),
        observed_utility=observed, best_utility=best, inaction_utility=inaction,
        eligible_events=int(eligible), detected_events=int(detected), missed_events=int(eligible-detected),
        event_sensitivity=_ratio(detected, eligible), false_alert_episodes=int(total("false_alerts")),
        sepsis_free_patient_days=total("sepsis_free_hours")/24,
        false_alerts_per_patient_day=_ratio(total("false_alerts")*24, total("sepsis_free_hours")),
        nonseptic_false_alert_episodes=int(total("nonseptic_false_alerts")),
        nonseptic_patient_days=total("nonseptic_hours")/24,
        nonseptic_false_alerts_per_patient_day=_ratio(total("nonseptic_false_alerts")*24, total("nonseptic_hours")),
        median_lead_time_hours=float(np.median(leads)) if len(leads) else None,
        excluded_left_censored=int(total("excluded_left_censored")),
        excluded_onset_after_followup=int(total("excluded_onset_after_followup")),
        septic_patients=int(total("septic")))


class _HourlyMetrics:
    """Cache score ordering so cluster bootstraps do not sort every replicate."""
    def __init__(self, labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray):
        self.y, self.p, self.groups = labels.astype(float), probabilities, groups
        self.order = np.argsort(-probabilities, kind="stable")
        ordered = probabilities[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(ordered)-1]
        self.bin = np.minimum((probabilities * 10).astype(int), 9)
        self.sorted_y = self.y[self.order]
        self.sorted_groups = groups[self.order]
        self.squared_errors = (self.p-self.y)**2

    def metrics(self, counts: np.ndarray, *, primary_only: bool = False) -> dict[str, Any]:
        float_counts = counts.astype(float)
        weights = float_counts[self.groups]
        n = float(weights.sum())
        w, y = float_counts[self.sorted_groups], self.sorted_y
        tps = np.cumsum(w*y)[self.ends]
        fps = np.cumsum(w*(1-y))[self.ends]
        positives, negatives = float(tps[-1]), float(fps[-1])
        auroc = None
        if positives and negatives:
            fpr, tpr = np.r_[0, fps/negatives], np.r_[0, tps/positives]
            auroc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
        ap = None
        if positives:
            precision = np.divide(tps, tps+fps, out=np.zeros_like(tps), where=tps+fps > 0)
            ap = float(np.sum(np.diff(np.r_[0, tps/positives])*precision))
        bins_n = np.bincount(self.bin, weights=weights, minlength=10)
        bins_p = np.bincount(self.bin, weights=weights*self.p, minlength=10)
        bins_y = np.bincount(self.bin, weights=weights*self.y, minlength=10)
        primary = dict(auroc=auroc, average_precision=ap,
            brier=float(np.dot(weights, self.squared_errors)/n),
            ece=float(np.abs(bins_p-bins_y).sum()/n))
        if primary_only:
            return primary
        return dict(**primary,
            prevalence=float(np.dot(weights,self.y)/n), patient_hours=int(n),
            reliability_bins=[dict(lower=i/10, upper=(i+1)/10, count=int(bins_n[i]),
                mean_probability=_ratio(bins_p[i], bins_n[i]), observed_fraction=_ratio(bins_y[i],bins_n[i]))
                for i in range(10)])


def _confidence_intervals(cache: _HourlyMetrics, stats: pd.DataFrame, bootstrap: int, seed: int) -> dict[str, Any]:
    if bootstrap < 0:
        raise ValueError("Bootstrap replicates cannot be negative")
    collected: dict[str, list[float]] = {name: [] for name in PRIMARY_KEYS}
    rng, n = np.random.default_rng(seed), len(stats)
    columns = {name: stats[name].to_numpy(dtype=float) for name in stats.columns}
    for _ in range(bootstrap):
        counts = np.bincount(rng.integers(0,n,size=n), minlength=n)
        result = cache.metrics(counts, primary_only=True) | _operational(stats, counts, _columns=columns)
        for key in PRIMARY_KEYS:
            value = result[key]
            if value is not None and np.isfinite(value):
                collected[key].append(float(value))
    return {key: dict(lower=float(np.quantile(values,.025)) if values else None,
                     upper=float(np.quantile(values,.975)) if values else None,
                     valid_replicates=len(values), requested_replicates=bootstrap,
                     method="patient_cluster_percentile_95")
            for key,values in collected.items()}


def _subgroups(frame: pd.DataFrame, p: np.ndarray, threshold: float, groups: np.ndarray) -> dict[str, Any]:
    first = frame.groupby(groups, sort=False).first()
    age = pd.to_numeric(first.get("Age", pd.Series(np.nan,index=first.index)), errors="coerce")
    gender = pd.to_numeric(first.get("Gender", pd.Series(np.nan,index=first.index)), errors="coerce")
    selections = {"age_lt65": age < 65, "age_ge65": age >= 65, "age_missing": age.isna(),
                  "gender_0": gender == 0, "gender_1": gender == 1, "gender_missing_or_other": ~gender.isin([0,1])}
    output: dict[str,Any] = {}
    for name, selection in selections.items():
        mask = np.isin(groups, first.index[selection])
        if not mask.any():
            output[name] = dict(patients=0, positive_patients=0, suppressed=True, reason="Fewer than 20 patients or 5 positive patients")
            continue
        part, probs, ids = _validated(frame.loc[mask], p[mask])
        positive_patients = int(part.groupby(ids).SepsisLabel.max().sum())
        count = int(len(np.unique(ids)))
        if count < 20 or positive_patients < 5:
            output[name] = dict(patients=count, positive_patients=positive_patients, suppressed=True,
                                reason="Fewer than 20 patients or 5 positive patients")
        else:
            metrics = _HourlyMetrics(part.SepsisLabel.to_numpy(),probs,ids).metrics(np.ones(count,dtype=int))
            op = _operational(_patient_summaries(part,probs >= threshold,ids))
            # Event endpoint needs adequate eligible events, independent of label positives.
            if op["eligible_events"] < 5:
                op["event_sensitivity"] = None
                op["median_lead_time_hours"] = None
                op["event_metrics_suppressed"] = True
            output[name] = dict(patients=count, positive_patients=positive_patients, suppressed=False,
                                hourly_metrics=metrics, pre_onset=op)
    return output


def evaluate(df: pd.DataFrame, probabilities: Iterable[float], threshold: float,
             bootstrap: int = 200, seed: int = 2019) -> dict[str, Any]:
    """Evaluate frozen predictions; all numbers are JSON-safe (undefined → None)."""
    _check_threshold(threshold)
    frame, p, groups = _validated(df, probabilities)
    n = len(np.unique(groups))
    labels = frame.SepsisLabel.to_numpy(dtype=int)
    predictions = p >= threshold
    stats = _patient_summaries(frame,predictions,groups)
    cache = _HourlyMetrics(labels,p,groups)
    hourly = cache.metrics(np.ones(n,dtype=int))
    hourly.update(threshold=float(threshold), sensitivity=_ratio(np.sum(predictions & (labels==1)),np.sum(labels)),
        precision=_ratio(np.sum(predictions & (labels==1)),np.sum(predictions)),
        accuracy=float(np.mean(predictions==labels)), positive_predictions=int(predictions.sum()))
    op = _operational(stats)
    challenge = {key:op.pop(key) for key in ("challenge_utility","observed_utility","best_utility","inaction_utility")}
    return dict(patients=n, patient_hours=len(frame), hourly_metrics=hourly,
        challenge=challenge, pre_onset=op, confidence_intervals=_confidence_intervals(cache,stats,bootstrap,seed),
        subgroups=_subgroups(frame,p,threshold,groups),
        protocol=dict(label_alignment="Original challenge labels; no additional shift",
            onset="first positive hour + 6; inferred reference onset",
            warning_window="[onset-6, onset)", warning_window_hours=WARNING_HOURS,
            alarm_policy="Emit on any high hour if at least 6 hours since previous emitted alarm; persistent highs can re-alert",
            cooldown_hours=COOLDOWN_HOURS,
            false_alert_denominator="Never-septic hours plus eligible septic hours before onset-6; excludes post-onset and censored septic cases",
            excluded_cases="First row positive, or inferred onset later than last observed hour, excluded only from pre-onset analysis",
            challenge_policy="Every hourly binary prediction scored; no alarm cooldown applied",
            auprc_definition="average precision (stepwise precision-recall area)",
            scorer_commit=SCORER_COMMIT, bootstrap_unit="patient within evaluated cohort", bootstrap_seed=seed))


def choose_threshold(df: pd.DataFrame, probabilities: Iterable[float]) -> float:
    """Maximize official normalized utility on the threshold-selection split only.

    Fixed candidate grid and conservative tie-break (higher threshold). All-negative
    is explicitly represented, including when calibrated probabilities equal 1.
    """
    frame, p, groups = _validated(df, probabilities)
    candidates = np.r_[0,.001,.002,.005,.01,.02,.03,.04,.05,.075,.1,.15,.2,.3,.4,.5,.65,.8,1,np.nextafter(1.,2.)]
    patients = [(part.SepsisLabel.to_numpy(dtype=int),p[part.index.to_numpy()])
                for _,part in frame.groupby(groups,sort=False)]
    best_threshold, best_score = float(candidates[-1]), -float("inf")
    for threshold in candidates:
        observed = sum(float(compute_prediction_utility(y, scores >= threshold,check_errors=False)) for y,scores in patients)
        # All candidates share the same normalization denominator and baseline.
        if observed >= best_score:
            best_score, best_threshold = observed, float(threshold)
    return best_threshold


def selective(df: pd.DataFrame, probabilities: Iterable[float], threshold: float,
              cutoffs: Iterable[float] = (0,.01,.025,.05,.1,.2)) -> dict[str,Any]:
    """Test heuristic abstention without treating confidence as proof of safety."""
    _check_threshold(threshold)
    frame,p,groups = _validated(df,probabilities)
    labels = frame.SepsisLabel.to_numpy(dtype=int)
    rows = []
    for cutoff in cutoffs:
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("Selective distance cutoffs must be finite and nonnegative")
        keep = np.abs(p-threshold) >= cutoff
        predictions = (p >= threshold) & keep
        op = _operational(_patient_summaries(frame,predictions,groups))
        withheld_events, all_withheld_events, withheld_warning_hours = 0,0,0
        for _,part in frame.groupby(groups,sort=False):
            y=part.SepsisLabel.to_numpy(dtype=int)
            if not y.any() or y[0]:
                continue
            onset=int(np.flatnonzero(y)[0])+6
            if onset>int(part.hour.max()):
                continue
            idx=part.index[(part.hour>=onset-6)&(part.hour<onset)].to_numpy()
            withheld=~keep[idx]
            withheld_events+=int(withheld.any())
            all_withheld_events+=int(withheld.all())
            withheld_warning_hours+=int(withheld.sum())
        rows.append(dict(distance_cutoff=float(cutoff), coverage=float(keep.mean()), retained_hours=int(keep.sum()),
            withheld_hours=int((~keep).sum()), withheld_positive_label_hours=int(labels[~keep].sum()),
            retained_classification_error=float(np.mean((p[keep]>=threshold)!=labels[keep])) if keep.any() else None,
            retained_brier=float(np.mean((p[keep]-labels[keep])**2)) if keep.any() else None,
            eligible_events_with_any_withheld_warning_hour=withheld_events,
            eligible_events_with_all_warning_hours_withheld=all_withheld_events,
            withheld_warning_hours=withheld_warning_hours, eligible_events=op["eligible_events"],
            detected_events=op["detected_events"], missed_events=op["missed_events"],
            event_sensitivity_with_withheld_as_no_alert=op["event_sensitivity"],
            false_alerts_per_patient_day=op["false_alerts_per_patient_day"]))
    return dict(method="Absolute probability distance from frozen decision threshold",
        limitation="A heuristic confidence proxy; abstention is not guaranteed to improve safety. Report coverage and withheld events together.",
        results=rows)
