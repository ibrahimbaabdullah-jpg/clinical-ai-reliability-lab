#!/usr/bin/env python3
"""Render standalone PNG/SVG research figures from an existing report.json.

Usage: python scripts/render_figures.py --report bundles/pilot-v1/report.json \
    --output artifacts/figures-pilot

Requires matplotlib (the optional reporting dependency). No model is fitted and
no patient records are loaded. Figure-data JSON preserves the exact plotted
estimates, available intervals, endpoint definitions and source-report digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


FAMILIES = ("logistic", "xgboost", "gru")
NAMES = {"logistic": "Logistic regression", "xgboost": "XGBoost", "gru": "Temporal GRU"}
COHORTS = {"test": ("A internal test", "#127D82", "o"),
           "external": ("B external test", "#B66322", "s")}
METRICS = (
    ("average_precision", "hourly_metrics", "Average precision", "Average precision (0–1)"),
    ("challenge_utility", "challenge", "Normalized challenge utility", "Utility; 0 = no positive predictions"),
    ("event_sensitivity", "pre_onset", "Pre-onset event sensitivity", "Detected / all eligible events"),
    ("false_alerts_per_patient_day", "pre_onset", "False-alert episode rate", "Episodes / eligible non-warning patient-day"),
)
DEFINITIONS = {
    "hourly_task": "Supplied six-hour-shifted challenge labels, including post-onset hours; labels are not shifted again.",
    "average_precision": "Stepwise average precision, not trapezoidal precision-recall area.",
    "challenge_utility": "Official normalized hourly challenge utility; the alert cooldown is not applied.",
    "event_sensitivity": "Detected eligible pre-onset events / all eligible pre-onset events; warning window [inferred onset − 6, inferred onset).",
    "false_alerts_per_patient_day": "False alert episodes / (eligible non-warning patient-hours / 24). Includes never-septic hours and eligible septic hours before onset − 6; excludes warning/post-onset hours and censored septic cases. Six-hour alert cooldown.",
    "intervals": "Available report-supplied 95% patient-cluster percentile intervals; separate marginal intervals are not a significance test.",
    "selective": "Withheld predictions count as no alert. Event sensitivity retains all eligible events in its denominator. Events with all warning hours withheld are a subset of events with any warning hour withheld.",
}


def finite(value: Any) -> float | None:
    """Undefined values stay undefined; in particular, never coerce them to zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def estimate(evaluation: dict, key: str, section: str) -> dict:
    point = finite(evaluation.get(section, {}).get(key))
    item = evaluation.get("confidence_intervals", {}).get(key, {})
    lower, upper = finite(item.get("lower")), finite(item.get("upper"))
    valid = item.get("valid_replicates")
    has_interval = (point is not None and lower is not None and upper is not None
                    and lower <= upper and finite(valid) is not None and valid > 0)
    if has_interval and item.get("method") != "patient_cluster_percentile_95":
        raise ValueError("Figure interval labels require report-supplied 95% patient-cluster percentile intervals")
    return {"estimate": point, "lower": lower if has_interval else None,
            "upper": upper if has_interval else None,
            "valid_replicates": valid, "requested_replicates": item.get("requested_replicates"),
            "interval_method": item.get("method"),
            "status": "undefined" if point is None else "estimate_and_interval" if has_interval else "estimate_only"}


def primary_data(report: dict) -> list[dict]:
    rows = []
    for family in FAMILIES:
        if family not in report["models"]:
            continue
        for split in COHORTS:
            evaluation = report["models"][family].get(split, {})
            for key, section, _, _ in METRICS:
                rows.append({"family": family, "split": split, "metric": key,
                             "patients": evaluation.get("patients"),
                             "eligible_events": evaluation.get("pre_onset", {}).get("eligible_events"),
                             "eligible_nonwarning_patient_days": evaluation.get("pre_onset", {}).get("sepsis_free_patient_days"),
                             **estimate(evaluation, key, section)})
    return rows


def calibration_data(report: dict) -> dict:
    family = report["selected_family"]
    evaluation = report["models"][family].get("external", {})
    bins = []
    for item in evaluation.get("hourly_metrics", {}).get("reliability_bins", []):
        count = finite(item.get("count"))
        lower, upper = finite(item.get("lower")), finite(item.get("upper"))
        mean, fraction = finite(item.get("mean_probability")), finite(item.get("observed_fraction"))
        if count is None or count < 0 or int(count) != count or lower is None or upper is None or not 0 <= lower < upper <= 1:
            raise ValueError("Calibration bins need ordered bounds in [0,1] and nonnegative integer counts")
        if count and (mean is None or fraction is None or not 0 <= mean <= 1 or not 0 <= fraction <= 1):
            raise ValueError("A nonempty calibration bin requires finite probability and label-fraction means")
        bins.append({"lower": lower, "upper": upper, "count": int(count),
                     "mean_probability": mean, "observed_fraction": fraction})
    bins.sort(key=lambda item: item["lower"])
    if any(left["upper"] > right["lower"] + 1e-12 for left, right in zip(bins, bins[1:])):
        raise ValueError("Calibration bins must not overlap")
    total = sum(item["count"] for item in bins)
    hours = finite(evaluation.get("patient_hours"))
    if bins and hours is not None and total != hours:
        raise ValueError("Calibration-bin counts do not reconcile with the reported patient-hour denominator")
    for item in bins:
        item["patient_hour_share"] = item["count"] / total if total else None
    return {"family": family, "split": "external", "patients": evaluation.get("patients"),
            "patient_hours": total, "empty_bins": sum(item["count"] == 0 for item in bins), "bins": bins}


def stress_data(report: dict) -> list[dict]:
    family = report["selected_family"]
    rows, seen = [], set()
    aliases = {"missingness": "missing", "missing": "missing", "delay": "delay", "noise": "noise"}
    for item in report.get("stress_tests", []):
        if item.get("family") != family or item.get("split") not in COHORTS:
            continue
        kind, severity = aliases.get(item.get("kind")), finite(item.get("severity"))
        if kind is None or severity is None or severity <= 0:
            continue
        identity = (item["split"], kind, severity)
        if identity in seen:
            raise ValueError(f"Ambiguous duplicate stress result: {identity}")
        seen.add(identity)
        for key, section, _, _ in METRICS:
            if key == "challenge_utility":
                continue
            rows.append({"family": family, "split": item["split"], "kind": kind,
                         "severity": severity, "baseline": False,
                         "metric": key, **estimate(item.get("evaluation", {}), key, section)})
    for split in COHORTS:
        evaluation = report["models"][family].get(split, {})
        for kind in ("missing", "delay", "noise"):
            for key, section, _, _ in METRICS:
                if key != "challenge_utility":
                    rows.append({"family": family, "split": split, "kind": kind,
                                 "severity": 0., "baseline": True, "metric": key,
                                 **estimate(evaluation, key, section)})
    return rows


def selective_data(report: dict) -> dict:
    family = report["selected_family"]
    entry = report.get("selective_prediction", {}).get(f"{family}_external", {})
    rows = []
    keys = ("distance_cutoff", "coverage", "eligible_events", "detected_events", "missed_events",
            "event_sensitivity_with_withheld_as_no_alert", "eligible_events_with_any_withheld_warning_hour",
            "eligible_events_with_all_warning_hours_withheld", "withheld_hours", "retained_hours")
    for item in entry.get("results", []):
        row = {key: finite(item.get(key)) for key in keys}
        coverage = row["coverage"]
        if coverage is not None and not 0 <= coverage <= 1:
            raise ValueError("Selective coverage must lie in [0,1]")
        eligible = row["eligible_events"]
        detected, sensitivity = row["detected_events"], row["event_sensitivity_with_withheld_as_no_alert"]
        if eligible is not None and eligible > 0 and detected is not None and sensitivity is not None:
            if not math.isclose(sensitivity, detected / eligible, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError("Selective sensitivity must retain all eligible events in its denominator")
        any_withheld, all_withheld = (row["eligible_events_with_any_withheld_warning_hour"],
                                    row["eligible_events_with_all_warning_hours_withheld"])
        if all(value is not None for value in (eligible, any_withheld, all_withheld)):
            if not 0 <= all_withheld <= any_withheld <= eligible:
                raise ValueError("Withheld-event counts must be nested within all eligible events")
        rows.append(row)
    denominators = {row["eligible_events"] for row in rows if row["eligible_events"] is not None}
    if len(denominators) > 1:
        raise ValueError("Selective policies must use the same all-eligible-event denominator")
    return {"family": family, "split": "external", "method": entry.get("method"), "rows": rows}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.edgecolor": "#88969B", "axes.linewidth": .65, "text.color": "#24363D",
        "axes.labelcolor": "#24363D", "xtick.color": "#46555B", "ytick.color": "#46555B",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E3E9EB", "grid.linewidth": .65, "savefig.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "clinical-ai-reliability-lab", "figure.dpi": 110})


def scope(report: dict) -> str:
    metadata = report.get("metadata", {})
    patients = metadata.get("patients", "unreported")
    patients = f"{patients:,}" if isinstance(patients, int) else str(patients)
    return f"{metadata.get('evidence_status', 'Evidence status unreported')} · {patients} patients · selected: {NAMES.get(report['selected_family'], report['selected_family'])}"


def heading(fig, title: str, report: dict) -> None:
    fig.suptitle(title, x=.065, y=.985, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.065, .943, scope(report), ha="left", va="top", fontsize=9, color="#59696E")


def footer(fig, text: str) -> None:
    fig.text(.065, .023, text, ha="left", va="bottom", fontsize=8.3, linespacing=1.45, color="#52636B")


def cohort_legend(fig, report: dict, *, y: float = .903) -> None:
    selected = report["models"][report["selected_family"]]
    handles = [Line2D([], [], color=color, marker=marker, linewidth=1.3,
                      markersize=5.5, label=f"{name} (n={selected.get(split, {}).get('patients', 'unreported')})")
               for split, (name, color, marker) in COHORTS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.061, y),
               frameon=False, ncol=2, fontsize=9, handlelength=2, columnspacing=2)


def missing_note(rows: list[dict]) -> str:
    absent = sum(row.get("status") == "undefined" for row in rows)
    no_ci = sum(row.get("status") == "estimate_only" for row in rows)
    return f"Undefined estimates omitted: {absent}; estimates without usable intervals: {no_ci}."


def save(fig, output: Path, name: str, title: str) -> list[str]:
    names = []
    for extension in ("png", "svg"):
        filename = f"{name}.{extension}"
        fig.savefig(output / filename, dpi=240, bbox_inches="tight", pad_inches=.12,
                    metadata={"Title": title, "Description": "Source values and endpoint definitions: figure-data.json"}
                    if extension == "svg" else {"Title": title})
        names.append(filename)
    plt.close(fig)
    return names


def interval_mark(ax, x: float, point: dict, color: str, marker: str, *, horizontal: bool = False) -> None:
    value = point["estimate"]
    if value is None:
        return
    lower, upper = point["lower"], point["upper"]
    # Draw interval independently of the estimate: percentile bounds need not
    # mathematically straddle the point estimate. Do not clamp or alter them.
    if lower is not None and upper is not None:
        if horizontal:
            ax.hlines(x, lower, upper, color=color, linewidth=1.3, zorder=2)
            ax.vlines([lower, upper], x-.045, x+.045, color=color, linewidth=1.1)
        else:
            ax.vlines(x, lower, upper, color=color, linewidth=1.1, zorder=2)
    ax.plot(value if horizontal else x, x if horizontal else value, marker=marker,
            markersize=5, linestyle="none", color=color, zorder=3)


def primary_figure(report: dict, rows: list[dict], output: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.subplots_adjust(left=.20, right=.97, top=.83, bottom=.21, wspace=.47, hspace=.63)
    heading(fig, "Model performance across hospital cohorts", report)
    cohort_legend(fig, report)
    families = [family for family in FAMILIES if family in report["models"]]
    for ax, (key, _, title, label) in zip(axes.flat, METRICS):
        selected_rows = [row for row in rows if row["metric"] == key]
        ax.set_title(title, loc="left", pad=11)
        for family_index, family in enumerate(families):
            for offset, split in zip((-.12, .12), COHORTS):
                row = next(item for item in selected_rows if item["family"] == family and item["split"] == split)
                _, color, marker = COHORTS[split]
                interval_mark(ax, family_index + offset, row, color, marker, horizontal=True)
                if row["estimate"] is None:
                    ax.text(.015, family_index + offset, "not estimable", transform=ax.get_yaxis_transform(),
                            fontsize=8, color=color, va="center")
        labels = [NAMES[family] + (" *" if family == report["selected_family"] else "") for family in families]
        ax.set_yticks(range(len(families)), labels)
        ax.set_ylim(len(families)-.55, -.55)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        values = [row[field] for row in selected_rows for field in ("estimate", "lower", "upper") if row[field] is not None]
        if key == "event_sensitivity":
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        elif key == "challenge_utility":
            low, high = min([0.] + values), max([.1] + values)
            padding = (high-low)*.10
            ax.set_xlim(low-padding if low < 0 else 0, high+padding)
            ax.axvline(0, linewidth=.8, linestyle=":", color="#7E8B91")
        else:
            upper = max(values, default=1)*1.15 or 1
            ax.set_xlim(0, min(upper, 1) if key == "average_precision" else upper)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    selected = report["models"][report["selected_family"]]
    event_counts = "; ".join(f"{split}: {selected.get(split, {}).get('pre_onset', {}).get('eligible_events', 'unreported')}" for split in COHORTS)
    caption = (f"Eligible events ({event_counts}); requested bootstrap replicates: {report.get('metadata', {}).get('bootstrap_replicates', 'unreported')}.\n"
               "Points and available 95% patient-cluster percentile intervals; * family selected on A development data.\n"
               "Event sensitivity uses all eligible pre-onset events; false-alert rates use eligible non-warning time and a 6 h cooldown.\n"
               + missing_note(rows) + " Separate marginal intervals do not test differences.")
    footer(fig, caption)
    return save(fig, output, "01_primary_performance", "Model performance across hospital cohorts")


def calibration_figure(report: dict, data: dict, output: Path) -> list[str]:
    fig, (ax, density) = plt.subplots(2, 1, figsize=(8.6, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(left=.13, right=.95, top=.85, bottom=.19, hspace=.11)
    heading(fig, "Calibration and score distribution · hospital B", report)
    ax.plot([0, 1], [0, 1], color="#859197", linewidth=1.1, linestyle="--", label="Perfect calibration")
    populated = [item for item in data["bins"] if item["count"] > 0]
    ax.scatter([item["mean_probability"] for item in populated], [item["observed_fraction"] for item in populated],
               color=COHORTS["external"][1], marker="s", s=40, zorder=3, clip_on=False, label="Observed nonempty bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed positive-label fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid()
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    if not populated:
        ax.text(.5, .45, "Calibration not estimable from this report", transform=ax.transAxes, ha="center")
    for item in populated:
        width = item["upper"]-item["lower"]
        density.bar(item["lower"], item["patient_hour_share"], width=width, align="edge",
                    color="#DCA774", edgecolor="white", linewidth=.8)
        share_label = "<0.01%" if 0 < item["patient_hour_share"] < .0001 else f"{item['patient_hour_share']:.2%}"
        density.annotate(share_label,
                         (item["lower"] + width/2, item["patient_hour_share"]),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
    density.set_ylim(0, max([item["patient_hour_share"] for item in populated] or [1])*1.28)
    density.set_ylabel("Patient-hours")
    density.set_xlabel("Mean predicted score above; score-bin range below")
    density.xaxis.set_major_formatter(PercentFormatter(1))
    density.yaxis.set_major_formatter(PercentFormatter(1))
    density.yaxis.set_major_locator(MaxNLocator(nbins=3))
    density.grid(axis="y")
    density.set_axisbelow(True)
    caption = (f"{data['patient_hours']:,} patient-hours; {len(data['bins'])} supplied score bins, {data['empty_bins']} empty. Empty-bin calibration points are omitted.\n"
               "Distribution bars show the fraction of all reported B patient-hours; bin-level intervals were not supplied.\n"
               "Scores target the supplied shifted hourly label, including post-onset hours; this is not a pure future-onset probability.")
    footer(fig, caption)
    return save(fig, output, "02_external_calibration", "Calibration and score distribution in hospital B")


def stress_figure(report: dict, rows: list[dict], output: Path) -> list[str]:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.7))
    fig.subplots_adjust(left=.105, right=.97, top=.84, bottom=.18, wspace=.28, hspace=.32)
    heading(fig, "Controlled observation stress · frozen selected model", report)
    cohort_legend(fig, report, y=.91)
    kinds = (("missing", "Additional missingness", "Observed cells deleted"),
             ("delay", "Measurement delay", "Arrival delay (hours)"),
             ("noise", "Measurement noise", "Noise SD / training-variable SD"))
    metrics = [item for item in METRICS if item[0] != "challenge_utility"]
    for row_index, (key, _, title, _) in enumerate(metrics):
        values = [item[field] for item in rows if item["metric"] == key
                  for field in ("estimate", "lower", "upper") if item[field] is not None]
        ylim = (0, 1) if key == "event_sensitivity" else (0, max(values, default=1)*1.15 or 1)
        for column, (kind, kind_title, xlabel) in enumerate(kinds):
            ax = axes[row_index, column]
            for split, (_, color, marker) in COHORTS.items():
                points = sorted([item for item in rows if item["metric"] == key and item["kind"] == kind and item["split"] == split],
                                key=lambda item: item["severity"])
                ax.plot([item["severity"] for item in points],
                        [item["estimate"] if item["estimate"] is not None else math.nan for item in points],
                        color=color, linewidth=1.2)
                for item in points:
                    interval_mark(ax, item["severity"], item, color, marker)
            if row_index == 0:
                ax.set_title(kind_title, pad=11)
            if column == 0:
                ax.set_ylabel({"average_precision": "Average precision (0–1)",
                               "event_sensitivity": "Pre-onset event sensitivity",
                               "false_alerts_per_patient_day": "False alerts / eligible\nnon-warning patient-day"}[key])
            if row_index == len(metrics)-1:
                ax.set_xlabel(xlabel, fontsize=9)
            if key == "event_sensitivity":
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            if kind == "missing":
                ax.xaxis.set_major_formatter(PercentFormatter(1))
            severities = sorted({item["severity"] for item in rows if item["kind"] == kind})
            ax.set_xticks(severities)
            ax.set_ylim(*ylim)
            maximum_severity = max(severities, default=1) or 1
            ax.set_xlim(-.025*maximum_severity, 1.05*maximum_severity)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid()
            ax.set_axisbelow(True)
            if not any(item["kind"] == kind and not item["baseline"] for item in rows):
                ax.text(.5, .82, "No stress results supplied", transform=ax.transAxes, ha="center", fontsize=8)
    caption = ("Severity 0 is the unchanged-data baseline. Points and available 95% patient-cluster percentile intervals use frozen models and thresholds.\n"
               "Axes share a scale within each metric row. Perturbations are controlled simulations, not estimates of actual clinical error.\n"
               + missing_note(rows))
    footer(fig, caption)
    return save(fig, output, "03_observation_stress", "Controlled observation stress in the selected model")


def selective_figure(report: dict, data: dict, output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.5))
    fig.subplots_adjust(left=.10, right=.96, top=.81, bottom=.26, wspace=.28)
    heading(fig, "Selective withholding · hospital B", report)
    points = sorted([item for item in data["rows"] if item["coverage"] is not None], key=lambda item: item["coverage"])
    ax = axes[0]
    valid = [item for item in points if item["event_sensitivity_with_withheld_as_no_alert"] is not None]
    ax.plot([item["coverage"] for item in valid], [item["event_sensitivity_with_withheld_as_no_alert"] for item in valid],
            color=COHORTS["external"][1], marker="s", linewidth=1.4, markersize=5)
    ax.set_title("Warnings among all eligible events", loc="left", fontsize=11)
    ax.set_ylabel("Detected / all eligible events")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    if not valid:
        ax.text(.5, .5, "Event sensitivity not estimable", ha="center", transform=ax.transAxes, fontsize=9)
    counts = axes[1]
    for key, label, color, marker, line in (
        ("eligible_events_with_any_withheld_warning_hour", "Any warning hour withheld", "#127D82", "o", "-"),
        ("eligible_events_with_all_warning_hours_withheld", "All warning hours withheld", "#B66322", "s", "--")):
        subset = [item for item in points if item[key] is not None]
        counts.plot([item["coverage"] for item in subset], [item[key] for item in subset],
                    color=color, marker=marker, linestyle=line, linewidth=1.3, markersize=5, label=label)
    denominators = [item["eligible_events"] for item in points if item["eligible_events"] is not None]
    if denominators:
        counts.axhline(denominators[0], linestyle=":", color="#72858C", linewidth=1.1,
                       label=f"All eligible events (n={int(denominators[0])})")
        counts.set_ylim(0, max(denominators[0], 1)*1.12)
    else:
        counts.set_ylim(bottom=0)
    counts.set_title("Events affected by withholding", loc="left", fontsize=11)
    counts.set_ylabel("Eligible events (count)")
    if not points:
        counts.text(.5, .5, "No selective-policy results supplied", ha="center", transform=counts.transAxes, fontsize=9)
    counts.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    fig.legend(*counts.get_legend_handles_labels(), frameon=False, loc="upper left",
               bbox_to_anchor=(.063, .90), fontsize=8.5, ncol=3, columnspacing=1.6)
    for ax in axes:
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Retained patient-hours (coverage)")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid()
        ax.set_axisbelow(True)
    undefined = sum(item["event_sensitivity_with_withheld_as_no_alert"] is None or item["coverage"] is None for item in data["rows"])
    missing_counts = sum(item[key] is None or item["coverage"] is None for item in data["rows"]
                         for key in ("eligible_events_with_any_withheld_warning_hour", "eligible_events_with_all_warning_hours_withheld"))
    caption = ("Withheld predictions count as no alert; sensitivity keeps every eligible event in its denominator.\n"
               "All-hours-withheld counts are a subset of any-hour-withheld counts. Points represent frozen development-selected policies.\n"
               f"{len(data['rows'])} recorded policies; undefined sensitivity/count points omitted: {undefined}/{missing_counts}. No policy intervals supplied; no safety benefit established.")
    footer(fig, caption)
    return save(fig, output, "04_selective_withholding", "Selective withholding and eligible events in hospital B")


def render(report_path: str | Path, output: str | Path) -> dict:
    report_path, output = Path(report_path), Path(output)
    source = report_path.read_bytes()
    report = json.loads(source)
    if not isinstance(report.get("models"), dict) or report.get("selected_family") not in report["models"]:
        raise ValueError("Report needs model evaluations and its recorded selected_family")
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory to preserve existing figures")
    datasets = {"primary": primary_data(report), "calibration": calibration_data(report),
                "stress": stress_data(report), "selective": selective_data(report)}
    output.mkdir(parents=True)
    style()
    files = []
    for name, function in (("primary", primary_figure), ("calibration", calibration_figure),
                           ("stress", stress_figure), ("selective", selective_figure)):
        files += function(report, datasets[name], output)
    evidence = {"source_report_sha256": hashlib.sha256(source).hexdigest(), "source_filename": report_path.name,
                "source_metadata": report.get("metadata", {}), "selected_family": report["selected_family"],
                "definitions": DEFINITIONS, "figure_data": datasets}
    (output / "figure-data.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    manifest = {"source_report_sha256": evidence["source_report_sha256"],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in files + ["figure-data.json"]}}
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    try:
        manifest = render(args.report, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Unable to render report figures: {exc}\n")
    print(f"Rendered 4 figure pairs plus source data and provenance: {args.output}")
    print(f"Report SHA256: {manifest['source_report_sha256']}")


if __name__ == "__main__":
    main()
