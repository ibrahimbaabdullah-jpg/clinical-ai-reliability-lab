#!/usr/bin/env python3
"""Write aggregate-only source characteristics from a canonical cohort Parquet.

This is descriptive source profiling, not model evaluation. Run on a full
external cohort only after the model-development choices have been frozen.
No patient identifiers, individual rows, or trajectories are exported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import duckdb

from clinical_lab.data import CANONICAL_COLUMNS, VARIABLES


PATIENT_WEIGHTED = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
MIN_QUANTILE_N = 5


def _rows(connection, query):
    cursor = connection.execute(query)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _quantiles(values, n):
    if n < MIN_QUANTILE_N or values is None:
        return {"q1": None, "median": None, "q3": None, "iqr": None,
                "small_sample_suppressed": n < MIN_QUANTILE_N}
    q1, median, q3 = (float(value) for value in values)
    return {"q1": q1, "median": median, "q3": q3, "iqr": q3 - q1,
            "small_sample_suppressed": False}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_cohort(cohort: str | Path) -> dict:
    """Return source aggregates with bounded DuckDB threads and working memory."""
    cohort = Path(cohort)
    if not cohort.is_file():
        raise ValueError("The canonical cohort Parquet file does not exist.")
    with tempfile.TemporaryDirectory(prefix="clinical-cohort-profile-") as temporary:
        with duckdb.connect(config={"threads": 2, "memory_limit": "1GB",
                                    "temp_directory": temporary}) as connection:
            connection.read_parquet(str(cohort)).create_view("source_cohort")
            schema = _rows(connection, "DESCRIBE source_cohort")
            missing = set(CANONICAL_COLUMNS) - {row["column_name"] for row in schema}
            if missing:
                raise ValueError(f"Missing canonical fields: {sorted(missing)}")
            source = _rows(connection, """
                SELECT COUNT(*) AS patient_hours,
                       COUNT(*) FILTER (WHERE hospital IS NULL OR hospital NOT IN ('A', 'B')
                           OR patient_id IS NULL OR patient_id = '' OR hour IS NULL
                           OR NOT isfinite(hour) OR hour < 0 OR hour != floor(hour)
                           OR SepsisLabel IS NULL OR SepsisLabel NOT IN (0, 1)) AS invalid_rows
                FROM source_cohort
            """)[0]
            if source["patient_hours"] == 0 or source["invalid_rows"]:
                raise ValueError("The cohort is empty or contains invalid record/time/outcome metadata.")
            first_values = ",\n".join(
                f'arg_min("{column}", hour) FILTER (WHERE isfinite("{column}")) AS "{column}"'
                for column in PATIENT_WEIGHTED
            )
            connection.execute(f"""
                CREATE TEMP TABLE patient_summary AS
                SELECT hospital, patient_id, COUNT(*) AS patient_hours,
                       MAX(SepsisLabel) AS ever_positive, {first_values}
                FROM source_cohort GROUP BY hospital, patient_id
            """)
            hospitals = {}
            for row in _rows(connection, """
                SELECT hospital, COUNT(*) AS patient_records,
                       SUM(patient_hours) AS patient_hours,
                       SUM(ever_positive) AS ever_positive_patients,
                       AVG(ever_positive) AS patient_sepsis_prevalence,
                       MIN(patient_hours) AS stay_min, MAX(patient_hours) AS stay_max,
                       quantile_cont(patient_hours, [0.25, 0.5, 0.75]) AS stay_quantiles,
                       COUNT(Age) AS age_observed_patients,
                       quantile_cont(Age, [0.25, 0.5, 0.75]) AS age_quantiles,
                       COUNT(*) FILTER (WHERE Age IS NULL) AS age_missing,
                       COUNT(*) FILTER (WHERE Age < 18) AS age_lt18,
                       COUNT(*) FILTER (WHERE Age >= 18 AND Age < 40) AS age_18_39,
                       COUNT(*) FILTER (WHERE Age >= 40 AND Age < 65) AS age_40_64,
                       COUNT(*) FILTER (WHERE Age >= 65 AND Age < 80) AS age_65_79,
                       COUNT(*) FILTER (WHERE Age >= 80 AND Age < 90) AS age_80_89,
                       COUNT(*) FILTER (WHERE Age = 100) AS age_coded100,
                       COUNT(*) FILTER (WHERE Age >= 90 AND Age != 100) AS age_ge90_other,
                       COUNT(*) FILTER (WHERE Gender = 0) AS gender_0,
                       COUNT(*) FILTER (WHERE Gender = 1) AS gender_1,
                       COUNT(*) FILTER (WHERE Gender IS NULL) AS gender_missing,
                       COUNT(*) FILTER (WHERE Gender IS NOT NULL AND Gender NOT IN (0, 1)) AS gender_other
                FROM patient_summary GROUP BY hospital ORDER BY hospital
            """):
                count = int(row["patient_records"])
                hospitals[row["hospital"]] = {
                    "patient_records": count,
                    "patient_hours": int(row["patient_hours"]),
                    "ever_positive_patients": int(row["ever_positive_patients"]),
                    "patient_sepsis_prevalence": float(row["patient_sepsis_prevalence"]),
                    "recorded_stay_hours": {
                        "minimum": int(row["stay_min"]) if count >= MIN_QUANTILE_N else None,
                        "maximum": int(row["stay_max"]) if count >= MIN_QUANTILE_N else None,
                        **_quantiles(row["stay_quantiles"], count),
                    },
                    "age": {
                        "observed_patients": int(row["age_observed_patients"]),
                        "missing_patients": int(row["age_missing"]),
                        **_quantiles(row["age_quantiles"], int(row["age_observed_patients"])),
                        "counts": {key: int(row[key]) for key in (
                            "age_lt18", "age_18_39", "age_40_64", "age_65_79", "age_80_89",
                            "age_coded100", "age_ge90_other", "age_missing")},
                    },
                    "recorded_gender_counts": {key: int(row[key]) for key in (
                        "gender_0", "gender_1", "gender_missing", "gender_other")},
                    "predictors": [],
                }
            # Four quantile states per batch prevent 40 full-length numeric
            # arrays accumulating at once. Parquet column pruning bounds reads.
            for start in range(0, len(VARIABLES), 4):
                columns = VARIABLES[start:start + 4]
                aggregations = []
                for index, column in enumerate(columns):
                    aggregations += [
                        f'COUNT(*) FILTER (WHERE isfinite("{column}")) AS n_{index}',
                        f'COUNT(*) FILTER (WHERE isinf("{column}")) AS inf_{index}',
                        f'quantile_cont("{column}", [0.25, 0.5, 0.75]) '
                        f'FILTER (WHERE isfinite("{column}")) AS q_{index}',
                    ]
                query = "SELECT hospital, " + ", ".join(aggregations)
                query += " FROM source_cohort GROUP BY hospital ORDER BY hospital"
                for row in _rows(connection, query):
                    hospital = hospitals[row["hospital"]]
                    for index, column in enumerate(columns):
                        if row[f"inf_{index}"]:
                            raise ValueError(f"Infinite numeric values in predictor {column}; use a validated canonical cohort.")
                        observed = int(row[f"n_{index}"])
                        hospital["predictors"].append({
                            "variable": column,
                            "observed_patient_hours": observed,
                            "missing_patient_hours": hospital["patient_hours"] - observed,
                            "missing_hour_fraction": 1.0 - observed / hospital["patient_hours"],
                            "distribution_weighting": "hourly_observed",
                            "distribution_n": observed,
                            **_quantiles(row[f"q_{index}"], observed),
                        })
            aggregations = []
            for index, column in enumerate(PATIENT_WEIGHTED):
                aggregations += [f'COUNT("{column}") AS n_{index}',
                                 f'quantile_cont("{column}", [0.25, 0.5, 0.75]) AS q_{index}']
            for row in _rows(connection, "SELECT hospital, " + ", ".join(aggregations)
                             + " FROM patient_summary GROUP BY hospital ORDER BY hospital"):
                by_variable = {entry["variable"]: entry for entry in hospitals[row["hospital"]]["predictors"]}
                for index, column in enumerate(PATIENT_WEIGHTED):
                    n = int(row[f"n_{index}"])
                    by_variable[column].update({
                        "distribution_weighting": "patient_first_nonmissing",
                        "distribution_n": n,
                        **_quantiles(row[f"q_{index}"], n),
                    })
    return {
        "report_type": "aggregate_source_cohort_characteristics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_parquet_sha256": _sha256(cohort),
        "patient_records": sum(hospital["patient_records"] for hospital in hospitals.values()),
        "patient_hours": int(source["patient_hours"]),
        "duckdb_version": duckdb.__version__,
        "resources": {"threads": 2, "memory_limit": "1GB", "predictors_per_quantile_batch": 4},
        "definitions": {
            "records": "Released patient records, not independently linked unique people or admissions.",
            "stay_length": "Number of recorded hourly rows; not reconstructed total hospital or ICU stay.",
            "event": "Ever positive supplied SepsisLabel; not the separate eligible pre-onset event count.",
            "missingness": "Fraction of all recorded hourly rows with NULL or NaN; infinite values are rejected.",
            "quantiles": "Exact continuous 25th, 50th, and 75th percentiles; IQR is Q3 minus Q1.",
            "weighting": "Hourly observed values except Age, Gender, Unit1, Unit2 and HospAdmTime, which use each record's first nonmissing value. ICULOS remains hourly because it changes over time.",
            "age": "Source Age=100 represents the >=90 category, not exact age 100. Numeric summaries retain source coding and must be interpreted accordingly.",
            "gender": "Source recorded codes 0/1, plus missing/other; not inferred gender identity.",
            "small_samples": f"Quantiles and stay extrema are suppressed when fewer than {MIN_QUANTILE_N} relevant observations/records are available; aggregate denominators remain reported.",
            "scope": "Source characteristics only. No model scores, performance, raw identifiers, or individual trajectories are exported. Cohort scope is determined by the accompanying run manifest.",
        },
        "hospitals": hospitals,
    }


def render_markdown(profile: dict) -> str:
    def value(number):
        return "not reported" if number is None else f"{number:,.4g}"

    lines = ["# Aggregate cohort characteristics", "",
             "Descriptive source profile; no model performance or individual records are included.", "",
             f"Patient records: **{profile['patient_records']:,}**. Recorded hours: **{profile['patient_hours']:,}**.",
             f"Parquet SHA256: `{profile['cohort_parquet_sha256']}`.", "",
             "Full versus reduced cohort status must be taken from the accompanying run manifest.", "",
             "## Hospital-level population", "",
             "| Hospital | Records | Recorded hours | Ever-positive records | Prevalence | Stay minimum | Stay median [Q1, Q3] | Stay maximum |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for name, hospital in profile["hospitals"].items():
        stay = hospital["recorded_stay_hours"]
        lines.append(f"|{name}|{hospital['patient_records']:,}|{hospital['patient_hours']:,}|"
                     f"{hospital['ever_positive_patients']:,}|{hospital['patient_sepsis_prevalence']:.2%}|"
                     f"{value(stay['minimum'])}|{value(stay['median'])} [{value(stay['q1'])}, {value(stay['q3'])}]|{value(stay['maximum'])}|")
    lines += ["", "Stay duration is recorded rows, not total hospital duration. Ever-positive labels differ from eligible pre-onset events.", "",
              "## Patient-weighted demographics", "",
              "Age uses the first nonmissing value per record. Source Age=100 denotes the >=90 category; it is not an exact age. Recorded Gender codes do not establish gender identity.", ""]
    for name, hospital in profile["hospitals"].items():
        age = hospital["age"]
        lines += [f"### Hospital {name}", "",
                  f"Age median [Q1, Q3]: {value(age['median'])} [{value(age['q1'])}, {value(age['q3'])}]; missing: {age['missing_patients']:,}.", "",
                  "| Age category | Records |", "|---|---:|"]
        labels = {"age_lt18": "<18", "age_18_39": "18–39", "age_40_64": "40–64",
                  "age_65_79": "65–79", "age_80_89": "80–89", "age_coded100": ">=90 coded 100",
                  "age_ge90_other": ">=90 other coding", "age_missing": "Missing"}
        lines += [f"|{labels[key]}|{count:,}|" for key, count in age["counts"].items()]
        lines += ["", "| Recorded Gender | Records |", "|---|---:|"]
        labels = {"gender_0": "Code 0", "gender_1": "Code 1", "gender_missing": "Missing", "gender_other": "Other code"}
        lines += [f"|{labels[key]}|{count:,}|" for key, count in hospital["recorded_gender_counts"].items()]
        lines += ["", f"### Hospital {name}: predictor availability and distributions", "",
                  "Missingness always uses hourly rows. Distribution counts/quantiles are hour-weighted except the five demographic/administrative fields marked patient. ICULOS is hourly.", "",
                  "| Predictor | Missing hours | Missing fraction | Distribution unit | Observations | Median | Q1 | Q3 | IQR |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for entry in hospital["predictors"]:
            unit = "patient" if entry["distribution_weighting"] == "patient_first_nonmissing" else "hour"
            lines.append(f"|{entry['variable']}|{entry['missing_patient_hours']:,}|{entry['missing_hour_fraction']:.2%}|{unit}|{entry['distribution_n']:,}|"
                         + "|".join(value(entry[key]) for key in ("median", "q1", "q3", "iqr")) + "|")
        lines.append("")
    lines += ["## Definitions and limits", ""]
    lines += [f"- **{key}:** {definition}" for key, definition in profile["definitions"].items()]
    lines += ["", "No identifier or row-level table is written. DuckDB uses two threads and a 1 GB memory limit; predictor quantiles are processed in batches of four.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="Validated run/cohort.parquet")
    parser.add_argument("--output", required=True, type=Path, help="New output directory for aggregate JSON and Markdown")
    args = parser.parse_args()
    paths = [args.output / "cohort_profile.json", args.output / "cohort_profile.md"]
    if any(path.exists() for path in paths):
        parser.error("Profile output already exists; choose a new directory to preserve provenance.")
    report = profile_cohort(args.cohort)
    args.output.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths[1].write_text(render_markdown(report), encoding="utf-8")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
