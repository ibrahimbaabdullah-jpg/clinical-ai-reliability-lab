"""Reproducible, patient-separated experiment entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import DYNAMIC_COLUMNS, cohort_summary, ingest_directory, make_splits, save_parquet
from .evaluation import choose_threshold, evaluate, selective
from .features import build_features
from .models import fit_calibrator, fit_model
from .perturbations import perturb
from .synthetic import make_synthetic


def clean_json(value):
    if isinstance(value, pd.DataFrame):
        return clean_json(value.to_dict(orient='records'))
    if isinstance(value, pd.Series):
        return clean_json(value.to_dict())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean_json(data), indent=2, allow_nan=False) + '\n')


def versions():
    result = {'python': sys.version.split()[0]}
    for package in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'duckdb', 'pyarrow', 'fastapi', 'mlflow-skinny']:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def verify_download(root):
    """Check recorded official-source provenance and exact local byte identity.

    This is a local integrity check, not publisher-authenticated signing.
    Missing provenance is allowed but must never become a verified-source claim.
    """
    root = Path(root).resolve()
    path = root / 'download_manifest.json'
    if not path.exists():
        return {'verified': False, 'reason': 'No download manifest; user-supplied PhysioNet-format files'}
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'complete' or manifest.get('version') != '1.0.0':
        raise ValueError('Data download is incomplete or has an unsupported source version')
    entries = manifest.get('files', [])
    observed = {str(p.relative_to(root)) for p in root.rglob('*.psv')}
    listed = [r['path'] for r in entries]
    if len(set(listed)) != len(listed) or set(listed) != observed:
        raise ValueError('Patient files do not match the download manifest exactly')
    for row in entries:
        file = (root / row['path']).resolve()
        if not file.is_relative_to(root):
            raise ValueError('Manifest path escapes the data root')
        url = urlparse(row['source_url'])
        official_origin = url.hostname in {'physionet.org', 'www.physionet.org', 'archive.physionet.org'}
        official_mirror = (url.hostname == 'physionet-open.s3.amazonaws.com'
                           and url.path.startswith('/challenge-2019/1.0.0/training/'))
        if url.scheme != 'https' or not (official_origin or official_mirror):
            raise ValueError('Manifest has a nonofficial data source')
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Data checksum mismatch: {row["path"]}')
    return {'verified': True, 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': manifest.get('cohort_scope'), 'integrity_basis': 'Recorded official download URLs and local SHA256, not publisher signatures'}


def fit_bundle(family, frames, features, config, seed, ablation='full'):
    model = fit_model(
        family, features['train'], frames['train'].SepsisLabel.to_numpy(),
        frames['train'].patient_id.to_numpy(), features['tune'],
        frames['tune'].SepsisLabel.to_numpy(), frames['tune'].patient_id.to_numpy(),
        config.get('models', {}), seed=seed,
    )
    raw_cal = model.predict_proba(features['calibrate'], frames['calibrate'].patient_id.to_numpy())
    calibrator = fit_calibrator(raw_cal, frames['calibrate'].SepsisLabel.to_numpy())
    raw_threshold = model.predict_proba(features['threshold'], frames['threshold'].patient_id.to_numpy())
    p = calibrator.transform(raw_threshold)
    threshold = choose_threshold(frames['threshold'], p)
    development = evaluate(frames['threshold'], p, threshold, bootstrap=0, seed=seed)
    return {'model': model, 'calibrator': calibrator, 'threshold': threshold,
            'ablation': ablation, 'family': family}, development, p, raw_threshold


def predict(bundle, df, features=None):
    if features is None:
        features = build_features(df, ablation=bundle['ablation'])
    raw = bundle['model'].predict_proba(features, df.patient_id.to_numpy())
    return bundle['calibrator'].transform(raw)


def render_report(report):
    m = report['metadata']
    lines = ['# Benchmark execution report', '', f"**Evidence status: {m['evidence_status']}**", '',
             f"Run UTC: {m['created_at']}. Patients: {m['patients']}; hours: {m['patient_hours']}.",
             f"Data SHA256: `{m['data_sha256']}`. Protocol SHA256: `{m['protocol_sha256']}`.", '',
             'These are retrospective software/research results. Synthetic results are not clinical evidence. Reduced-cohort results are preliminary, not the complete 40,336-patient benchmark.', '',
             f"Selected family: **{report['selected_family']}**, selected solely using development threshold-partition utility.", '',
             '| Model | Evaluation set | AP | Utility | Brier | Event sensitivity | False alerts/patient-day |',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'undefined' if x is None else f'{x:.4f}'
    for family, results in report['models'].items():
        for split in ['test', 'external']:
            v = results[split]
            a, b, c = v['hourly_metrics'], v['challenge'], v['pre_onset']
            vals = [a['average_precision'], b['challenge_utility'], a['brier'], c['event_sensitivity'], c['false_alerts_per_patient_day']]
            lines.append('| ' + ' | '.join([family, split] + [fmt(x) for x in vals]) + ' |')
    lines += ['', '95% patient-cluster intervals, exact denominators, excluded events, calibration bins, subgroups, stress experiments and selective coverage are in `report.json`.', '',
              '## Interpretation limits', '',
              '- Held-out B represents one hospital transfer, not universal transportability.',
              '- Alert burden and pre-onset analysis use the prespecified six-hour window and six-hour refractory period.',
              '- First-row-positive and unobserved inferred-onset events are excluded only from pre-onset event analysis; counts are reported.',
              '- Perturbations are simulated; they do not identify actual clinical error mechanisms.',
              '- Model contributions describe mathematical association, not causes or treatment recommendations.',
              '- No prospective validation, clinician usability study or patient benefit has been established.', '',
              '## Reproducibility', '', f"Configuration: `{m['config_sha256']}`. Split manifest: `split_manifest.json`. Dependencies: `environment.json`.",
              f"Primary bootstrap replicates: {m['bootstrap_replicates']}. See JSON for valid-replicate counts and undefined results.", '']
    return '\n'.join(lines)


def run(args):
    output = Path(args.output)
    if (output / 'report.json').exists():
        raise ValueError('A completed report already exists here. Choose a new output directory to preserve evidence.')
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config.get('seed', 2019))
    started = time.time()
    print('Loading and validating patient trajectories...', flush=True)
    if args.synthetic:
        df = make_synthetic(int(config.get('synthetic_patients', 360)), seed=seed)
        source = 'synthetic'
        provenance = {'verified': False, 'reason': 'Generated data, not clinical records'}
    else:
        if not args.data:
            raise ValueError('--data is required unless --synthetic is specified')
        root = Path(args.data)
        provenance = verify_download(root)
        df = pd.concat([ingest_directory(root / 'training_setA', 'A'), ingest_directory(root / 'training_setB', 'B')], ignore_index=True)
        source = 'physionet-2019' if provenance['verified'] else 'unverified-physionet-format'
    df = df.sort_values(['patient_id', 'hour']).reset_index(drop=True)
    split_ids = make_splits(df, seed=seed)
    write_json(output / 'split_manifest.json', split_ids)
    config_text = yaml.safe_dump(config, sort_keys=True)
    (output / 'config.yaml').write_text(config_text)
    protocol_text = importlib.resources.files('clinical_lab').joinpath('resources/protocol.md').read_text()
    (output / 'protocol.md').write_text(protocol_text)
    env = versions()
    write_json(output / 'environment.json', env)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    source_hash = hashlib.sha256()
    for source_file in sorted(Path(__file__).parent.rglob('*.py')):
        source_hash.update(str(source_file.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source_file.read_bytes())
    counts = df.groupby('hospital').patient_id.nunique().to_dict()
    complete = counts == {'A': 20336, 'B': 20000} and source == 'physionet-2019' and provenance.get('scope') == 'full'
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'source': source,
                'evidence_status': 'synthetic software demonstration' if args.synthetic else ('complete public cohort retrospective experiment' if complete else 'reduced-cohort preliminary experiment'),
                'complete_public_cohort': complete, 'patients': df.patient_id.nunique(), 'patient_hours': len(df),
                'patients_by_hospital': counts, 'bootstrap_replicates': int(config.get('bootstrap', 1000)),
                'data_sha256': data_hash, 'protocol_sha256': hashlib.sha256(protocol_text.encode()).hexdigest(),
                'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
                'split_sha256': hashlib.sha256((output / 'split_manifest.json').read_bytes()).hexdigest(),
                'code_revision': revision, 'source_tree_sha256': source_hash.hexdigest(), 'data_provenance': provenance, 'environment': env, 'seed': seed}
    write_json(output / 'run_manifest.json', metadata)
    save_parquet(df, output / 'cohort.parquet')
    write_json(output / 'cohort_summary.json', cohort_summary(output / 'cohort.parquet', use_duckdb=True))
    frames = {name: df[df.patient_id.isin(ids)].reset_index(drop=True) for name, ids in split_ids.items()}
    # Keep external features out of model development and selection.
    features = {name: build_features(frames[name]) for name in ['train', 'tune', 'calibrate', 'threshold']}
    bundles, dev, threshold_probabilities = {}, {}, {}
    for family in config.get('families', ['logistic', 'xgboost', 'gru']):
        print(f'Training {family} on hospital A only...', flush=True)
        bundle, metrics, p_threshold, raw_threshold = fit_bundle(family, frames, features, config, seed)
        bundles[family], dev[family] = bundle, metrics
        threshold_probabilities[family] = (p_threshold, raw_threshold)
        joblib.dump(bundle, output / f'{family}.joblib')
        write_json(output / f'{family}_development.json', {'params': bundle['model'].params, 'threshold': bundle['threshold'], 'evaluation': metrics})
    complexity = {'logistic': 0, 'xgboost': 1, 'gru': 2}
    def selection_key(family):
        utility = dev[family]['challenge']['challenge_utility']
        return (float('-inf') if utility is None else utility, -complexity[family])
    selected = max(bundles, key=selection_key)
    selection = {'selected_family': selected, 'criterion': 'threshold-partition official utility; complexity tie break',
                 'frozen_before_test_evaluation': True, 'development_utility': {k: v['challenge']['challenge_utility'] for k, v in dev.items()}}
    write_json(output / 'selection.json', selection)
    print(f'Frozen model selection: {selected}. Beginning held-out evaluation.', flush=True)
    features.update({name: build_features(frames[name]) for name in ['test', 'external']})
    report = {'metadata': metadata, 'selected_family': selected, 'selection': selection,
              'splits': {name: {'patients': len(ids), 'hours': len(frames[name])} for name, ids in split_ids.items()},
              'models': {}, 'stress_tests': [], 'ablations': {}, 'selective_prediction': {}}
    bootstrap = int(config.get('bootstrap', 1000))
    for family, bundle in bundles.items():
        result = {'parameters': bundle['model'].params, 'threshold': bundle['threshold'], 'development': dev[family]}
        for split in ['test', 'external']:
            p = predict(bundle, frames[split], features[split])
            result[split] = evaluate(frames[split], p, bundle['threshold'], bootstrap=bootstrap, seed=seed)
            prediction_frame = frames[split][['patient_id', 'hospital', 'hour', 'SepsisLabel']].copy()
            prediction_frame['probability'] = p
            prediction_frame.to_parquet(output / f'{family}_{split}_predictions.parquet', index=False)
            development_distances = abs(threshold_probabilities[family][0] - bundle['threshold'])
            cutoffs = sorted(set([0.] + [float(np.quantile(development_distances, q)) for q in [.1, .25, .5]]))
            report['selective_prediction'][f'{family}_{split}'] = selective(frames[split], p, bundle['threshold'], cutoffs=cutoffs)
        report['models'][family] = result
    bundle = bundles[selected]
    scales = frames['train'][DYNAMIC_COLUMNS].std(ddof=0).fillna(0).to_dict()
    write_json(output / 'training_noise_scales.json', scales)
    if config.get('stress', True):
        for kind, severities in [('missing', [.1, .3, .5]), ('delay', [1, 3, 6]), ('noise', [.1, .25, .5])]:
            for severity in severities:
                print(f'Stress {kind}: {severity}', flush=True)
                for split in ['test', 'external']:
                    changed = perturb(frames[split], kind, severity, seed=seed, scales=scales)
                    p = predict(bundle, changed)
                    report['stress_tests'].append({'family': selected, 'split': split, 'kind': kind, 'severity': severity,
                                                  'evaluation': evaluate(frames[split], p, bundle['threshold'], bootstrap=bootstrap, seed=seed)})
    if config.get('ablations', True):
        # Prespecified logistic ablations give an interpretable component study.
        for ablation in ['no_trends', 'no_missingness', 'values_only']:
            print(f'Logistic component ablation: {ablation}', flush=True)
            f = {name: build_features(frame, ablation=ablation) for name, frame in frames.items()}
            candidate, metrics, _, _ = fit_bundle('logistic', frames, f, config, seed, ablation=ablation)
            report['ablations'][ablation] = {'family': 'logistic', 'development': metrics, 'threshold': candidate['threshold']}
            for split in ['test', 'external']:
                p = predict(candidate, frames[split], f[split])
                report['ablations'][ablation][split] = evaluate(frames[split], p, candidate['threshold'], bootstrap=bootstrap, seed=seed)
        raw_threshold = threshold_probabilities[selected][1]
        threshold = choose_threshold(frames['threshold'], raw_threshold)
        report['ablations']['no_calibration'] = {'family': selected, 'threshold': threshold}
        for split in ['test', 'external']:
            p = bundle['model'].predict_proba(features[split], frames[split].patient_id.to_numpy())
            report['ablations']['no_calibration'][split] = evaluate(frames[split], p, threshold, bootstrap=bootstrap, seed=seed)
    report['metadata']['elapsed_seconds'] = round(time.time() - started, 2)
    write_json(output / 'report.json', report)
    (output / 'report.md').write_text(render_report(report))
    if config.get('tracking', False):
        import mlflow
        mlflow.set_tracking_uri('sqlite:///' + str((output.parent / 'mlflow.db').resolve()))
        mlflow.set_experiment('clinical-ai-reliability-lab')
        with mlflow.start_run(run_name=output.name):
            mlflow.log_params({'source': source, 'selected_family': selected, 'seed': seed, 'data_sha256': data_hash})
            for family, result in report['models'].items():
                for split in ['test', 'external']:
                    for key in ['average_precision', 'brier']:
                        value = result[split]['hourly_metrics'][key]
                        if value is not None:
                            mlflow.log_metric(f'{family}_{split}_{key}', value)
            for name in ['report.json', 'report.md', 'run_manifest.json', 'environment.json', 'config.yaml', 'selection.json']:
                mlflow.log_artifact(str(output / name))
    print(f'Completed: {output / "report.md"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('run', help='Train and evaluate a frozen retrospective experiment')
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--data', help='Directory containing training_setA and training_setB')
    command.add_argument('--config', default='configs/demo.yaml')
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Experiment stopped: {exc}\n')


if __name__ == '__main__':
    main()
