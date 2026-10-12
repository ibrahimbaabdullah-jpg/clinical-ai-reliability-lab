#!/usr/bin/env python3
"""Package only aggregate research evidence and trusted fitted weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def package(run: Path, output: Path):
    if output.exists():
        raise ValueError('Choose a new release directory; do not overwrite evidence')
    report = json.loads((run / 'report.json').read_text())
    files = ['report.json', 'report.md', 'run_manifest.json', 'config.yaml',
             'protocol.md', 'environment.json', 'selection.json',
             'training_noise_scales.json', 'cohort_summary.json']
    files += [f'{family}.joblib' for family in report['models']]
    missing = [name for name in files if not (run / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete run: {missing}')
    output.mkdir(parents=True)
    checksums = {}
    for name in files:
        shutil.copy2(run / name, output / name)
        checksums[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / 'SHA256SUMS.json').write_text(json.dumps(checksums, indent=2) + '\n')
    return {'output': str(output.resolve()), 'files': len(files),
            'evidence_status': report['metadata']['evidence_status']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.output), indent=2))
