#!/usr/bin/env python3
"""Export trusted trained artifacts into portable JSON (no Python server needed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_lab.export_models import export_browser_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Trusted local trained run directory')
    parser.add_argument('--output', required=True, help='Destination for JSON assets')
    parser.add_argument('--synthetic-patients', type=int, default=120)
    parser.add_argument('--seed', type=int, default=2019)
    args = parser.parse_args()
    result = export_browser_run(args.run, args.output, synthetic_patients=args.synthetic_patients, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
