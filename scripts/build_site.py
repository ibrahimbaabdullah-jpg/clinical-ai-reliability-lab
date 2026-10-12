#!/usr/bin/env python3
"""Build a self-contained research dashboard from a trusted completed run.

Only fitted model parameters, aggregate metrics, generated patient trajectories,
and project documentation enter the output. Never copy the source run wholesale.
The output path must be a new directory to preserve earlier releases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clinical_lab.export_models import export_browser_run


def build(run: Path, output: Path):
    if output.exists():
        raise ValueError('Choose a new output directory; existing releases are preserved')
    output.mkdir(parents=True)
    result = export_browser_run(run, output / 'data')
    web = ROOT / 'clinical_lab' / 'web'
    static = output / 'static'
    static.mkdir()
    for name in ['app.js', 'style.css', 'inference.js', 'site-adapter.js']:
        shutil.copy2(web / name, static / name)
    html = (web / 'index.html').read_text()
    html = html.replace('<html lang="en">', '<html lang="en" data-runtime="static" data-base="/data">')
    html = html.replace('<script src="/static/app.js" defer></script>',
                        '<script type="module" src="/static/site-adapter.js"></script>')
    html = html.replace('<head>', '<head><meta name="description" content="Compare ICU sepsis models, hospital transfer and observation-quality stress tests. Retrospective research with synthetic patient replay.">'
                        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\'; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data:; connect-src \'self\'; object-src \'none\'; base-uri \'none\'">')
    (output / 'index.html').write_text(html)
    docs = output / 'docs'
    docs.mkdir()
    for name in ['protocol.md', 'model_card.md', 'data_provenance.md', 'user_guide_ar.md']:
        path = ROOT / 'docs' / name
        if path.is_file():
            shutil.copy2(path, docs / name)
    forbidden = [p for p in output.rglob('*') if p.suffix in {'.joblib', '.parquet', '.psv', '.pkl'}]
    if forbidden:
        raise ValueError('Unexpected patient data or executable model artifact in site output')
    return {'output': str(output.resolve()), 'export': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.output), indent=2))
