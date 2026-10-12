#!/usr/bin/env python3
"""Run the experiment and retain Linux child-process wall time and peak RSS."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--telemetry', required=True)
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = [sys.executable, '-u', '-m', 'clinical_lab.cli'] + args.arguments
started = time.monotonic()
result = subprocess.run(command, check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
metrics = {'command': command, 'exit_code': result.returncode,
           'elapsed_seconds': round(time.monotonic() - started, 3),
           'peak_child_rss_kib': usage.ru_maxrss,
           'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime,
           'platform': sys.platform,
           'note': 'ru_maxrss units are KiB on Linux; excludes filesystem cache and other processes.'}
path = Path(args.telemetry)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics), flush=True)
sys.exit(result.returncode)
