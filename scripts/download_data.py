#!/usr/bin/env python3
"""Fetch the public PhysioNet 2019 A/B cohorts without credentials.

Only Python's standard library is required. Clinical records stay outside Git.
An archive is used only when explicitly supplied or advertised by PhysioNet;
otherwise the current official per-hospital directory indexes are used.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile

try:
    import requests
except ImportError:  # Optional; stdlib-only use remains supported.
    requests = None

VERSION = "1.0.0"
LANDING_URL = f"https://physionet.org/content/challenge-2019/{VERSION}/"
BASE_URL = f"https://physionet.org/files/challenge-2019/{VERSION}/"
EXPECTED = {"A": 20336, "B": 20000}
LICENSE_NOTE = (
    "The landing page identifies CC BY 4.0, while its downloadable LICENSE.txt "
    "contains ODbL. Both source notices are retained; no resolution of this "
    "discrepancy or permission to relicense the data is asserted."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def official_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "physionet.org", "www.physionet.org", "archive.physionet.org"
    } or parsed.username or parsed.password:
        raise ValueError(f"Only HTTPS URLs on official PhysioNet hosts are accepted: {url}")
    return url


class Fetcher:
    """Small shared client: bounded retries and a shared Retry-After cooldown."""

    def __init__(self, retries: int = 4, timeout: int = 45) -> None:
        self.retries = retries
        self.timeout = timeout
        self.lock = threading.Lock()
        self.resume_at = 0.0
        self.local = threading.local()

    @contextmanager
    def _response(self, url):
        headers = {"User-Agent": "ClinicalAIReliabilityLab/0.1 (research downloader)", "Accept-Encoding": "identity"}
        if requests is None:
            with urlopen(Request(url, headers=headers), timeout=self.timeout) as response:
                yield response
            return
        # One session per worker avoids sharing mutable Session state while
        # retaining TLS connections across that worker's consecutive files.
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        try:
            with self.local.session.get(url, headers=headers, timeout=self.timeout, stream=True) as response:
                if response.status_code >= 400:
                    raise HTTPError(url, response.status_code, response.reason, response.headers, None)
                yield SimpleNamespace(url=response.url, headers=response.headers, read=response.raw.read)
        except (requests.RequestException, requests.packages.urllib3.exceptions.HTTPError) as error:
            raise OSError(str(error)) from error

    def _pause(self) -> None:
        while True:
            with self.lock:
                remaining = self.resume_at - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def download(self, url: str, destination: Path) -> dict:
        official_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        for attempt in range(self.retries + 1):
            self._pause()
            try:
                with self._response(url) as response, temporary.open("wb") as target:
                    official_url(response.url)
                    h = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        target.write(block)
                        h.update(block)
                        size += len(block)
                    length = response.headers.get("Content-Length")
                    if length and size != int(length):
                        raise OSError(f"Incomplete response: expected {length}, received {size}")
                os.replace(temporary, destination)
                return {"source_url": url, "retrieved_at_utc": utc_now(), "sha256": h.hexdigest(), "bytes": size}
            except (HTTPError, URLError, OSError, TimeoutError) as error:
                temporary.unlink(missing_ok=True)
                if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == self.retries:
                    raise
                delay = min(30.0, 2.0 ** attempt)
                if isinstance(error, HTTPError) and error.headers.get("Retry-After"):
                    value = error.headers["Retry-After"]
                    try:
                        requested = float(value)
                    except ValueError:
                        try:
                            requested = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError):
                            requested = 0.0
                    # Respect the whole advertised delay; no early retry on 429.
                    delay = max(delay, requested)
                with self.lock:
                    self.resume_at = max(self.resume_at, time.monotonic() + delay)
        raise RuntimeError("Download retry loop exhausted")


def select_names(names: list[str], limit: int | None, seed: int, hospital: str) -> list[str]:
    names = sorted(names)
    if limit is None or limit >= len(names):
        return names
    return sorted(random.Random(f"{seed}:{hospital}").sample(names, limit))


def safe_zip_members(archive: zipfile.ZipFile) -> dict[str, dict[str, zipfile.ZipInfo]]:
    """Validate *every* member before extracting only normalized patient paths."""
    records: dict[str, dict[str, zipfile.ZipInfo]] = {"A": {}, "B": {}}
    total_size = 0
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (
            path.is_absolute() or ".." in path.parts or "\\" in member.filename
            or re.match(r"^[A-Za-z]:", member.filename) or stat.S_ISLNK(mode)
            or (member.flag_bits & 1)
        ):
            raise ValueError(f"Unsafe or encrypted ZIP member: {member.filename}")
        total_size += member.file_size
        if member.file_size > 128 * 1024 ** 2 or total_size > 4 * 1024 ** 3:
            raise ValueError("Archive exceeds the documented extraction size limit")
        if member.is_dir() or not path.name.endswith(".psv"):
            continue
        hospital = next((h for h in EXPECTED if f"training_set{h}" in path.parts), None)
        if hospital is None:
            continue
        if not re.fullmatch(r"p\d+\.psv", path.name):
            raise ValueError(f"Unexpected patient filename: {member.filename}")
        if path.name in records[hospital]:
            raise ValueError(f"Duplicate patient in ZIP: {hospital}/{path.name}")
        records[hospital][path.name] = member
    if any(not records[h] for h in EXPECTED):
        raise ValueError("Archive must contain both training_setA and training_setB")
    return records


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit-per-hospital", type=int, help="Uniform seeded subset; omitting this fetches all patients")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--source", choices=("auto", "files", "archive"), default="auto")
    parser.add_argument("--archive-url", help="Exact whole-dataset ZIP URL on an official PhysioNet host")
    args = parser.parse_args(argv)
    if args.limit_per_hospital is not None and args.limit_per_hospital < 1:
        parser.error("--limit-per-hospital must be at least 1")
    if args.source == "files" and args.archive_url:
        parser.error("--archive-url cannot be combined with --source files")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_records = {item["path"]: item for item in previous.get("files", [])}
    fetcher = Fetcher()
    manifest = {
        "schema_version": 1, "dataset": "PhysioNet/CinC Challenge 2019", "version": VERSION,
        "landing_url": LANDING_URL, "run_started_at_utc": utc_now(), "status": "in_progress",
        "license_note": LICENSE_NOTE, "seed": args.seed, "limit_per_hospital": args.limit_per_hospital,
        "selection": "Uniform sampling of filenames without replacement; labels are not consulted",
        "files": [], "source_notices": [], "expected_full_patient_counts": EXPECTED,
    }
    try:
        notices = output / ".provenance"
        for url, name in [(LANDING_URL, "dataset-landing.html"), (BASE_URL + "LICENSE.txt", "LICENSE.txt"),
                          (BASE_URL + "SHA256SUMS.txt", "upstream-SHA256SUMS.txt")]:
            record = fetcher.download(url, notices / name)
            record["path"] = str((notices / name).relative_to(output))
            manifest["source_notices"].append(record)

        archive_url = args.archive_url
        if archive_url is None and args.source != "files":
            links = Links()
            links.feed((notices / "dataset-landing.html").read_text(encoding="utf-8"))
            candidates = sorted({urljoin(LANDING_URL, link) for link in links.links
                                 if "challenge-2019" in link and urlparse(link).path.endswith(".zip")
                                 and "/static/published-projects/" in link})
            archive_url = candidates[0] if len(candidates) == 1 else None
        if args.source == "archive" and archive_url is None:
            raise ValueError("No unique whole ZIP advertised; specify --archive-url or use --source files")

        counts = {}
        if archive_url:
            archive_url = official_url(archive_url)
            archive_path = output / ".archives" / "challenge-2019-1.0.0.zip"
            old_archive = previous.get("archive", {})
            if (archive_path.is_file() and old_archive.get("source_url") == archive_url
                    and old_archive.get("sha256") == digest_file(archive_path)):
                manifest["archive"] = {**old_archive, "reused_verified_at_utc": utc_now()}
            else:
                manifest["archive"] = fetcher.download(archive_url, archive_path)
            manifest["source_route"] = "official_archive"
            with zipfile.ZipFile(archive_path) as archive:
                members = safe_zip_members(archive)
                for hospital, group in members.items():
                    counts[hospital] = len(group)
                    for name in select_names(list(group), args.limit_per_hospital, args.seed, hospital):
                        destination = output / f"training_set{hospital}" / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name(name + ".part")
                        with archive.open(group[name]) as source, temporary.open("wb") as target:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                target.write(block)
                        os.replace(temporary, destination)
                        manifest["files"].append({
                            "hospital": hospital, "path": str(destination.relative_to(output)),
                            "source_url": archive_url, "archive_member": group[name].filename,
                            "retrieved_at_utc": manifest["archive"]["retrieved_at_utc"],
                            "sha256": digest_file(destination), "bytes": destination.stat().st_size,
                        })
        else:
            manifest["source_route"] = "official_patient_files"
            jobs = []
            for hospital in EXPECTED:
                url = BASE_URL + f"training/training_set{hospital}/"
                index_path = notices / f"training_set{hospital}-index.html"
                index_record = fetcher.download(url, index_path)
                index_record["path"] = str(index_path.relative_to(output))
                manifest["source_notices"].append(index_record)
                links = Links()
                links.feed(index_path.read_text(encoding="utf-8"))
                names = sorted({unquote(link) for link in links.links if re.fullmatch(r"p\d+\.psv", link)})
                if not names:
                    raise ValueError(f"No patient files in official index for hospital {hospital}")
                counts[hospital] = len(names)
                for name in select_names(names, args.limit_per_hospital, args.seed, hospital):
                    jobs.append((hospital, name, urljoin(url, name)))

            def get_patient(job):
                hospital, name, url = job
                destination = output / f"training_set{hospital}" / name
                path = str(destination.relative_to(output))
                old = old_records.get(path, {})
                if destination.is_file() and old.get("source_url") == url and old.get("sha256") == digest_file(destination):
                    return {**old, "reused_verified_at_utc": utc_now()}
                record = fetcher.download(url, destination)
                return {**record, "hospital": hospital, "path": path}

            print(f"Downloading {len(jobs)} patient files with {args.workers} workers", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(get_patient, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        manifest["files"].append(future.result())
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise
                    if len(manifest["files"]) % 100 == 0:
                        atomic_json(manifest_path, manifest)
                        print(f"Verified {len(manifest['files'])}/{len(jobs)} patient files", flush=True)

        if counts != EXPECTED:
            raise ValueError(f"Official cohort count mismatch: observed {counts}; expected {EXPECTED}")
        manifest["available_patient_counts"] = counts
        selected = {h: sum(r["hospital"] == h for r in manifest["files"]) for h in EXPECTED}
        manifest["selected_patient_counts"] = selected
        manifest["cohort_scope"] = "full" if selected == EXPECTED else "reduced"
        # Mixing old full/subset files silently would defeat the selection manifest.
        intended = {r["path"] for r in manifest["files"]}
        extras = [str(path.relative_to(output)) for h in EXPECTED
                  for path in (output / f"training_set{h}").glob("*.psv")
                  if str(path.relative_to(output)) not in intended]
        if extras:
            raise ValueError(f"Output contains {len(extras)} unselected patient files; use a clean --output directory")
        manifest["files"].sort(key=lambda x: x["path"])
        canonical = "\n".join(f"{r['path']}:{r['sha256']}" for r in manifest["files"])
        manifest["dataset_manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        manifest["status"] = "complete"
        manifest["run_completed_at_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"Complete: {selected}; scope={manifest['cohort_scope']}; manifest={manifest_path}", flush=True)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["run_failed_at_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"Download failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
