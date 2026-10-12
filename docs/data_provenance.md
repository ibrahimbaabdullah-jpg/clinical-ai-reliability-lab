# Data provenance and acquisition

The research dataset is **Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019**, version **1.0.0**, DOI [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857). Its public cohorts contain 20,336 hospital A and 20,000 hospital B patient files. Each file provides hourly observations, 40 input columns and the supplied `SepsisLabel`. Hospital identifiers denote the released cohorts, not interchangeable populations.

## Sources and access

- [Official landing page](https://physionet.org/content/challenge-2019/1.0.0/)
- [Hospital A directory](https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/)
- [Hospital B directory](https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/)
- [Source LICENSE.txt](https://physionet.org/files/challenge-2019/1.0.0/LICENSE.txt)

The public files can be downloaded without a personal account. On 2026-10-12 both directories were reachable and matched the expected counts. Historical ZIP URLs directly under the version directory were not working; the downloader does not assume those routes exist.

There is a source-license discrepancy: the landing page identifies **Creative Commons Attribution 4.0**, whereas its downloadable `LICENSE.txt` contains **ODbL**. The downloader preserves both documents locally. This project does not resolve that discrepancy or relicense source data. Source data are excluded from Git; obtain them from PhysioNet and retain attribution and source notices. The project's software license does not replace dataset or third-party-code terms.

## Reproducible acquisition

From the repository root:

```bash
python scripts/download_data.py --output data/raw
```

The official unsigned S3 mirror supports the complete download with eight pooled connections and verified reuse of an earlier pilot:

```bash
python -m pip install -e '.[acquisition]'
python scripts/download_data.py --source s3 --output data/full --workers 8 --reuse-from data/raw
```

Omit `--reuse-from` when no earlier verified download exists. The mirror is `s3://physionet-open/challenge-2019/1.0.0/`, linked from the official release page. Access is unsigned and needs no AWS account or credentials. The S3 route validates the expected A/B key counts, hashes its index snapshots and downloads only the exact dataset prefix. It uses the same bounded retry policy, with at most eight workers.

The default first checks for a unique whole-dataset ZIP advertised on the official landing page. Otherwise it reads the A/B directory indexes and downloads patient files with at most four concurrent workers. HTTP 429 and temporary server/network failures receive bounded retries, with a shared cooldown that respects `Retry-After`. ZIP extraction validates every member, rejects traversal and symbolic links, and writes only recognized patient files into normalized A/B directories. A whole-dataset ZIP, when available, can be supplied explicitly using `--source archive --archive-url URL`; only official HTTPS PhysioNet hosts are accepted.

For a reproducible acquisition smoke test:

```bash
python scripts/download_data.py --source files --output data/smoke_raw --limit-per-hospital 100 --seed 2019
```

The subset samples filenames uniformly without replacement; labels are never consulted for selection. A subset might contain too few sepsis cases for the model-development splits, in which case training should fail clearly rather than silently select additional positive patients. Reduced-cohort runs are software checks, not the full research benchmark. Use a separate output directory for each sampling scheme: unselected leftover patient files cause an explicit failure instead of silently contaminating the cohort.

`download_manifest.json` records source URLs, version, UTC retrieval time, per-file SHA256 and byte count, selection seed, actual counts, full/reduced scope, and an aggregate hash over the selected file paths and hashes. An archive additionally receives its own SHA256. Retained source notices and directory snapshots are hashed. These are recorded download checksums, **not independently authenticated publisher checksums**. The upstream checksum file is preserved; it is not assumed to cover every patient record. Failed runs retain an explicit failed manifest; a rerun reuses only files whose bytes match previously recorded hashes.

## Interpretation and limits

The complete acquisition finished on 2026-10-12: **40,336 files, 255,057,484 bytes**, comprising 20,336 A and 20,000 B records. Of these, 2,400 verified pilot files were reused without changing their bytes or original source URLs; 37,936 were downloaded from the official S3 mirror. Independent checks verified every digest, byte count, expected header, source URL, source notice and exact file-set agreement with the official listings. No files failed those checks.

- Aggregate path-and-file-digest hash: `70dcd7498dc68470339b0bdcbc7219981659551d29dcbe77bc7113c5c9ebabd6`.
- Completed acquisition-manifest SHA256: `1a3dc66330b9657587fa2ac998ed0f7676d74a664e88965cc8d9e6dc8984395e`.
- Raw records and the detailed per-record manifest remain outside the Git repository. The experiment manifest records the acquisition-manifest digest.
- The full B cohort includes the previously inspected 1,200-record B feasibility sample. The full protocol/configuration remained frozen; see [execution notes](execution_notes.md). No wholly new confirmatory validation is claimed.

- Supplied positive labels begin six hours before the reference sepsis onset and persist afterward. Do not shift them again. See [the evaluation protocol](protocol.md) for the separate pre-onset analysis and censoring rules.
- Files identify released patient records. Linkage of repeat admissions or people across hospitals cannot be verified from these files alone.
- Missing observations, actual delays in laboratory availability, and absent variables limit clinical interpretation. Artificial delay experiments simulate availability; they do not reconstruct original specimen/result timestamps.
- The public demonstration uses generated trajectories. It must not present those trajectories or their scores as clinical validation evidence.
- One held-out hospital provides one transport evaluation; it does not establish generalizability to other hospitals, present-day workflows or clinical benefit.

## Official scoring provenance

The official scorer is [physionetchallenges/evaluation-2019](https://github.com/physionetchallenges/evaluation-2019), file `evaluate_sepsis_score.py`, pinned to commit `4edb43177ee7d89dc86ab211705753c2d27b71e4` (2019-05-18). Its SHA256 is `26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5`. Preserve its BSD-2-Clause license and attribution separately from project and dataset notices. The original normalized utility evaluates hourly outputs; operational alert episodes are a distinct analysis.

## Citations

Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*. 2020;48(2):210–217. [doi:10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145).

Reyna M, Josef C, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019* (version 1.0.0). PhysioNet; 2019. [doi:10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).
