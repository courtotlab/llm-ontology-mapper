#!/usr/bin/env python
"""
Vendors the raw per-query text2term-evaluation output files needed for the
Scenario 1 common-query-aligned graph-relationship comparison
(src/llm_ontology_mapper/benchmarking/text2term_alignment.py).

This is the ONLY step in this workflow allowed to reach the network. It
downloads exactly nine files from ONE pinned commit of
https://github.com/rsgoncalves/text2term-evaluation -- the SAME commit
(b999dbb670fa13c9ceb1ba631a7abc7557f3293b) already used as the graph-
evaluator reference by scenario1_graph_distance.py / graph_reference_metadata.json
(that commit's tree was audited and confirmed to contain all nine required
output/*.{tsv,csv} files -- no commit mixing).

Every downloaded file's SHA256 is verified against a pinned expected value
before it is accepted, and a provenance.json manifest (repository URL,
pinned commit, upstream path, raw URL, local path, SHA256, fetch timestamp)
is written alongside them. The downstream alignment/plotting code never
touches the network -- it only reads these already-vendored files.

Usage
─────
    uv run python scripts/fetch_text2term_evaluation_outputs.py
    uv run python scripts/fetch_text2term_evaluation_outputs.py --force   # re-fetch even if present
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

from llm_ontology_mapper.benchmarking.scenario1_graph_distance import (  # noqa: E402
    PINNED_COMMIT,
    SOURCE_REPOSITORY,
)

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_DIR / "data" / "text2term_evaluation" / "original_outputs"
RAW_BASE_URL = f"https://raw.githubusercontent.com/rsgoncalves/text2term-evaluation/{PINNED_COMMIT}"

# Audited against the pinned commit's git tree (2026-09-03): all nine files
# are present there, so no second commit is ever consulted. Upstream path ->
# (local filename, expected SHA256).
REQUIRED_FILES: dict[str, tuple[str, str]] = {
    "output/UKBB-EFO_results.tsv": ("UKBB-EFO_results.tsv", "89e84596b33dbd4ca666eab2984f79160bf8a3e102d59da325332a1d1a643732"),
    "output/UKBB-EFO_mappings.tsv": ("UKBB-EFO_mappings.tsv", "1b13e3ca24d3dd51d527289298fb6a0b56a24677dbf1d0e23cda6bfbd564a027"),
    "output/UKBB-EFO_t2t_mappings.csv": ("UKBB-EFO_t2t_mappings.csv", "1ed0670dc02b3976a8c09c135aa9a6b8447450ed777c3076648b44479870f09f"),
    "output/Biomappings_results.tsv": ("Biomappings_results.tsv", "dc90304cb2213869b27f70c2f9744fc9f5606d75d7b4b0ddd16e8f97ef9ad1e9"),
    "output/Biomappings_mappings.tsv": ("Biomappings_mappings.tsv", "e9f8e052960a86202b3ba0154dd80407920a9fa02a6aff2fe26b335b1d42f11d"),
    "output/Biomappings_t2t_mappings.csv": ("Biomappings_t2t_mappings.csv", "67d724cae089b04ec75545cb608fd6d2b128ae8687c1a6842d66254984323fd7"),
    "output/OLS-EFO_results.tsv": ("OLS-EFO_results.tsv", "d85aa2f75db4a3c014be57f72eb3a258b79fac218d96b51ad0fb91ab4b5ed59d"),
    "output/OLS-EFO_mappings.tsv": ("OLS-EFO_mappings.tsv", "d92df7a7dc32293ddd9af699a0065c3239f0dcaad427a5486a35618e8ab9c3b1"),
    "output/OLS-EFO_t2t_mappings.csv": ("OLS-EFO_t2t_mappings.csv", "b423af08b07d0148195a74d909927face6a1314db7a2e869889b5807642e44b5"),
}


class FetchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(upstream_path: str, local_filename: str, expected_sha256: str, output_dir: Path, *, force: bool) -> dict:
    dest = output_dir / local_filename
    if dest.exists() and not force:
        actual = _sha256(dest)
        if actual == expected_sha256:
            return {"upstream_path": upstream_path, "local_path": str(dest), "sha256": actual, "status": "already_present"}
        raise FetchError(
            f"{dest} exists but SHA256 {actual} != expected {expected_sha256} -- re-run with --force to re-fetch"
        )

    url = f"{RAW_BASE_URL}/{upstream_path}"
    print(f"  fetching {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-ontology-mapper-vendoring-script"})
    with urllib.request.urlopen(request, timeout=60, context=_SSL_CONTEXT) as response:  # noqa: S310
        if response.status != 200:
            raise FetchError(f"{url}: HTTP {response.status}")
        data = response.read()

    output_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    actual = _sha256(dest)
    if actual != expected_sha256:
        raise FetchError(
            f"{dest} SHA256 mismatch after download: expected {expected_sha256}, got {actual}. Refusing to keep "
            "a file that does not match the pinned commit's known content."
        )
    return {"upstream_path": upstream_path, "local_path": str(dest), "sha256": actual, "status": "fetched"}


def write_provenance(entries: list[dict], output_dir: Path) -> Path:
    manifest = {
        "repository_url": SOURCE_REPOSITORY,
        "pinned_commit": PINNED_COMMIT,
        "raw_base_url": RAW_BASE_URL,
        "fetch_timestamp": datetime.now(UTC).isoformat(),
        "files": entries,
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--force", action="store_true", help="Re-download even if a matching file is already present.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    print("=" * 78)
    print("Fetching text2term-evaluation raw per-query output files (pinned commit)")
    print("=" * 78)
    print(f"Repository:   {SOURCE_REPOSITORY}")
    print(f"Pinned commit: {PINNED_COMMIT}")
    print(f"Output dir:   {output_dir}")
    print("=" * 78)

    entries = []
    try:
        for upstream_path, (local_filename, expected_sha256) in REQUIRED_FILES.items():
            entry = fetch_one(upstream_path, local_filename, expected_sha256, output_dir, force=args.force)
            entries.append(entry)
            print(f"  OK  {entry['status']:15s} {entry['local_path']}  sha256={entry['sha256']}")
    except FetchError as exc:
        print(f"\nERROR: {exc}")
        return 1

    manifest_path = write_provenance(entries, output_dir)
    print(f"\nWrote provenance manifest: {manifest_path}")
    print(f"\n=== {len(entries)}/{len(REQUIRED_FILES)} required files vendored under {output_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
