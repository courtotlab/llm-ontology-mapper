#!/usr/bin/env python
"""
Deterministic, checksum-verified fetch of the EFO hierarchy edge tables used
for Scenario 1 graph-distance classification (compare_ontology_mappings.py
semantics from the published text2term evaluation repository).

Source repository: https://github.com/rsgoncalves/text2term-evaluation
Pinned commit:      b999dbb670fa13c9ceb1ba631a7abc7557f3293b (2024-04-25)
EFO version used by that repository's comparison: v3.62.0
    http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl

Fetches from the exact pinned commit (not `main`) so the files can never
silently change under us, and verifies each download against the SHA256
recorded below before writing it to disk -- a checksum mismatch is a hard
error, never a silent fallback.

Usage:
    uv run python scripts/fetch_efo_graph_reference_data.py
    uv run python scripts/fetch_efo_graph_reference_data.py --data-dir /custom/path
    uv run python scripts/fetch_efo_graph_reference_data.py --force   # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests  # type: ignore[import-untyped]

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_DIR / "data" / "text2term_evaluation"

SOURCE_REPOSITORY = "https://github.com/rsgoncalves/text2term-evaluation"
PINNED_COMMIT = "b999dbb670fa13c9ceb1ba631a7abc7557f3293b"
EFO_VERSION = "3.62.0"
EFO_URL = "http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl"

_RAW_BASE = f"https://raw.githubusercontent.com/rsgoncalves/text2term-evaluation/{PINNED_COMMIT}"

# filename -> (relative path in source repo, expected SHA256)
FILES: dict[str, tuple[str, str]] = {
    "efo_edges.tsv": (
        "data/efo_edges.tsv",
        "6aa7182b70e23addb9f6d4e24bab94520bf9ff26ea26471403dd4a568689e90c",
    ),
    "efo_entailed_edges.tsv": (
        "data/efo_entailed_edges.tsv",
        "589ab467d24ddd22065abc75dff3aace28a9377197edf7caef201a273015d243",
    ),
    "compare_ontology_mappings.py": (
        "compare_ontology_mappings.py",
        "cb70116e0f6cab58921dae60ccc8cba790153d1ee8831f613d01e0442a133781",
    ),
}


class FetchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(filename: str, *, data_dir: Path, force: bool) -> Path:
    relative_path, expected_sha256 = FILES[filename]
    dest = data_dir / filename

    if dest.exists() and not force:
        actual = _sha256(dest)
        if actual == expected_sha256:
            print(f"OK (cached, checksum verified): {dest}")
            return dest
        raise FetchError(
            f"{dest} exists but its SHA256 ({actual}) does not match the pinned "
            f"expected value ({expected_sha256}). Refusing to use a possibly "
            "corrupted or tampered file. Re-run with --force to re-download."
        )

    url = f"{_RAW_BASE}/{relative_path}"
    print(f"Downloading {url} -> {dest} ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.content
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"Failed to download {url}: {exc}") from exc

    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise FetchError(
            f"Downloaded {url} but its SHA256 ({actual}) does not match the pinned "
            f"expected value ({expected_sha256}). Refusing to write a file that does "
            "not match the reference implementation exactly."
        )

    dest.write_bytes(payload)
    print(f"OK (downloaded, checksum verified): {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid cached copy exists")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    print(f"Source repository: {SOURCE_REPOSITORY}")
    print(f"Pinned commit:      {PINNED_COMMIT}")
    print(f"EFO version:        {EFO_VERSION} ({EFO_URL})")
    print(f"Data directory:     {data_dir}\n")

    try:
        for filename in FILES:
            fetch_one(filename, data_dir=data_dir, force=args.force)
    except FetchError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nAll EFO graph reference files fetched and checksum-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
