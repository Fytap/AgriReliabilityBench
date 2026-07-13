#!/usr/bin/env python3
"""Validate public-release metadata before tagging an archival version."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHORS = [
    ("Liu", "Shuhao", "0009-0000-2996-9868"),
    ("Chen", "Zhuo", "0009-0007-7510-8648"),
    ("Liu", "Jie", "0009-0004-8073-0085"),
    ("Ling", "Zhi", "0009-0003-2787-156X"),
    ("Yan", "Yu", "0009-0006-4410-4444"),
]


def main() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert not (ROOT / "METADATA_TO_CONFIRM.md").exists(), "Remove stale confirmation notes."
    assert citation["version"] == zenodo["version"] == version, "Version mismatch."
    assert citation["title"] == zenodo["title"], "Title mismatch."

    cff_authors = [
        (item["family-names"], item["given-names"], item["orcid"].removeprefix("https://orcid.org/"))
        for item in citation["authors"]
    ]
    zenodo_authors = [
        tuple(reversed(item["name"].split(", ", maxsplit=1))) + (item["orcid"],)
        for item in zenodo["creators"]
    ]
    expected_cff = [(family, given, orcid) for family, given, orcid in EXPECTED_AUTHORS]
    expected_zenodo = [(given, family, orcid) for family, given, orcid in EXPECTED_AUTHORS]
    assert cff_authors == expected_cff, "CITATION.cff author order or ORCID mismatch."
    assert zenodo_authors == expected_zenodo, ".zenodo.json author order or ORCID mismatch."
    print(f"Release metadata is internally consistent for v{version}.")


if __name__ == "__main__":
    main()
