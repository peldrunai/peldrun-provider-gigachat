#!/usr/bin/env python3
# scripts/generate_catalog_entry.py
"""
Generates catalog-entry.json for the PELDRUN Store.

The catalog entry is the manifest augmented with two release-time fields:

    - download_url    : direct URL to the release ZIP
    - checksum_sha256 : SHA-256 of that ZIP

Every other field is copied verbatim from manifest.json, so any field
declared in the manifest (display_name, description, developer, tags,
docs_url, connection_fields, ...) flows through automatically.

Manifest-only fields that must never leak into the catalog entry
(release tooling metadata) are stripped via MANIFEST_ONLY_FIELDS.

Usage:
    python scripts/generate_catalog_entry.py \
        --download-url https://.../provider-1.0.0.zip \
        --sha256 <64-lowercase-hex>

    # Optional overrides (fall back to manifest values when omitted):
    python scripts/generate_catalog_entry.py ... \
        --brand-color "#F38020" --icon cloudflare

Exit codes:
    0 - success
    2 - invalid --sha256 format
    1 - manifest missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "manifest.json"
OUTPUT_PATH = BASE_DIR / "dist" / "catalog-entry.json"

# Fields that belong only to the manifest (release tooling, sandboxing hints)
# and must not appear in the public catalog entry.
MANIFEST_ONLY_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "kind",
    "entry_point",
    "min_peldrun_version",
})

HEX_CHARS = frozenset("0123456789abcdef")


def _validate_sha256(value: str) -> str:
    """Return the normalized (lowercase) SHA-256 or raise ValueError."""
    sha = value.strip().lower()
    if len(sha) != 64 or any(c not in HEX_CHARS for c in sha):
        raise ValueError(
            f"--sha256 must be 64 lowercase hex characters, got {value!r}"
        )
    return sha


def build_entry(
    manifest: dict,
    download_url: str,
    sha256: str,
    brand_color: str | None,
    icon: str | None,
) -> dict:
    """Pure function: manifest + release fields -> catalog entry."""
    entry = {k: v for k, v in manifest.items() if k not in MANIFEST_ONLY_FIELDS}

    # Release-time fields are authoritative; overwrite any manifest values.
    entry["download_url"] = download_url
    entry["checksum_sha256"] = sha256

    # CLI overrides take precedence over manifest values when provided.
    if brand_color:
        entry["brand_color"] = brand_color
    if icon:
        entry["icon"] = icon

    # Every PELDRUN provider uses named icons by convention.
    entry.setdefault("icon_type", "name")
    return entry


def generate_entry(
    download_url: str,
    sha256: str,
    brand_color: str | None = None,
    icon: str | None = None,
) -> int:
    if not MANIFEST_PATH.is_file():
        print(f"error: {MANIFEST_PATH} not found", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = build_entry(manifest, download_url, sha256, brand_color, icon)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {OUTPUT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the PELDRUN catalog-entry.json from manifest.json.",
    )
    parser.add_argument(
        "--download-url",
        required=True,
        help="Direct URL to the provider release ZIP.",
    )
    parser.add_argument(
        "--sha256",
        required=True,
        help="SHA-256 of the release ZIP (64 lowercase hex characters).",
    )
    parser.add_argument(
        "--brand-color",
        default=None,
        help="Override the manifest brand_color (e.g. '#F38020').",
    )
    parser.add_argument(
        "--icon",
        default=None,
        help="Override the manifest icon slug (e.g. 'cloudflare').",
    )
    args = parser.parse_args(argv)

    try:
        sha = _validate_sha256(args.sha256)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return generate_entry(
        args.download_url,
        sha,
        brand_color=args.brand_color,
        icon=args.icon,
    )


if __name__ == "__main__":
    sys.exit(main())