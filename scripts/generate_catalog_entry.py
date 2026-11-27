#!/usr/bin/env python3
"""
Generates catalog-entry.json. This file is NOT part of the ZIP. 
It is the metadata file used by the PELDRUN Store to display the provider.
"""
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def generate_entry(download_url, brand_color, icon):
    manifest_path = BASE_DIR / "manifest.json"
    if not manifest_path.is_file():
        print("error: manifest.json not found")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entry = {
        "id": manifest["id"],
        "name": manifest["name"],
        "version": manifest["version"],
        "transport": manifest["transport"],
        "features": manifest.get("features", {}),
        "permissions": manifest.get("permissions", {}),
        "brand_color": brand_color,
        "icon": icon,
        "icon_type": "name",
        "download_url": download_url,
        "checksum_sha256": "PLACEHOLDER_SHA256_FROM_VERIFY_SCRIPT",
    }

    if manifest.get("author"):
        entry["developer"] = {
            "name": manifest["author"],
            "url": manifest.get("author_url", ""),
            "verified": False
        }

    out_path = BASE_DIR / "dist" / "catalog-entry.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-url", required=True)
    parser.add_argument("--brand-color", default="#000000")
    parser.add_argument("--icon", default="default")
    args = parser.parse_args()

    generate_entry(args.download_url, args.brand_color, args.icon)