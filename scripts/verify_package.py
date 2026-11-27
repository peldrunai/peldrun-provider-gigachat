#!/usr/bin/env python3
"""
Simulates PELDRUN's strict install-time security checks. Verifies no path 
traversal, no symlinks, and calculates the final SHA-256 checksum.
"""
import hashlib
import zipfile
from pathlib import Path
import sys

def verify_zip(zip_path: Path):
    print(f"Verifying {zip_path.name}...")

    if not zip_path.is_file():
        print(f"[err] File not found: {zip_path}")
        return False

    hasher = hashlib.sha256()
    hasher.update(zip_path.read_bytes())
    print(f"  sha256: {hasher.hexdigest()}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        if "manifest.json" not in names:
            print("[err] manifest.json missing from root!")
            return False

        for name in names:
            if ".." in name or name.startswith("/"):
                print(f"[err] Zip Slip vulnerability detected: {name}")
                return False

            info = zf.getinfo(name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                print(f"[err] Symlink detected: {name}")
                return False

    print("[ok] ZIP verification passed. Safe for PELDRUN.")
    return True

if __name__ == "__main__":
    dist_dir = Path(__file__).resolve().parent.parent / "dist"
    zips = list(dist_dir.glob("*.zip"))
    if not zips:
        print("[err] No ZIP files found in dist/")
        sys.exit(1)
    
    target_zip = zips[-1] # Get the most recent one
    success = verify_zip(target_zip)
    sys.exit(0 if success else 1)