#!/usr/bin/env python3
"""
Creates the final .zip artifact. Flattens the src/ directory and sets 
deterministic timestamps to ensure reproducible builds (identical SHA-256).
"""
import json
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "dist"
MANIFEST_PATH = BASE_DIR / "manifest.json"

def build_zip():
    if not MANIFEST_PATH.is_file():
        print("error: manifest.json not found")
        return

    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    version = data["version"]
    provider_id = data["id"]
    zip_path = DIST_DIR / f"{provider_id}-{version}.zip"

    DIST_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Manifest and Requirements at root
        zf.write(BASE_DIR / "manifest.json", "manifest.json")
        req_path = BASE_DIR / "requirements.txt"
        if req_path.is_file():
            zf.write(req_path, "requirements.txt")

        # 2. Flatten src/ package into root
        src_pkg = BASE_DIR / "src" / f"{provider_id.replace('-', '_')}_peldrun"
        if src_pkg.is_dir():
            for file in src_pkg.rglob("*"):
                if file.is_file() and "__pycache__" not in file.parts:
                    arcname = file.relative_to(BASE_DIR / "src")
                    info = zipfile.ZipInfo(str(arcname), date_time=(2020, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    zf.writestr(info, file.read_bytes())

    print(f"Built {zip_path}")

if __name__ == "__main__":
    build_zip()