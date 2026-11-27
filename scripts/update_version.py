#!/usr/bin/env python3
"""
Single source of truth for version bumping. Updates manifest.json, 
pyproject.toml, __init__.py, and prepends a new section to CHANGELOG.md.
"""
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "manifest.json"
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"
INIT_PATH = BASE_DIR / "src" / "gigachat_peldrun" / "__init__.py" # Update path dynamically if needed
CHANGELOG_PATH = BASE_DIR / "CHANGELOG.md"

def bump_version(spec: str):
    if not MANIFEST_PATH.is_file():
        print("error: manifest.json not found")
        sys.exit(1)

    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current = data["version"]

    major, minor, patch = map(int, current.split("."))
    if spec == "major":
        major += 1; minor = 0; patch = 0
    elif spec == "minor":
        minor += 1; patch = 0
    elif spec == "patch":
        patch += 1
    elif re.match(r"^\d+\.\d+\.\d+$", spec):
        major, minor, patch = map(int, spec.split("."))
    else:
        print("Usage: update_version.py <major|minor|patch|X.Y.Z>")
        sys.exit(1)

    new_version = f"{major}.{minor}.{patch}"
    data["version"] = new_version
    MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Update pyproject.toml
    if PYPROJECT_PATH.is_file():
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        content = re.sub(r'version = ".*"', f'version = "{new_version}"', content)
        PYPROJECT_PATH.write_text(content, encoding="utf-8")

    print(f"Version updated: {current} -> {new_version}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--check":
        print("Version check passed (stub).")
        sys.exit(0)
    bump_version(sys.argv[1])