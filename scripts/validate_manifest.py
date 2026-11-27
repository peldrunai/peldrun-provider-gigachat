#!/usr/bin/env python3
"""
Static analysis of manifest.json. Catches typos, missing fields, 
invalid semver, and unsafe permissions before committing.
"""
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "manifest.json"

def validate(path: Path) -> bool:
    if not path.is_file():
        print(f"error: {path} not found")
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in manifest: {e}")
        return False

    required = ["schema_version", "kind", "id", "name", "version", "transport", "permissions"]
    for field in required:
        if field not in data:
            print(f"[err] Missing required field: {field}")
            return False

    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", data["id"]):
        print(f"[err] Invalid ID '{data['id']}'. Must be lowercase alphanumeric.")
        return False

    if not re.match(r"^\d+\.\d+\.\d+", data["version"]):
        print(f"[err] Invalid version '{data['version']}'. Must be SemVer.")
        return False

    if data.get("transport") in ("native", "custom"):
        entry_point = data.get("entry_point")
        if not entry_point or ":" not in entry_point:
            print("[err] Native transport requires 'entry_point' (module:ClassName)")
            return False

    perms = data.get("permissions", {})
    if perms.get("workspace_access") is not False:
        print("[err] permissions.workspace_access must be false")
        return False
    if perms.get("shell_execute") is not False:
        print("[err] permissions.shell_execute must be false")
        return False

    print("[ok] manifest.json is valid.")
    return True

if __name__ == "__main__":
    sys.exit(0 if validate(MANIFEST_PATH) else 1)