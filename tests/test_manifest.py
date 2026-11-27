"""
Basic integrity tests for the provider manifest.
Ensures the manifest exists and is valid JSON before CI proceeds.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "manifest.json"

def test_manifest_exists():
    """Verify that manifest.json is present at the repository root."""
    assert MANIFEST_PATH.is_file(), "manifest.json is missing from the repository root."

def test_manifest_is_valid_json():
    """Verify that manifest.json contains valid JSON."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "manifest.json root must be a JSON object."
    assert "id" in data, "manifest.json must contain an 'id' field."
    assert "version" in data, "manifest.json must contain a 'version' field."