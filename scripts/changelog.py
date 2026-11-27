#!/usr/bin/env python3
"""
Automatically generates release notes by parsing Git commit history 
since the last tag, looking for Conventional Commits.
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def get_commits_since_last_tag():
    try:
        tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()
        logs = subprocess.check_output(["git", "log", f"{tag}..HEAD", "--pretty=format:%s"], text=True).strip()
        return logs.split("\n") if logs else []
    except subprocess.CalledProcessError:
        return []

def generate_changelog(version):
    commits = get_commits_since_last_tag()
    categories = {"Added": [], "Fixed": [], "Changed": []}

    for msg in commits:
        if msg.startswith("feat:"):
            categories["Added"].append(f"- {msg.replace('feat:', '').strip()}")
        elif msg.startswith("fix:"):
            categories["Fixed"].append(f"- {msg.replace('fix:', '').strip()}")
        elif msg.startswith("refactor:") or msg.startswith("perf:") or msg.startswith("chore:"):
            categories["Changed"].append(f"- {msg.split(':', 1)[1].strip() if ':' in msg else msg}")

    md = f"## [{version}] - YYYY-MM-DD\n\n"
    for cat, items in categories.items():
        if items:
            md += f"### {cat}\n" + "\n".join(items) + "\n\n"

    print("Generated CHANGELOG section:")
    print(md)

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "Unreleased"
    generate_changelog(version)