#!/usr/bin/env bash
# scripts/build.sh
#
# Builds the distributable GigaChat provider ZIP from the contents of
# package/. The resulting ZIP contains manifest.json at its root, which
# is the layout required by PELDRUN's package validator.
#
# Usage:
#   ./scripts/build.sh
#
# Output:
#   dist/gigachat-<version>.zip
#
# Requires:
#   - bash 4+
#   - zip (available on Linux/macOS; on Windows use Git Bash or WSL)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_DIR="${REPO_ROOT}/package"
DIST_DIR="${REPO_ROOT}/dist"

# Read the version from manifest.json so the ZIP name always matches.
if [[ ! -f "${PACKAGE_DIR}/manifest.json" ]]; then
    echo "ERROR: package/manifest.json not found" >&2
    exit 1
fi

VERSION="$(grep -oE '"version"[[:space:]]*:[[:space:]]*"[^"]+"' \
    "${PACKAGE_DIR}/manifest.json" \
    | head -n1 \
    | sed -E 's/.*"([^"]+)"$/\1/')"

if [[ -z "${VERSION}" ]]; then
    echo "ERROR: could not extract version from manifest.json" >&2
    exit 1
fi

ZIP_NAME="gigachat-${VERSION}.zip"
ZIP_PATH="${DIST_DIR}/${ZIP_NAME}"

echo "Building ${ZIP_NAME} (version ${VERSION})..."
mkdir -p "${DIST_DIR}"
rm -f "${ZIP_PATH}"

# Change into package/ so the ZIP root contains manifest.json directly.
# Do NOT zip the package/ directory itself; that would nest everything
# one level too deep and the validator would reject it.
(
    cd "${PACKAGE_DIR}"
    zip -r -9 "${ZIP_PATH}" . \
        -x "*.pyc" \
        -x "__pycache__/*" \
        -x "*.egg-info/*" \
        -x ".pytest_cache/*" \
        -x "*.log" \
        > /dev/null
)

echo "Done: ${ZIP_PATH}"
echo "Size: $(stat -c%s "${ZIP_PATH}" 2>/dev/null || stat -f%z "${ZIP_PATH}") bytes"

echo ""
echo "Verify contents:"
unzip -l "${ZIP_PATH}"