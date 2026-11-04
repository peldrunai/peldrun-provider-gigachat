# scripts/build.ps1
# Builds the distributable GigaChat provider ZIP from package/ contents.
#
# Produces:
#   dist/gigachat-<version>.zip
#
# ZIP root layout (required by the PELDRUN validator):
#   manifest.json
#   requirements.txt
#   gigachat_peldrun/__init__.py
#   gigachat_peldrun/adapter.py
#
# Behavior:
#   * Removes __pycache__ and *.pyc from package/ BEFORE zipping
#     (Compress-Archive has no exclusion flag; pre-cleanup is the
#     only reliable way to keep bytecode out of the archive).
#   * Reads the version from manifest.json so the ZIP name always
#     matches the manifest.
#   * Verifies the result: manifest at root, no bytecode inside.
#
# Usage:
#   .\scripts\build.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PackageDir = Join-Path $RepoRoot "package"
$DistDir = Join-Path $RepoRoot "dist"
$ManifestPath = Join-Path $PackageDir "manifest.json"

# ---- 1. Sanity check --------------------------------------------------------
if (-not (Test-Path $ManifestPath)) {
    Write-Error "package/manifest.json not found"
    exit 1
}

# ---- 2. Read version --------------------------------------------------------
$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$version = $manifest.version
if (-not $version) {
    Write-Error "Could not read 'version' from manifest.json"
    exit 1
}

$zipName = "gigachat-$version.zip"
$zipPath = Join-Path $DistDir $zipName

# ---- 3. Clean Python bytecode ----------------------------------------------
Write-Host "Cleaning Python bytecode from package/..."
$removed = 0

Get-ChildItem -Path $PackageDir -Recurse -Directory -Filter "__pycache__" `
    -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Removing dir:  $($_.FullName)"
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $removed++
    }

Get-ChildItem -Path $PackageDir -Recurse -Filter "*.pyc" `
    -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Removing file: $($_.FullName)"
        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
        $removed++
    }

if ($removed -eq 0) {
    Write-Host "  (nothing to clean)"
} else {
    Write-Host "  Removed $removed bytecode item(s)"
}

# ---- 4. Recreate dist/ ------------------------------------------------------
if (Test-Path $DistDir) {
    Remove-Item -Path $DistDir -Recurse -Force
}
New-Item -ItemType Directory -Path $DistDir | Out-Null

# ---- 5. Build the ZIP -------------------------------------------------------
# -Path "$PackageDir\*" adds the CONTENTS of package/, not the folder.
# This ensures manifest.json lands at the ZIP root, not inside "package/".
Write-Host ""
Write-Host "Building $zipName (version $version)..."
Compress-Archive -Path (Join-Path $PackageDir "*") -DestinationPath $zipPath -Force

# ---- 6. Verify contents -----------------------------------------------------
Write-Host ""
Write-Host "Verifying ZIP contents:"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$hasRootManifest = $false
$hasPycache = $false
$hasPyc = $false
$fileCount = $zip.Entries.Count

foreach ($entry in $zip.Entries) {
    Write-Host ("  {0,-45} {1,10} bytes" -f $entry.FullName, $entry.Length)
    if ($entry.FullName -eq "manifest.json")     { $hasRootManifest = $true }
    if ($entry.FullName -like "*__pycache__*")   { $hasPycache = $true }
    if ($entry.FullName -like "*.pyc")           { $hasPyc = $true }
}
$zip.Dispose()

$size = (Get-Item $zipPath).Length

# ---- 7. Summary + sanity exit ----------------------------------------------
Write-Host ""
Write-Host "Summary:"
Write-Host "  File count:              $fileCount"
Write-Host "  manifest.json at root:   $hasRootManifest"
Write-Host "  contains __pycache__:    $hasPycache"
Write-Host "  contains *.pyc:          $hasPyc"
Write-Host "  ZIP size:                $size bytes"
Write-Host "  ZIP path:                $zipPath"

if (-not $hasRootManifest) {
    Write-Error "FAIL: manifest.json is not at the ZIP root"
    exit 1
}
if ($hasPycache -or $hasPyc) {
    Write-Error "FAIL: Python bytecode found inside the ZIP"
    exit 1
}

Write-Host ""
Write-Host "DONE: $zipPath"