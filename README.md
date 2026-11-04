# peldrun-provider-gigachat

GigaChat (Sber) Provider Package for [PELDRUN](https://github.com/peldrun).

This package teaches PELDRUN how to talk to the GigaChat family of
Russian language models from Sber. It implements the
`peldrun.llm.provider.v1` adapter contract and runs inside PELDRUN's
isolated worker subprocess.

---

## What's inside

| Path | Purpose |
|------|---------|
| `package/manifest.json` | Package metadata, connection fields, certificate declaration. |
| `package/requirements.txt` | Python dependencies installed into the isolated venv. |
| `package/gigachat_peldrun/adapter.py` | The adapter class implementing `ProviderAdapter`. |
| `package/gigachat_peldrun/__init__.py` | Package marker and version metadata. |
| `scripts/build.sh` | Builds the distributable `.zip` from `package/`. |
| `tests/test_adapter.py` | Local test harness (requires real credentials). |

The contents of `package/` are what gets zipped and uploaded to
PELDRUN. The rest of the repository is tooling and documentation.

---

## Requirements

### On the PELDRUN backend host

- PELDRUN with the `peldrun.llm.provider.v1` contract enabled.
- Python `>=3.12,<3.13` (validated by the dependency installer).
- Network egress to:
  - `https://gigachat.devices.sberbank.ru` (API endpoint)
  - `https://gu-st.ru` (Russian Trusted Root CA download)

### Credentials

A valid GigaChat account on Sber Studio. You will need:

- **Authorization Key** — the base64-encoded `client_id:client_secret`
  shown in the Sber Studio dashboard.
- **Account Scope** — one of `PERS`, `B2B`, `CORP`.

### TLS certificate

GigaChat uses the **Russian Trusted Root CA**, which is NOT included
in Python's default `certifi` trust store. The package declares this
certificate in `manifest.json` under the `certificates` array.

PELDRUN's **Certificate Manager** handles the download and installation
automatically:

1. From the Store → Providers → GigaChat card, click **Certificates**.
2. In the Certificate Manager modal, click **Download & Install**.
3. The backend downloads the CA from `gu-st.ru`, verifies its format,
   and writes it to the managed certificate directory.
4. The adapter reads the CA path from the installed record at request
   time.

No manual file copying is required.

---

## Installation

### 1. Build the distributable ZIP

From the repository root:

```bash
# Linux / macOS / Git Bash on Windows
./scripts/build.sh
```

The script reads `version` from `package/manifest.json` and produces
`dist/gigachat-<version>.zip`. The ZIP contains `manifest.json` at
its root, as required by PELDRUN's package validator.

On Windows without Git Bash, use PowerShell:

```powershell
Compress-Archive -Path package\* -DestinationPath gigachat-1.0.0.zip -Force
```

### 2. Install the package in PELDRUN

1. Open PELDRUN's UI → **Store** → **Providers** → **Upload Package**.
2. Select the `.zip` file produced above.
3. Wait for the install pipeline to complete. The isolated venv will
   install `gigachat>=0.2.0,<1.0` and `httpx>=0.27,<0.28` — this
   typically takes 30–60 seconds.
4. The GigaChat card now appears in the **Installed** tab.

### 3. Install the TLS certificate

1. On the GigaChat card, click the amber **Certificates missing**
   badge or the **Certificates** button in the footer.
2. In the Certificate Manager modal, click **Download & Install**.
3. Wait for the download (~2–5 seconds). The status changes from
   `Missing` to `Installed`, and the badge turns purple.
4. Optionally click **Test** to verify the file, hash, and format.

### 4. Configure a connection

1. Click **Configure** on the GigaChat card.
2. Fill in the fields:
   - **Connection Name** — e.g. `Personal`.
   - **Endpoint URL** — defaults to `https://gigachat.devices.sberbank.ru/api/v1`.
   - **Authorization Key** — the base64 credentials from Sber Studio.
   - **Account Scope** — `PERS`, `B2B`, or `CORP`.
   - **Verify SSL Certificates** — leave checked.
3. Click **Create Connection**.

### 5. Test and use

1. In the connection list, click the **⚡** icon to run a connectivity test.
2. Open a chat, select **GigaChat** as the provider, pick a model
   (e.g. `GigaChat`), and send a message.

---

## How it works

### Adapter lifecycle

For the `native` transport, PELDRUN:

1. Extracts the package into `storage/store/providers/gigachat/versions/<version>/`.
2. Creates an isolated Python venv for that version.
3. Installs `requirements.txt` into that venv.
4. Spawns a worker subprocess (`peldrun.llm.plugin_host.worker`) on the
   first request.
5. Loads `gigachat_peldrun.adapter:GigaChatAdapter` inside the worker.
6. Routes every `generate()` / `stream()` call through IPC.

The adapter's SDK client (`gigachat.GigaChat`) handles OAuth token
acquisition and refresh internally. Access tokens expire after ~30
minutes; the SDK refreshes them transparently.

### Certificate flow

When you click **Download & Install** in the Certificate Manager:

1. The backend (main process, not the worker) downloads the CA file
   from `https://gu-st.ru/...`.
2. Size is capped at 1 MB, timeout at 60 seconds.
3. If `download_sha256` is declared in `manifest.json`, the hash is
   verified. When it is `null` (as in v1.0.0), only the HTTPS
   transport and the PEM format are checked.
4. The file is written atomically to
   `storage/store/providers/gigachat/certificates/<filename>`.
5. A record is written to `models_vault.json` under
   `provider_certificates`.
6. The certificate path is then passed to the adapter via the
   connection's `ca_bundle_path` field.

The download happens in the **main backend process** (not the worker)
because the worker's network allowlist restricts outbound requests to
the connection's `base_url` host. The CA host is a different host, so
only the main process can reach it.

### Pinning the certificate hash

For production deployments, pin the SHA-256 of the CA file:

1. After a successful install, open the Certificate Manager.
2. Copy the value shown under **SHA-256**.
3. Update `package/manifest.json`:
   ```json
   "download_sha256": "<paste-the-hex-here>"
   ```
4. Bump `version` to `1.0.1`.
5. Rebuild the ZIP and reinstall.

With a pinned hash, any tampering with the CA file — even over
HTTPS — is detected before installation.

---

## Development

### Local testing

The `tests/test_adapter.py` script exercises the adapter directly
without PELDRUN. It requires:

- Python 3.12
- `pip install -r package/requirements.txt`
- `pip install peldrun` (for the contract types)
- A valid GigaChat `Authorization Key`

Run it:

```bash
cd tests
python test_adapter.py
```

The script prints the response content, token usage, and streamed
tokens. It is intended for adapter development only — never commit
real credentials.

### Editing the manifest

When adding new connection fields or certificates:

1. Update `package/manifest.json`.
2. Bump `version` (e.g. `1.0.0` → `1.0.1`).
3. Rebuild the ZIP (`./scripts/build.sh`).
4. Reinstall in PELDRUN. The old version remains available for
   rollback until you explicitly delete it.

### Schema validation

The manifest uses `extra="forbid"`. A typo in a field name — or a
field that does not exist in `ProviderManifest` — causes an install
failure with a clear error message. Always validate locally:

```bash
python -c "import json; json.load(open('package/manifest.json'))"
```

For full Pydantic validation, install PELDRUN and import
`omweb.provider_packages.schema.ProviderManifest`:

```python
import json
from omweb.provider_packages.schema import ProviderManifest

with open("package/manifest.json") as f:
    raw = json.load(f)

try:
    ProviderManifest.model_validate(raw)
    print("OK")
except Exception as e:
    print(f"INVALID: {e}")
```

---

## Troubleshooting

### `manifest.certificates: Extra inputs are not permitted`

You are installing this package on an older PELDRUN build that does
not yet understand the `certificates` field. Update PELDRUN to a
version that ships `CertificateSpec` in
`omweb.provider_packages.schema`, then retry.

### `entry_point module 'gigachat_peldrun.adapter' does not resolve`

The ZIP is missing `package/gigachat_peldrun/adapter.py`. Rebuild from
the `package/` directory — not from the repository root.

### `pip install failed (exit 1)`

Check the `pip_log_tail` in the error response. Common causes:

- Network unreachable from the PELDRUN host to PyPI.
- Incompatible Python version (the manifest requires `>=3.12,<3.13`).

### `HTTP 401` when chatting

The Authorization Key is wrong or expired. Regenerate it from Sber
Studio and update the connection.

### `SSL: CERTIFICATE_VERIFY_FAILED`

The Russian Trusted Root CA is not installed. Open the Certificate
Manager (see above) and click **Download & Install**.

### `HTTP 404` when chatting

The base URL is wrong. GigaChat's default is
`https://gigachat.devices.sberbank.ru/api/v1` — note the `/api/v1`
suffix.

---

## Security

- The adapter never spawns subprocesses, never reads workspace files,
  and never reaches any host other than its connection's `base_url`.
- The certificate download runs in the main backend process, isolated
  from the worker's network allowlist. This is by design; see
  `certificate_manager.py` in PELDRUN for the full rationale.
- Credentials (the Authorization Key) are stored in
  `models_vault.json` in plaintext (PELDRUN does not encrypt at rest)
  and masked in API responses. Protect the vault file with filesystem
  permissions.
- Never commit real credentials. `.gitignore` excludes common
  secret-bearing filenames such as `.env` and `*.local.json`.

---

## License

MIT. See `LICENSE`.

---

## Links

- [GigaChat API documentation](https://developers.sber.ru/docs/ru/gigachat/guides/main)
- [Russian Trusted Root CA](https://www.gosuslugi.ru/crt)
- [PELDRUN](https://github.com/peldrun/peldrun)
- [PELDRUN Provider Packages documentation](https://github.com/peldrun/peldrun/tree/main/docs/providers)
```

---
Windows  PowerShell .\scripts\build.ps1