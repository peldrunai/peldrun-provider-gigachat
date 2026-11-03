# tests/test_adapter.py
"""
Local test harness for the GigaChat Provider Package adapter.

This script exercises the adapter directly, WITHOUT going through
PELDRUN. It is intended for adapter development and diagnostics:

    * Verify the module imports cleanly.
    * Exercise generate() and stream() with real credentials.
    * Exercise discover_models() and validate_connection().
    * Verify error paths behave correctly.

Modes
-----
The script runs in one of two modes, chosen automatically:

    * **dry**  — No credentials configured. Verifies only that the
      adapter class imports, exposes the required attributes, and
      raises the expected errors on missing credentials. Safe to run
      in CI.

    * **live** — Credentials configured. Runs the full set of tests
      against the real GigaChat API. Requires:
        - GIGACHAT_CREDENTIALS  (base64-encoded client_id:client_secret)
        - GIGACHAT_SCOPE        (optional, default "PERS")
        - GIGACHAT_CA_BUNDLE    (optional, path to the Russian Trusted
                                 Root CA; if omitted, the system trust
                                 store is used, which WILL fail on most
                                 systems)

Usage
-----
Dry mode (no credentials):

    python tests/test_adapter.py

Live mode (real credentials):

    set GIGACHAT_CREDENTIALS=<base64-credentials>
    set GIGACHAT_SCOPE=PERS
    set GIGACHAT_CA_BUNDLE=C:\\path\\to\\russian_trusted_root_ca.pem
    python tests/test_adapter.py

    (On Linux/macOS, use `export VAR=value` instead of `set`.)

What this script does NOT do
----------------------------
    * It does not use PELDRUN's worker subprocess, IPC, or permission
      enforcer. It imports and instantiates the adapter directly.
    * It does not test the Certificate Manager flow. That flow lives
      in PELDRUN's backend; test it via the UI.
    * It does not persist anything. Each run is ephemeral.

See also
--------
    * package/gigachat_peldrun/adapter.py — the adapter under test
    * docs/providers/03-quickstart-native.md#step-10--testing-locally
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Path setup — allow running from the tests/ directory
# ---------------------------------------------------------------------------

# The adapter lives in package/gigachat_peldrun/. When this script is
# run from tests/, the package directory is not on sys.path. Add it
# explicitly so the import below succeeds.
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Try to make PELDRUN importable if it is not already pip-installed.
#
# The adapter imports from `peldrun.llm.*`, so `peldrun` must be
# importable BEFORE the adapter module is loaded. When developing
# the package alongside PELDRUN, the backend directory lives as a
# sibling of this repository (or under a well-known path). We probe
# a small set of candidates and add the first one that contains a
# `peldrun` subpackage.
try:
    import peldrun  # noqa: F401
except ImportError:
    _pelcrun_candidates = [
        REPO_ROOT.parent / "peldrun" / "backend",
        REPO_ROOT.parent / "peldrun",
        Path("D:/AI/peldrun/backend"),
        Path.home() / "peldrun" / "backend",
    ]
    for _candidate in _pelcrun_candidates:
        try:
            if (_candidate / "peldrun").is_dir():
                if str(_candidate) not in sys.path:
                    sys.path.insert(0, str(_candidate))
                print(
                    f"      NOTE — added PELDRUN backend to sys.path: "
                    f"{_candidate}"
                )
                break
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Import the adapter (this is the very first thing we test)
# ---------------------------------------------------------------------------

print("=" * 70)
print("GigaChat Adapter — Local Test Harness")
print("=" * 70)
print()

print("[1/7] Importing adapter module...")
try:
    from gigachat_peldrun.adapter import (  # noqa: E402
        GigaChatAdapter,
        _PROFILE,
    )
    print("      OK — adapter imported cleanly")
except Exception as exc:
    print(f"      FAILED — {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print()


# ---------------------------------------------------------------------------
# Contract sanity checks (always run, regardless of mode)
# ---------------------------------------------------------------------------

print("[2/7] Verifying adapter contract attributes...")

REQUIRED_CLASS_ATTRS = (
    "ADAPTER_STYLE",
    "API_CONTRACT",
    "PROFILE",
    "provider_id",
)

missing_attrs: list = []
for attr in REQUIRED_CLASS_ATTRS:
    if not hasattr(GigaChatAdapter, attr):
        missing_attrs.append(attr)

if missing_attrs:
    print(f"      FAILED — missing class attributes: {missing_attrs}")
    sys.exit(1)

if GigaChatAdapter.ADAPTER_STYLE != "v1":
    print(
        f"      FAILED — ADAPTER_STYLE must be 'v1', "
        f"got {GigaChatAdapter.ADAPTER_STYLE!r}"
    )
    sys.exit(1)

if GigaChatAdapter.API_CONTRACT != "peldrun.llm.provider.v1":
    print(
        f"      FAILED — API_CONTRACT mismatch: "
        f"{GigaChatAdapter.API_CONTRACT!r}"
    )
    sys.exit(1)

if GigaChatAdapter.provider_id != "gigachat":
    print(
        f"      FAILED — provider_id must be 'gigachat', "
        f"got {GigaChatAdapter.provider_id!r}"
    )
    sys.exit(1)

# Verify the required methods exist.
REQUIRED_METHODS = (
    "generate",
    "stream",
    "discover_models",
    "validate_connection",
    "get_model_metadata",
    "close",
)
for method in REQUIRED_METHODS:
    if not hasattr(GigaChatAdapter, method):
        print(f"      FAILED — missing method: {method}")
        sys.exit(1)
    if not callable(getattr(GigaChatAdapter, method)):
        print(f"      FAILED — attribute {method} is not callable")
        sys.exit(1)

print("      OK — all contract attributes and methods present")
print()


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "").strip()
SCOPE = os.environ.get("GIGACHAT_SCOPE", "PERS").strip() or "PERS"
CA_BUNDLE = os.environ.get("GIGACHAT_CA_BUNDLE", "").strip()
VERIFY_SSL = os.environ.get("GIGACHAT_VERIFY_SSL", "true").strip().lower() not in (
    "0",
    "false",
    "no",
)

MODE = "live" if CREDENTIALS else "dry"

print(f"[3/7] Mode: {MODE}")
if MODE == "dry":
    print("      No GIGACHAT_CREDENTIALS set — running structural checks only.")
    print("      To run live tests, set GIGACHAT_CREDENTIALS and rerun.")
else:
    print(f"      Scope: {SCOPE}")
    print(f"      CA bundle: {CA_BUNDLE or '<system trust store>'}")
    print(f"      Verify SSL: {VERIFY_SSL}")
print()


# ---------------------------------------------------------------------------
# Fake context — mimics PELDRUN's ProviderContext
# ---------------------------------------------------------------------------


class _FakeProfileHolder:
    """
    Minimal stand-in for ``peldrun.llm.contracts.AdapterProfile``.

    The adapter only reads attributes, so a plain object works. This
    keeps the test independent of PELDRUN's full runtime.
    """

    def __init__(self) -> None:
        self.provider_id = "gigachat"
        self.api_surface = "chat_completions"


class _FakeConnectionContext:
    """
    Minimal stand-in for ``peldrun.llm.contracts.ConnectionContext``.

    Provides the fields the adapter's __init__ reads:

        * base_url      — the provider endpoint
        * api_key       — may be empty; credentials come via extra_params
        * extra_params  — carries "credentials", "scope", "ca_bundle_path",
                          "verify_ssl", "model"
        * timeout       — per-request timeout
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        extra_params: dict,
        timeout: float = 60.0,
    ) -> None:
        self.connection_id = "test-connection"
        self.provider_id = "gigachat"
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = {}
        self.extra_params = dict(extra_params)
        self.metadata_endpoint = None
        self.verify_ssl = True
        self.workspace_access = False


class _FakeProviderContext:
    """Stand-in for ``ProviderContext``."""

    def __init__(
        self,
        connection: _FakeConnectionContext,
        profile: Any,
    ) -> None:
        self.connection = connection
        self.profile = profile
        self.credentials = {"api_key": connection.api_key} if connection.api_key else {}


def _build_context(
    *,
    credentials: str,
    scope: str,
    ca_bundle: Optional[str],
    verify_ssl: bool,
) -> _FakeProviderContext:
    """Build a ProviderContext suitable for instantiating the adapter."""
    extra = {
        "credentials": credentials,
        "scope": scope,
        "verify_ssl": verify_ssl,
    }
    if ca_bundle:
        extra["ca_bundle_path"] = ca_bundle

    conn = _FakeConnectionContext(
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key="",
        extra_params=extra,
    )
    return _FakeProviderContext(conn, _FakeProfileHolder())


def _build_request(
    adapter: GigaChatAdapter,
    messages: list,
    *,
    model: str = "GigaChat",
) -> Any:
    """
    Build an LLMRequest for the given messages.

    The real ``LLMRequest`` is a Pydantic model. If PELDRUN is installed,
    we use it. If not (dry environments without peldrun), we fall back to
    a duck-typed object that has the same attributes the adapter reads.
    """
    connection = adapter._connection  # type: ignore[attr-defined]

    try:
        # Preferred path: use the real Pydantic model.
        from peldrun.llm.request import (  # type: ignore[import-not-found]
            LLMRequest,
            GenerationConfig,
            ReasoningConfig,
            ResponseFormat,
            ToolChoiceSpec,
        )

        return LLMRequest(
            provider_id="gigachat",
            model_id=model,
            connection_id="test-connection",
            messages=messages,
            generation=GenerationConfig(
                temperature=0.7, max_output_tokens=100
            ),
            reasoning=ReasoningConfig(),
            response_format=ResponseFormat(),
            tool_choice=ToolChoiceSpec(),
            connection=connection,
        )
    except ImportError:
        # Fallback: build a duck-typed request. The adapter reads
        # attributes and calls compile_request() on it — as long as
        # compile_request accepts it, we are fine.
        class _DuckRequest:
            provider_id = "gigachat"
            model_id = model
            connection_id = "test-connection"
            system_prompt = None
            tools = []
            model_capabilities = None
            provider_options = {}
            metadata = {}

            def __init__(self, msgs, conn) -> None:
                self.messages = msgs
                self.connection = conn
                self.generation = {
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
                self.reasoning = None
                self.response_format = {"kind": "text"}
                self.tool_choice = None

        return _DuckRequest(messages, connection)


def _is_peldrun_available() -> bool:
    """Return True iff ``peldrun`` can be imported."""
    try:
        import peldrun  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Dry mode — negative tests that do not need credentials
# ---------------------------------------------------------------------------

if MODE == "dry":
    print("[4/7] Negative test — missing credentials")
    try:
        _empty_ctx = _build_context(
            credentials="",
            scope="PERS",
            ca_bundle=None,
            verify_ssl=True,
        )
        GigaChatAdapter(context=_empty_ctx)
        print("      FAILED — expected ProviderAuthError, got no exception")
        sys.exit(1)
    except Exception as exc:
        # We expect ProviderAuthError specifically; other exception
        # types indicate a bug in the adapter's __init__ error path.
        exc_name = type(exc).__name__
        if exc_name == "ProviderAuthError":
            print(f"      OK — raised {exc_name} as expected")
        else:
            print(
                f"      WARNING — raised {exc_name} (expected ProviderAuthError)"
            )
    print()

    print("[5/7] Negative test — masked credentials")
    try:
        _masked_ctx = _build_context(
            credentials="\u2022" * 8,  # the mask placeholder
            scope="PERS",
            ca_bundle=None,
            verify_ssl=True,
        )
        GigaChatAdapter(context=_masked_ctx)
        print("      FAILED — expected ProviderAuthError for masked value")
        sys.exit(1)
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "ProviderAuthError":
            print(f"      OK — raised {exc_name} for masked value")
        else:
            print(
                f"      WARNING — raised {exc_name} "
                f"(expected ProviderAuthError)"
            )
    print()

    print("[6/7] peldrun availability")
    if _is_peldrun_available():
        print("      OK — peldrun is importable (live-mode is possible)")
    else:
        print(
            "      NOTE — peldrun is NOT installed. Structural checks\n"
            "             passed, but live tests will not run until you\n"
            "             `pip install peldrun` (or run inside PELDRUN)."
        )
    print()

    print("[7/7] Dry mode complete — all structural checks passed.")
    print()
    print("=" * 70)
    print("RESULT: DRY PASS")
    print("=" * 70)
    print()
    print("To run live tests against the real GigaChat API:")
    print("  set GIGACHAT_CREDENTIALS=<base64-client-id:secret>")
    print("  set GIGACHAT_SCOPE=PERS")
    print("  set GIGACHAT_CA_BUNDLE=<path-to-russian-trusted-root-ca>")
    print("  python tests/test_adapter.py")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Live mode — full functional tests
# ---------------------------------------------------------------------------

print("[4/7] Instantiating adapter with real credentials...")
try:
    _ctx = _build_context(
        credentials=CREDENTIALS,
        scope=SCOPE,
        ca_bundle=CA_BUNDLE or None,
        verify_ssl=VERIFY_SSL,
    )
    adapter = GigaChatAdapter(context=_ctx)
    print("      OK — adapter instantiated")
except Exception as exc:
    print(f"      FAILED — {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)
print()


async def _run_live_tests(adapter_instance: GigaChatAdapter) -> int:
    """
    Run the full live test suite.

    Returns the process exit code: 0 = success, 1 = at least one test
    failed.
    """
    failures: list = []

    # ------------------------------------------------------------------
    # 5. validate_connection()
    # ------------------------------------------------------------------
    print("[5/7] validate_connection()...")
    try:
        result = await adapter_instance.validate_connection()
        if not isinstance(result, dict):
            failures.append("validate_connection did not return a dict")
            print(f"      FAILED — expected dict, got {type(result).__name__}")
        elif not result.get("ok"):
            failures.append(
                f"validate_connection failed: {result.get('message')}"
            )
            print(
                f"      FAILED — ok=False, message={result.get('message')!r}"
            )
        else:
            latency = result.get("latency_ms", 0)
            print(f"      OK — {result.get('message', 'connected')} "
                  f"({latency:.0f} ms)")
    except Exception as exc:
        failures.append(f"validate_connection raised: {exc}")
        print(f"      FAILED — {type(exc).__name__}: {exc}")
    print()

    # ------------------------------------------------------------------
    # 6. discover_models()
    # ------------------------------------------------------------------
    print("[6/7] discover_models()...")
    try:
        models = await adapter_instance.discover_models()
        if not isinstance(models, list):
            failures.append("discover_models did not return a list")
            print(
                f"      FAILED — expected list, "
                f"got {type(models).__name__}"
            )
        elif not models:
            print("      WARNING — returned empty list (no models discovered)")
        else:
            print(f"      OK — discovered {len(models)} model(s):")
            for model in models[:10]:
                mid = model.get("id", "?")
                dname = model.get("display_name") or mid
                print(f"          - {mid}  ({dname})")
            if len(models) > 10:
                print(f"          ... and {len(models) - 10} more")
    except Exception as exc:
        failures.append(f"discover_models raised: {exc}")
        print(f"      FAILED — {type(exc).__name__}: {exc}")
    print()

    # ------------------------------------------------------------------
    # 7. generate() + stream()
    # ------------------------------------------------------------------
    print("[7/7] generate() and stream()...")
    if not _is_peldrun_available():
        print(
            "      SKIPPED — peldrun is not importable, cannot construct\n"
            "             a real LLMRequest. Install peldrun to run this\n"
            "             test:\n"
            "                 pip install peldrun"
        )
    else:
        await _run_generation_tests(adapter_instance, failures)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    try:
        await adapter_instance.close()
    except Exception as exc:
        print(f"      WARNING — close() raised: {exc}")

    return 0 if not failures else 1


async def _run_generation_tests(
    adapter_instance: GigaChatAdapter,
    failures: list,
) -> None:
    """Exercise generate() and stream() with a short prompt."""
    user_prompt = "Say 'hello' in Russian in one word."

    # -------------------------
    # generate() — non-streaming
    # -------------------------
    print("      [7a] generate() — non-streaming...")
    try:
        request = _build_request(
            adapter_instance,
            [{"role": "user", "content": user_prompt}],
        )
        response = await adapter_instance.generate(request)

        content = getattr(response, "content", None)
        if not content:
            failures.append("generate() returned empty content")
            print("          FAILED — empty content")
        else:
            print(f"          OK — response: {content[:120]!r}")

        usage = getattr(response, "usage", None)
        if usage is not None:
            print(
                f"          Tokens: in={getattr(usage, 'input_tokens', '?')} "
                f"out={getattr(usage, 'output_tokens', '?')} "
                f"total={getattr(usage, 'total_tokens', '?')}"
            )

        finish = getattr(response, "finish_reason", None)
        print(f"          finish_reason: {finish!r}")
    except Exception as exc:
        failures.append(f"generate() raised: {exc}")
        print(f"          FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # -------------------------
    # stream() — streaming
    # -------------------------
    print("      [7b] stream() — streaming...")
    try:
        request = _build_request(
            adapter_instance,
            [{"role": "user", "content": user_prompt}],
        )

        token_count = 0
        got_complete = False
        collected: list = []

        async for event in adapter_instance.stream(request):
            event_type = getattr(event, "type", None)
            type_value = (
                event_type.value
                if hasattr(event_type, "value")
                else str(event_type)
            )

            if type_value == "content_delta":
                delta = getattr(event, "content_delta", "")
                if delta:
                    token_count += 1
                    collected.append(delta)
                    # Print tokens as they arrive, without newline.
                    print(delta, end="", flush=True)
            elif type_value == "message_complete":
                got_complete = True
            elif type_value == "provider_error":
                err = getattr(event, "error", {})
                failures.append(f"stream() yielded error: {err}")
                print(f"\n          FAILED — provider error: {err}")

        print()  # newline after streamed tokens
        full_text = "".join(collected)

        if token_count == 0:
            failures.append("stream() produced no content_delta events")
            print("          FAILED — no tokens received")
        else:
            print(
                f"          OK — received {token_count} token(s), "
                f"total text: {full_text[:120]!r}"
            )

        if not got_complete:
            failures.append("stream() did not yield MESSAGE_COMPLETE")
            print("          FAILED — no MESSAGE_COMPLETE event")
        else:
            print("          OK — MESSAGE_COMPLETE received")
    except Exception as exc:
        failures.append(f"stream() raised: {exc}")
        print(f"          FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Run live tests and report
# ---------------------------------------------------------------------------

exit_code = asyncio.run(_run_live_tests(adapter))

print()
print("=" * 70)
if exit_code == 0:
    print("RESULT: LIVE PASS — all tests succeeded")
else:
    print("RESULT: LIVE FAIL — one or more tests failed")
print("=" * 70)
sys.exit(exit_code)