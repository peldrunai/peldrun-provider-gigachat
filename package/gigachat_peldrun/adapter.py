# package/gigachat_peldrun/adapter.py
"""
GigaChat adapter for PELDRUN.

Translates PELDRUN's LLMRequest into GigaChat's wire protocol and back.
Runs inside an isolated worker subprocess managed by peldrun.llm.plugin_host.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from gigachat import GigaChat
from gigachat.exceptions import AuthenticationError as GigaAuthError
from gigachat.exceptions import GigaChatException
from gigachat.models import (
    # Old v1 interface — kept for the text-only path (working, no
    # reason to disturb). The old interface's ``Messages.attachments``
    # accepts file_ids but does NOT pass the image pixels to the model,
    # which is why vision never worked with it.
    Chat,
    ChatCompletion,
    Function,
    FunctionCall,
    Messages,
    MessagesRole,
    # New v2 interface — required for vision. Images are passed as
    # ``ChatContentPart.files`` entries, which the model actually sees.
    ChatMessage,
    ChatCompletionRequest,
    ChatContentPart,
    ChatContentFile,
    ChatFunctionCall,
    ChatFunctionResult,
    ChatModelOptions,
)

from peldrun.llm.client import (
    LLMResponse,
    TokenUsage,
    ToolCall,
    UsageSource,
)
from peldrun.llm.contracts import (
    AdapterFeatures,
    AdapterProfile,
    ConnectionContext,
    LLMStreamEvent,
    StreamEventType,
    ToolCallStyle,
)
from peldrun.llm.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from peldrun.llm.request import LLMRequest
from peldrun.llm.request_compiler import CompiledRequest, compile_request


logger = logging.getLogger("gigachat_peldrun.adapter")


# ---------------------------------------------------------------------------
# Global httpx verify override
# ---------------------------------------------------------------------------
#
# The GigaChat SDK has a persistent bug: when it performs the OAuth
# token request, it spawns its own httpx.Client (sync) internally
# WITHOUT passing the `ca_bundle_file` we gave to the GigaChat
# constructor. That internal client uses `verify=True` (certifi),
# which does NOT contain the Russian Trusted Root CA. The OAuth
# call therefore fails with:
#
#   ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
#   failed: self-signed certificate in chain
#
# The SDK's main API client DOES respect `ca_bundle_file`, but the
# first API call is always preceded by the OAuth call, so we never
# get that far.
#
# The only reliable workaround is to monkey-patch httpx at the module
# level: every httpx.Client / httpx.AsyncClient created inside this
# worker gets the SAME `verify` target we want. This applies to:
#
#   * The SDK's OAuth httpx.Client (sync)
#   * The SDK's main API httpx.AsyncClient (async)
#   * Any future httpx client the SDK spawns internally
#
# The override target is:
#
#   * ``<path>``    — verify against the CA bundle at <path>
#                     (certificate mode — the default when a CA is
#                     installed)
#   * ``False``     — skip verification entirely (insecure mode —
#                     dev-only, explicit user opt-in)
#   * ``None``      — no override; let each caller decide its own
#                     verify policy (system mode — rare, usually only
#                     happens when no CA bundle is installed)
#
# The patch is idempotent: installing it once is enough, and
# subsequent calls only update the ``_VERIFY_OVERRIDE`` global —
# existing httpx clients pick up the new value on their NEXT call
# (which is what we want, because we set the override BEFORE
# constructing the GigaChat client).
# ---------------------------------------------------------------------------

_VERIFY_OVERRIDE: Any = None


def _install_httpx_verify_override(target: Any) -> None:
    """
    Install (or update) a global httpx verify override.

    Parameters
    ----------
    target
        * ``str`` path  — verify against the CA bundle at that path.
        * ``False``     — skip TLS verification entirely.
        * ``None``      — no override (restore default behaviour for
                          future clients).

    Idempotent: patching the same class twice is a no-op.
    """
    global _VERIFY_OVERRIDE
    _VERIFY_OVERRIDE = target

    # Lazy import so the module can be introspected (e.g. by AST
    # tooling) without requiring httpx at parse time.
    import httpx as _httpx

    orig_async = _httpx.AsyncClient.__init__
    if not getattr(orig_async, "_peldrun_verify_patched", False):
        def _patched_async_init(self, *args, **kwargs):
            # ``verify`` is a keyword-only param on httpx.Client /
            # AsyncClient, so overriding via kwargs is safe.
            if _VERIFY_OVERRIDE is not None:
                kwargs["verify"] = _VERIFY_OVERRIDE
            orig_async(self, *args, **kwargs)

        _patched_async_init._peldrun_verify_patched = True  # type: ignore[attr-defined]
        _httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]

    orig_sync = _httpx.Client.__init__
    if not getattr(orig_sync, "_peldrun_verify_patched", False):
        def _patched_sync_init(self, *args, **kwargs):
            if _VERIFY_OVERRIDE is not None:
                kwargs["verify"] = _VERIFY_OVERRIDE
            orig_sync(self, *args, **kwargs)

        _patched_sync_init._peldrun_verify_patched = True  # type: ignore[attr-defined]
        _httpx.Client.__init__ = _patched_sync_init  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Adapter profile
# ---------------------------------------------------------------------------

_PROFILE = AdapterProfile(
    provider_id="gigachat",
    api_surface="chat_completions",
    features=AdapterFeatures(
        model_discovery=True,
        native_streaming=True,
        buffered_streaming=True,
        tool_calling=True,
        vision=True,
        usage_reporting=True,
        reasoning=False,
    ),
    reasoning_strategies=[],
    tool_call_style=ToolCallStyle.OPENAI_FUNCTION,
)


# ---------------------------------------------------------------------------
# Tool schema sanitization for GigaChat
# ---------------------------------------------------------------------------
#
# GigaChat's server-side Pydantic validator is unusually strict about
# the JSON Schema `format` field on tool parameters. It rejects ANY
# value other than:
#
#     "date", "date-time", "time"
#
# With:
#
#     422 Value properties.<field>.format must be one of:
#         'date', 'date-time', 'time'
#
# OpenAI-compatible tool schemas routinely use formats that GigaChat
# rejects — most commonly `"uri"` on URL parameters (browser, search,
# image_search, fetch tools). This means an agent run fails with a
# cryptic 422 the moment it tries to pass tools, even though the SAME
# tool definitions work perfectly against OpenAI, Anthropic, Google,
# and LM Studio.
#
# Fix: strip any non-whitelisted `format` value BEFORE sending the
# tool definition to GigaChat. The parameter itself (type, description,
# required flag, etc.) is preserved — only the cosmetic format hint is
# dropped. The model treats it as a plain string, which is exactly what
# GigaChat's schema accepts anyway.
#
# Recursion covers all common JSON Schema nesting points:
#   * `properties`            — object fields
#   * `items`                 — array element schemas
#   * `oneOf` / `anyOf` / `allOf` — union / intersection schemas
#
# Additional JSON Schema features GigaChat might also reject (e.g.
# `$ref`, `patternProperties`) are NOT stripped by this helper,
# because in practice PELDRUN tool schemas do not use them. If a future
# tool breaks on something else, extend this helper rather than
# weakening the sanitizer.

_GIGACHAT_ALLOWED_FORMATS: frozenset = frozenset({"date", "date-time", "time"})


# ---------------------------------------------------------------------------
# Image MIME detection
# ---------------------------------------------------------------------------
#
# GigaChat's file API rejects `application/octet-stream` with:
#
#   400 File format application/octet-stream is not supported
#
# The `gigachat` SDK defaults to that content type whenever it is given
# raw `bytes` without a filename or `content_type` hint. So we ALWAYS
# need to know the real MIME type before calling `aupload_file`.
#
# Two sources of truth:
#
#   1. The data URI header (``data:image/png;base64,...``). Modern
#      browsers set this correctly. But some produce bare
#      ``data:;base64,...`` or ``data:application/octet-stream;base64,...``
#      which are useless to GigaChat.
#
#   2. Magic bytes. These are authoritative for the small set of formats
#      GigaChat supports (PNG, JPEG, TIFF, BMP) and detect WebP / GIF
#      even when the provider rejects them — so we can fail fast with a
#      clear message instead of a generic 400.
#
# `_detect_image_mime` prefers magic bytes (ground truth) but accepts a
# data-URI hint as fallback for truncated/rare payloads.

_IMAGE_MAGIC_BYTES: tuple = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _detect_image_mime(image_bytes: bytes, hint: Optional[str] = None) -> str:
    """
    Return the best-known MIME type for ``image_bytes``.

    Detection order:
        1. Magic bytes — ground truth. Handles every format GigaChat
           supports plus WebP/GIF (which GigaChat rejects — see the
           caller for format filtering).
        2. ``hint`` — the MIME from the data URI, when magic bytes
           do not match.
        3. ``"image/png"`` — safe default; the SDK will still reject
           unsupported content, but we give it the best shot.

    Never raises; a corrupt or truncated payload falls through to the
    default.
    """
    if image_bytes:
        head = image_bytes[:32]
        # RIFF....WEBP
        if (
            len(head) >= 12
            and head[:4] == b"RIFF"
            and head[8:12] == b"WEBP"
        ):
            return "image/webp"
        for signature, mime in _IMAGE_MAGIC_BYTES:
            if head.startswith(signature):
                return mime

    if isinstance(hint, str) and hint.startswith("image/"):
        return hint.lower()

    return "image/png"


def _sanitize_parameters_for_gigachat(params: Any) -> Any:
    """
    Recursively strip non-whitelisted `format` values from a JSON schema.

    Idempotent. Safe to call on non-dict inputs (returns them unchanged).
    Never mutates the input — returns a fresh dict/list every time.
    """
    if isinstance(params, list):
        return [_sanitize_parameters_for_gigachat(item) for item in params]
    if not isinstance(params, dict):
        return params

    result = dict(params)

    # Drop the format hint if it is not in GigaChat's whitelist.
    fmt = result.get("format")
    if isinstance(fmt, str) and fmt not in _GIGACHAT_ALLOWED_FORMATS:
        result.pop("format", None)

    # Recurse into common schema sub-structures.
    if isinstance(result.get("properties"), dict):
        result["properties"] = {
            key: _sanitize_parameters_for_gigachat(val)
            for key, val in result["properties"].items()
        }
    if "items" in result:
        result["items"] = _sanitize_parameters_for_gigachat(result["items"])
    for union_key in ("oneOf", "anyOf", "allOf"):
        if isinstance(result.get(union_key), list):
            result[union_key] = [
                _sanitize_parameters_for_gigachat(sub)
                for sub in result[union_key]
            ]

    return result


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GigaChatAdapter:
    """Native GigaChat adapter."""

    ADAPTER_STYLE = "v1"
    API_CONTRACT = "peldrun.llm.provider.v1"
    PROFILE = _PROFILE
    provider_id = "gigachat"

    def __init__(self, context: Any) -> None:
        self._connection: ConnectionContext = context.connection
        self._profile: AdapterProfile = context.profile or _PROFILE
        self.profile = self._profile

        # ------------------------------------------------------------------
        # Image upload cache.
        #
        # GigaChat requires images to be uploaded to its file API first,
        # then referenced by file_id in ``Messages.attachments``. Without
        # caching, a single image that appears in N messages of the same
        # conversation (e.g. a user re-mentions the same attachment) would
        # be uploaded N times.
        #
        # Key: SHA-256 hash of the image bytes (first 16 hex chars, hex
        #      deterministic across processes).
        # Value: file_id returned by ``aupload_file``.
        #
        # The cache lives for the lifetime of the adapter instance — one
        # chat turn. That is the right scope: after the turn ends, the
        # adapter is closed (see ``close()``), a fresh adapter is created
        # for the next turn, and the cache is rebuilt. This means the
        # cache cannot become stale across GigaChat file-expiry windows.
        # ------------------------------------------------------------------
        self._uploaded_image_cache: Dict[str, tuple[str, str]] = {}

        # ------------------------------------------------------------------
        # Extract custom connection fields AND load model_facts.
        #
        # The plugin descriptor (``_peldrun_plugin``) contains the
        # ``package_dir`` pointing to the installed version directory.
        # We read ``manifest.json`` from there to load the per-model
        # facts declared by the package author.
        #
        # Why read from disk instead of the profile?
        #
        # The framework's AdapterProfile does NOT carry a model_facts
        # field — threading it through would require modifying 7
        # framework files (contracts, package_manager, custom_plugin,
        # dispatcher, supervisor, worker). Reading the manifest directly
        # is both simpler and arguably MORE correct: ``manifest.json``
        # is the authoritative source, not a copy propagated through
        # the supervisor's JSON-RPC profile_dict.
        #
        # Cost: one small JSON read per adapter instantiation. The
        # manifest is ~5 KB, cached by the OS filesystem. Negligible.
        # ------------------------------------------------------------------
        raw_extra = dict(self._connection.extra_params or {})
        plugin_info = raw_extra.get("_peldrun_plugin") or {}
        package_dir = plugin_info.get("package_dir")

        self._model_facts: Dict[str, Dict[str, Any]] = {}
        if isinstance(package_dir, str) and package_dir:
            try:
                manifest_path = Path(package_dir) / "manifest.json"
                if manifest_path.is_file():
                    manifest_data = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mf = manifest_data.get("model_facts")
                    if isinstance(mf, dict):
                        self._model_facts = {
                            str(k): dict(v)
                            for k, v in mf.items()
                            if isinstance(k, str) and isinstance(v, dict)
                        }
                        logger.info(
                            "GigaChatAdapter: loaded %d model_facts "
                            "entries from %s",
                            len(self._model_facts),
                            manifest_path,
                        )
                    else:
                        logger.debug(
                            "GigaChatAdapter: manifest at %s has no "
                            "model_facts block; per-model facts will "
                            "fall back to package-wide features.",
                            manifest_path,
                        )
                else:
                    logger.debug(
                        "GigaChatAdapter: manifest not found at %s; "
                        "model_facts unavailable.",
                        manifest_path,
                    )
            except Exception as exc:
                logger.warning(
                    "GigaChatAdapter: could not read model_facts from "
                    "%r: %s",
                    package_dir,
                    exc,
                )

        # Now strip the plugin descriptor — the rest of __init__ must
        # not see it.
        extra = {
            k: v for k, v in raw_extra.items() if k != "_peldrun_plugin"
        }

        credentials = extra.get("credentials") or self._connection.api_key or ""
        raw_scope = str(extra.get("scope") or "PERS").strip()

        # GigaChat SDK expects the FULL scope name (GIGACHAT_API_PERS,
        # GIGACHAT_API_B2B, GIGACHAT_API_CORP), not the short alias
        # (PERS/B2B/CORP) that our manifest's connection_fields declare.
        # Sber's OAuth endpoint rejects the short form with:
        #   {"code":1,"message":"scope data format invalid"}
        #
        # We accept BOTH forms so existing connections continue to work
        # without editing the manifest. Any value that already starts
        # with GIGACHAT_API_ is passed through unchanged.
        _SCOPE_MAP = {
            "PERS": "GIGACHAT_API_PERS",
            "B2B": "GIGACHAT_API_B2B",
            "CORP": "GIGACHAT_API_CORP",
        }
        if raw_scope.upper() in _SCOPE_MAP:
            scope = _SCOPE_MAP[raw_scope.upper()]
        else:
            scope = raw_scope

        ca_bundle_path = extra.get("ca_bundle_path") or None

        # Robust boolean conversion: the connection may store
        # ``verify_ssl`` as a JSON boolean (``true``/``false``) or as a
        # string (``"true"``/``"false"``) depending on the frontend
        # version that saved the connection. ``bool("false")`` returns
        # True (non-empty string is truthy), which would incorrectly
        # enable verification — the exact bug that produced
        # ``[SSL: CERTIFICATE_VERIFY_FAILED]`` on the OAuth call even
        # when the UI showed "Verify SSL: Disabled".
        raw_verify = extra.get("verify_ssl", True)
        if isinstance(raw_verify, str):
            verify_ssl = raw_verify.strip().lower() not in (
                "false", "0", "no", "off", ""
            )
        else:
            verify_ssl = bool(raw_verify)

        if not credentials:
            raise ProviderAuthError(
                "GigaChat requires an 'credentials' field "
                "(base64-encoded client_id:client_secret).",
                provider="gigachat",
            )

        # Guard against the masked placeholder leaking into the request.
        if credentials == "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022":
            raise ProviderAuthError(
                "Credentials appear to be masked. Please re-enter them.",
                provider="gigachat",
            )

        # ------------------------------------------------------------------
        # Resolve the CA bundle path (if any).
        #
        # The Certificate Manager stores the installed file under
        # ``storage/store/providers/gigachat/certificates/`` and the
        # HTTP layer (``_inject_installed_certificates`` in
        # routers/store/providers.py) copies that path into
        # ``fields["ca_bundle_path"]`` before constructing the
        # ConnectionContext. So by the time we get here, the value
        # SHOULD already be a real absolute path — unless the user
        # manually deleted the file after installing it. We verify
        # the file exists; if it does not, we log and downgrade.
        # ------------------------------------------------------------------
        ca_bundle_path_raw = extra.get("ca_bundle_path")
        ca_bundle_path: Optional[str] = None
        if isinstance(ca_bundle_path_raw, str) and ca_bundle_path_raw.strip():
            try:
                candidate = Path(ca_bundle_path_raw.strip())
                if candidate.is_file():
                    ca_bundle_path = str(candidate)
                else:
                    logger.warning(
                        "GigaChatAdapter: ca_bundle_path %r is not a "
                        "regular file. If you installed the Russian "
                        "Trusted Root CA via the Certificate Manager, "
                        "reinstall it — the file appears to have been "
                        "deleted from disk.",
                        ca_bundle_path_raw,
                    )
            except Exception as exc:
                logger.warning(
                    "GigaChatAdapter: could not validate ca_bundle_path "
                    "%r: %s",
                    ca_bundle_path_raw,
                    exc,
                )

        # ------------------------------------------------------------------
        # Determine the effective TLS verification mode.
        #
        # The connection exposes two independent flags and they MUST
        # NOT be conflated:
        #
        #   * ``ca_bundle_path`` — a CA file the user installed via the
        #                          Certificate Manager (Russian Trusted
        #                          Root CA, etc.).
        #   * ``verify_ssl``     — a dev-only escape hatch intended for
        #                          local mock servers, NOT for bypassing
        #                          TLS in production.
        #
        # A correct interpretation (this is the fix):
        #
        #   1. If a CA bundle EXISTS       → USE IT. ``verify_ssl`` is
        #      ignored: installing the certificate IS the user's
        #      explicit choice to keep verification enabled against a
        #      non-standard trust root.
        #
        #   2. No bundle + verify_ssl=True → fall back to the system
        #      trust store (certifi). Works for public endpoints,
        #      but Sber's Russian Root CA is NOT in certifi, so this
        #      will fail for GigaChat B2B/CORP unless the user
        #      installed the CA bundle first.
        #
        #   3. No bundle + verify_ssl=False → skip verification
        #      entirely. Dev-only. Monkey patches apply so the SDK's
        #      internal httpx clients also skip verification.
        #
        # The previous implementation did the OPPOSITE: whenever
        # ``verify_ssl`` was False it IGNORED the CA bundle entirely
        # (passing ``ca_bundle_file=None``). That made the
        # certificate the user installed literally useless — the
        # whole point of the Certificate Manager is to make TLS
        # work WITHOUT disabling verification. Ignoring the bundle
        # also forced the user into the "insecure" path even when
        # they had a perfectly valid trust root, and the resulting
        # monkey patches obscured the failure mode (TLS errors
        # surfaced as opaque empty-exception messages).
        # ------------------------------------------------------------------
        if ca_bundle_path is not None:
            effective_verify_mode = "certificate"
        elif verify_ssl:
            effective_verify_mode = "system"
        else:
            effective_verify_mode = "insecure"

        logger.info(
            "GigaChatAdapter: effective verify mode = %r "
            "(ca_bundle_path=%r, verify_ssl=%s).",
            effective_verify_mode,
            ca_bundle_path,
            verify_ssl,
        )

        # ------------------------------------------------------------------
        # Monkey-patch httpx ONLY in the ``insecure`` mode.
        #
        # The GigaChat SDK's OAuth code path (and, on some versions, its
        # main client) creates httpx clients internally without
        # honouring ``verify_ssl_certs=False``. When the user truly
        # wants to skip verification, the only reliable way is to force
        # ``verify=False`` on every httpx client the SDK spawns.
        #
        # In ``certificate`` and ``system`` modes the SDK's
        # ``ca_bundle_file`` / ``certifi`` path works correctly — we
        # must NOT patch httpx there, otherwise we would silently
        # disable the very verification we want to keep.
        #
        # Each patch is idempotent (guarded by a marker attribute) so a
        # re-instantiation in the same worker does not stack patches.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Install the global httpx verify override per mode.
        #
        # Rationale: the GigaChat SDK's OAuth code path ignores the
        # ``ca_bundle_file`` constructor argument and creates its own
        # internal httpx.Client with ``verify=True`` (certifi), which
        # does NOT contain the Russian Root CA. Without patching
        # httpx, the OAuth call ALWAYS fails with
        # ``self-signed certificate in chain`` — no matter how correct
        # the installed CA bundle is.
        #
        # The override forces every httpx client spawned by the SDK
        # (sync OAuth + async API) to verify against the same target.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Configure TLS trust WITHOUT the httpx patch.
        #
        # Earlier versions of this adapter monkey-patched
        # ``httpx.AsyncClient.__init__`` to force ``verify=<path>`` on
        # every client the SDK spawns. That worked for verifying the
        # CA bundle, but introduced a hard freeze:
        #
        #   * The patched client held an SSLContext created from the
        #     CA file.
        #   * On ``aclose()``, httpx waited for the SSL shutdown
        #     handshake of every live connection.
        #   * On Windows, those SSL shutdowns occasionally hang if the
        #     connection was already half-closed (e.g. after a failed
        #     TLS retry).
        #   * ``adapter.close()`` -> ``supervisor.shutdown()`` ->
        #     worker.aclose() blocked the main event loop forever;
        #     only Ctrl+C could interrupt it.
        #
        # Instead of patching httpx, we now rely on the standard
        # Python/OpenSSL mechanism: the ``SSL_CERT_FILE`` environment
        # variable. ``ssl.create_default_context()`` (which httpx uses
        # whenever ``verify=True``) automatically reads this variable
        # and uses the specified file as the trust root.
        #
        # Benefits:
        #   * No patching — nothing to break on upgrade.
        #   * No SSLContext held by the adapter — the SDK's own
        #     clients use the env var when they create their contexts.
        #   * Works for the SDK's internal OAuth client without any
        #     additional hacks.
        #   * Compatible with requests / urllib3 (the GigaChat SDK
        #     may use either layer depending on version).
        #
        # The env vars are set in the WORKER process only (this file
        # runs exclusively inside the worker subprocess), so they do
        # not leak into the main backend.
        # ------------------------------------------------------------------
        if effective_verify_mode == "certificate":
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle_path
            os.environ["CURL_CA_BUNDLE"] = ca_bundle_path
            logger.info(
                "GigaChatAdapter: SSL_CERT_FILE set to %s "
                "(verification via custom CA bundle).",
                ca_bundle_path,
            )
        elif effective_verify_mode == "insecure":
            # Dev-only. Do NOT set the env vars — let the SDK use
            # whatever it wants. The GigaChat constructor's
            # verify_ssl_certs=False takes care of the API client;
            # the OAuth client will still verify against certifi,
            # which fails for Sber — but that only matters if the
            # user explicitly chose "insecure" mode and knows the
            # consequences.
            logger.warning(
                "GigaChatAdapter: insecure mode — TLS verification "
                "will likely FAIL for the OAuth call (Sber requires "
                "the Russian Root CA)."
            )

        # ------------------------------------------------------------------
        # Construct the GigaChat client per the effective mode.
        #
        # The constructor arguments below mirror the effective_verify_mode
        # for the SDK's OWN behaviour (as opposed to the httpx override,
        # which handles the SDK's internal OAuth client). Both layers
        # are needed because the SDK does not consistently respect its
        # own constructor flags across versions.
        # ------------------------------------------------------------------
        if effective_verify_mode == "certificate":
            self._client = GigaChat(
                credentials=credentials,
                scope=scope,
                verify_ssl_certs=True,
                ca_bundle_file=ca_bundle_path,
                model=str(extra.get("model") or "GigaChat"),
                timeout=float(self._connection.timeout or 60.0),
            )
            logger.info(
                "GigaChatAdapter: verification ON via custom CA bundle "
                "(%s) + httpx override installed.",
                ca_bundle_path,
            )
        elif effective_verify_mode == "system":
            self._client = GigaChat(
                credentials=credentials,
                scope=scope,
                verify_ssl_certs=True,
                ca_bundle_file=None,
                model=str(extra.get("model") or "GigaChat"),
                timeout=float(self._connection.timeout or 60.0),
            )
            logger.warning(
                "GigaChatAdapter: verification ON via system trust store "
                "(no custom CA bundle configured). If TLS fails with "
                "CERTIFICATE_VERIFY_FAILED, install the Russian Trusted "
                "Root CA via the Certificate Manager."
            )
        else:
            self._client = GigaChat(
                credentials=credentials,
                scope=scope,
                verify_ssl_certs=False,
                ca_bundle_file=None,
                model=str(extra.get("model") or "GigaChat"),
                timeout=float(self._connection.timeout or 60.0),
            )
            logger.warning(
                "GigaChatAdapter: verification DISABLED. This is a "
                "dev-only mode — do NOT use it in production."
            )

        logger.debug(
            "GigaChatAdapter ready: scope=%s mode=%s ca_bundle=%s "
            "httpx_override=%r",
            scope,
            effective_verify_mode,
            ca_bundle_path or "<system>",
            _VERIFY_OVERRIDE,
        )

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        compiled = compile_request(request, self._profile)
        try:
            if self._has_images(compiled):
                # Multimodal path: use v2 ChatCompletionRequest so the
                # model actually receives the image pixels via
                # ChatContentPart.files.
                #
                # IMPORTANT: use ``achat.create`` (async client) not
                # ``chat.create`` (sync client). The sync client's
                # ``create`` returns a ChatCompletion object, not a
                # coroutine — ``await`` on it raises TypeError. The
                # async client's ``create`` is a proper coroutine.
                req = await self._build_chat_completion_request(
                    compiled, stream=False
                )
                try:
                    result = await self._client.achat.create(req)
                except (AttributeError, TypeError):
                    # Defensive fallback: some SDK builds expose the v2
                    # route as ``achat(req)`` (via ChatAsyncClient
                    # ``__call__``) rather than ``achat.create(req)``.
                    result = await self._client.achat(req)
            else:
                # Text-only path: keep the working v1 interface.
                chat = await self._build_chat_payload(compiled)
                result = await self._client.achat(chat)
        except GigaAuthError as exc:
            raise ProviderAuthError(str(exc), provider="gigachat") from exc
        except GigaChatException as exc:
            raise self._map_exception(exc) from exc
        except Exception as exc:
            # ------------------------------------------------------------------
            # [FIX-VALIDATION] GigaChat-2-Max sometimes returns a v1-style
            # response (with ``choices``) while the installed SDK expects
            # a v2-style response (with ``messages``). This triggers a
            # Pydantic ``ValidationError`` inside the SDK *before* our
            # ``_convert_response`` can normalise the shape.
            #
            # We catch the validation error here and attempt to extract
            # the raw response data from the error context. The
            # ``_convert_response`` helper already knows how to handle
            # both shapes, so once we get the raw dict it will work.
            # ------------------------------------------------------------------
            from pydantic import ValidationError

            if isinstance(exc, ValidationError):
                # The exception exposes the raw input via ``exc.errors()``.
                # Each error entry carries an ``input`` key with the
                # original payload. We scan for the first entry that
                # contains a dict (the raw response) and reconstruct a
                # minimal object that ``_convert_response`` can consume.
                raw_response = None
                for err in exc.errors():
                    inp = err.get("input")
                    if isinstance(inp, dict):
                        raw_response = inp
                        break

                if raw_response is not None:
                    logger.warning(
                        "GigaChatAdapter: caught Pydantic ValidationError "
                        "on ChatCompletionResponse; attempting to salvage "
                        "raw response. Errors: %s",
                        exc.errors(),
                    )
                    # Wrap the raw dict in a lightweight object that
                    # exposes the attributes ``_convert_response`` reads.
                    # We use a simple namespace class because Pydantic
                    # models cannot be constructed from a v1 payload
                    # without triggering the same validation error.
                    class _RawResponse:
                        def __init__(self, data):
                            self._data = data
                            # v1 shape: choices[0].message
                            self.choices = data.get("choices") or []
                            # v2 shape: messages[0]
                            self.messages = data.get("messages") or []
                            self.usage = data.get("usage")

                    result = _RawResponse(raw_response)
                else:
                    raise self._map_exception(exc) from exc
            else:
                raise self._map_exception(exc) from exc

        return self._convert_response(result, compiled.model)

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    async def stream(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        compiled = compile_request(request, self._profile)
        seq = 0
        captured_usage: Optional[TokenUsage] = None
        finish_reason: Optional[str] = None

        # Diagnostic: log which path is being taken and how many
        # messages carry images. This makes it easy to confirm from a
        # single log line whether the multimodal path is active.
        _has_img = self._has_images(compiled)
        logger.info(
            "GigaChatAdapter[stream]: has_images=%s, messages=%d, "
            "model=%s",
            _has_img,
            len(compiled.messages),
            compiled.model,
        )

        try:
            # ------------------------------------------------------------------
            # Dispatch based on content:
            #
            #   * Multimodal → non-streaming via v2 (only path that
            #     delivers image pixels).
            #   * Text-only → TRUE streaming via ``astream()``.
            #
            # The prior implementation awaited the full response and
            # emitted a single CONTENT_DELTA with the complete text.
            # That made the UI appear frozen for the entire LLM call
            # (10-30 seconds per step for GigaChat) and then dumped
            # everything at once — the exact "no live stream" symptom
            # reported by the user, contrasting sharply with LM Studio
            # whose native adapter streams token-by-token via httpx.
            # ------------------------------------------------------------------

            # ==============================================================
            # Multimodal path — non-streaming (GigaChat v2 image analysis
            # does not expose incremental chunks).
            # ==============================================================
            if self._has_images(compiled):
                req = await self._build_chat_completion_request(
                    compiled, stream=False
                )
                try:
                    response = await self._client.achat.create(req)
                except (AttributeError, TypeError):
                    try:
                        response = await self._client.achat(req)
                    except (AttributeError, TypeError):
                        # Last resort: sync path via chat.create.
                        response = self._client.chat.create(req)

                llm_resp = self._convert_response(response, compiled.model)
                if llm_resp.content:
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.CONTENT_DELTA,
                        content_delta=llm_resp.content,
                        sequence=seq,
                    )
                for tc in llm_resp.tool_calls:
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        sequence=seq,
                    )
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.TOOL_CALL_COMPLETE,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_arguments_complete=tc.arguments,
                        sequence=seq,
                    )
                seq += 1
                yield LLMStreamEvent(
                    type=StreamEventType.MESSAGE_COMPLETE,
                    finish_reason=llm_resp.finish_reason or "stop",
                    usage=llm_resp.usage,
                    sequence=seq,
                )
                return

            # ==============================================================
            # Text-only path — TRUE streaming via ``astream()``.
            #
            # Chunk shape depends on the installed SDK version:
            #
            #   * v2 (current SDK): ``chunk.messages[0]`` with
            #     ``content: List[ChatContentPart]`` and
            #     ``function_call: Optional[ChatFunctionCall]``.
            #   * v1 (older SDK): ``chunk.choices[0].delta`` with
            #     ``content: Optional[str]`` and
            #     ``function_call: Optional[DeltaFunctionCall]``.
            #
            # Both shapes are parsed below, so the adapter works with
            # either SDK generation. Content deltas are yielded as they
            # arrive, giving the frontend live token-by-token rendering
            # identical to the LM Studio native adapter.
            # ==============================================================
            chat = await self._build_chat_payload(compiled, stream=True)

            accumulated_text: List[str] = []
            fn_name: Optional[str] = None
            fn_args_dict: Dict[str, Any] = {}
            fn_args_raw: str = ""
            last_finish_reason: Optional[str] = None
            got_any_chunk = False

            async for chunk in self._client.astream(chat):
                got_any_chunk = True

                # Usage may appear on the final chunk.
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    captured_usage = TokenUsage(
                        input_tokens=getattr(chunk_usage, "prompt_tokens", None),
                        output_tokens=getattr(chunk_usage, "completion_tokens", None),
                        total_tokens=getattr(chunk_usage, "total_tokens", None),
                        source=UsageSource.PROVIDER,
                    )

                # ------------------------------------------------------
                # v2 shape: chunk.messages[0]
                # ------------------------------------------------------
                v2_messages = getattr(chunk, "messages", None)
                if v2_messages:
                    v2_msg = v2_messages[0] if v2_messages else None
                    if v2_msg is not None:
                        raw_content = getattr(v2_msg, "content", None)
                        if isinstance(raw_content, list):
                            for part in raw_content:
                                txt = (
                                    part.get("text")
                                    if isinstance(part, dict)
                                    else getattr(part, "text", None)
                                )
                                if isinstance(txt, str) and txt:
                                    accumulated_text.append(txt)
                                    seq += 1
                                    yield LLMStreamEvent(
                                        type=StreamEventType.CONTENT_DELTA,
                                        content_delta=txt,
                                        sequence=seq,
                                    )
                        elif isinstance(raw_content, str) and raw_content:
                            accumulated_text.append(raw_content)
                            seq += 1
                            yield LLMStreamEvent(
                                type=StreamEventType.CONTENT_DELTA,
                                content_delta=raw_content,
                                sequence=seq,
                            )

                        fc = getattr(v2_msg, "function_call", None)
                        if fc is not None:
                            if getattr(fc, "name", None):
                                fn_name = str(fc.name)
                            p_args = (
                                getattr(fc, "partial_arguments", None)
                                or getattr(fc, "arguments", None)
                            )
                            if isinstance(p_args, dict):
                                fn_args_dict = p_args
                            elif isinstance(p_args, str) and p_args:
                                fn_args_raw += p_args

                        v2_fr = getattr(v2_msg, "finish_reason", None)
                        if isinstance(v2_fr, str) and v2_fr:
                            last_finish_reason = v2_fr

                # ------------------------------------------------------
                # v1 shape: chunk.choices[0].delta
                # ------------------------------------------------------
                v1_choices = getattr(chunk, "choices", None)
                if v1_choices:
                    v1_choice = v1_choices[0] if v1_choices else None
                    if v1_choice is not None:
                        v1_delta = getattr(v1_choice, "delta", None)
                        if v1_delta is not None:
                            d_txt = getattr(v1_delta, "content", None)
                            if isinstance(d_txt, str) and d_txt:
                                accumulated_text.append(d_txt)
                                seq += 1
                                yield LLMStreamEvent(
                                    type=StreamEventType.CONTENT_DELTA,
                                    content_delta=d_txt,
                                    sequence=seq,
                                )

                            d_fc = getattr(v1_delta, "function_call", None)
                            if d_fc is not None:
                                if getattr(d_fc, "name", None):
                                    fn_name = str(d_fc.name)
                                d_args = (
                                    getattr(d_fc, "arguments", None)
                                    or getattr(d_fc, "partial_arguments", None)
                                )
                                if isinstance(d_args, dict):
                                    fn_args_dict = d_args
                                elif isinstance(d_args, str) and d_args:
                                    fn_args_raw += d_args

                        v1_fr = getattr(v1_choice, "finish_reason", None)
                        if isinstance(v1_fr, str) and v1_fr:
                            last_finish_reason = v1_fr

            # Fallback if streaming yielded nothing (defensive — should
            # not happen for a working SDK, but a broken version must
            # not crash the run).
            if not got_any_chunk:
                logger.warning(
                    "GigaChatAdapter[stream]: astream() yielded zero "
                    "chunks for model=%s; falling back to non-streaming.",
                    compiled.model,
                )
                chat_nostream = await self._build_chat_payload(
                    compiled, stream=False
                )
                try:
                    response = await self._client.achat(chat_nostream)
                except (AttributeError, TypeError):
                    response = self._client.chat(chat_nostream)
                llm_resp = self._convert_response(response, compiled.model)
                if llm_resp.content:
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.CONTENT_DELTA,
                        content_delta=llm_resp.content,
                        sequence=seq,
                    )
                for tc in llm_resp.tool_calls:
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        sequence=seq,
                    )
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.TOOL_CALL_COMPLETE,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_arguments_complete=tc.arguments,
                        sequence=seq,
                    )
                seq += 1
                yield LLMStreamEvent(
                    type=StreamEventType.MESSAGE_COMPLETE,
                    finish_reason=llm_resp.finish_reason or "stop",
                    usage=llm_resp.usage,
                    sequence=seq,
                )
                return

            # ------------------------------------------------------
            # Emit the assembled tool call (if any) AFTER streaming
            # the text deltas. GigaChat sends the function call
            # metadata in the final chunk(s), so we can only build it
            # once the stream ends.
            # ------------------------------------------------------
            if fn_name:
                final_args: Dict[str, Any] = {}
                if fn_args_dict:
                    final_args = fn_args_dict
                elif fn_args_raw:
                    try:
                        parsed = json.loads(fn_args_raw)
                        if isinstance(parsed, dict):
                            final_args = parsed
                        else:
                            final_args = {"value": parsed}
                    except Exception:
                        final_args = {"raw": fn_args_raw}

                call_id = f"call_{uuid.uuid4().hex[:8]}"
                seq += 1
                yield LLMStreamEvent(
                    type=StreamEventType.TOOL_CALL_START,
                    tool_call_id=call_id,
                    tool_name=fn_name,
                    sequence=seq,
                )
                seq += 1
                yield LLMStreamEvent(
                    type=StreamEventType.TOOL_CALL_COMPLETE,
                    tool_call_id=call_id,
                    tool_name=fn_name,
                    tool_arguments_complete=final_args,
                    sequence=seq,
                )
                # Normalize GigaChat's "function_call" marker to the
                # OpenAI-compatible "tool_calls" name that the
                # downstream normalizer expects.
                if last_finish_reason in (None, "function_call"):
                    last_finish_reason = "tool_calls"

            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason=last_finish_reason or "stop",
                usage=captured_usage,
                sequence=seq,
            )
            return

        except asyncio.CancelledError:
            # Supervisor cancelled the stream (user pressed Stop).
            try:
                await self._client.aclose()
            except Exception:
                pass
            raise

        except GigaAuthError as exc:
            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.PROVIDER_ERROR,
                error=ProviderAuthError(
                    str(exc), provider="gigachat"
                ).to_dict(),
                sequence=seq,
            )
            return

        except GigaChatException as exc:
            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.PROVIDER_ERROR,
                error=self._map_exception(exc).to_dict(),
                sequence=seq,
            )
            return

        except Exception as exc:
            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.PROVIDER_ERROR,
                error=self._map_exception(exc).to_dict(),
                sequence=seq,
            )
            return

        # Defensive fallback: SDK ended without a finish_reason.
        if finish_reason is None:
            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason="stop",
                usage=captured_usage,
                sequence=seq,
            )

    async def discover_models(
        self, connection: Optional[ConnectionContext] = None
    ) -> List[Dict[str, Any]]:
        # ------------------------------------------------------------------
        # Two-layer discovery:
        #
        #   1. LIVE query — ``GET /models`` against Sber's API.
        #      This is the AUTHORITATIVE source for:
        #        * which models exist right now,
        #        * their exact ``id`` strings (required for dispatch),
        #        * their ``type`` (chat / aicheck / embedder).
        #
        #      Sber may add, rename, or deprecate models at any time.
        #      Only the live query reflects that. Never hardcode the
        #      model list.
        #
        #   2. ``model_facts`` overlay — read from ``manifest.json``.
        #      The API does NOT expose ``context_length``,
        #      ``max_output_tokens``, vision / tools / reasoning flags,
        #      or audio support. Sber publishes those facts only in
        #      their HTML documentation. The manifest encodes them
        #      once so every user sees accurate values.
        #
        # When ``model_facts`` does not contain an entry for a model
        # returned by the live query, the package-wide ``features``
        # block is used as a fallback. This is the correct behaviour
        # for a brand-new model that Sber releases before the package
        # is updated — the user still gets a working entry, just with
        # coarse-grained capability defaults.
        # ------------------------------------------------------------------
        logger.info(
            "GigaChatAdapter[discover]: SSL_CERT_FILE=%r, "
            "REQUESTS_CA_BUNDLE=%r",
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("REQUESTS_CA_BUNDLE"),
        )

        try:
            models = await self._client.aget_models()
        except Exception as exc:
            err_msg = str(exc) or repr(exc) or "<empty exception>"
            cause = exc.__cause__ or exc.__context__
            cause_msg = ""
            if cause is not None:
                cause_msg = (
                    f" | caused by {type(cause).__name__}: "
                    f"{str(cause) or repr(cause) or '<empty>'}"
                )
            logger.warning(
                "GigaChat model discovery failed: %s: %s%s",
                type(exc).__name__,
                err_msg,
                cause_msg,
            )
            return []

        # Package-wide fallback values (used when model_facts has no
        # entry for a given model).
        feats = getattr(self._profile, "features", None)
        default_vision = bool(getattr(feats, "vision", False))
        default_tools = bool(getattr(feats, "tool_calling", False))
        default_reasoning = bool(getattr(feats, "reasoning", False))
        default_audio = bool(getattr(feats, "audio", False))

        # Per-model overrides from ``manifest.json::model_facts``.
        #
        # Loaded once in ``__init__`` by reading ``manifest.json``
        # directly from ``package_dir`` (the profile does not carry
        # this field — see the __init__ comment for the rationale).
        #
        # The dict is intentionally read fresh on every adapter
        # instantiation, so a user who edits the manifest on disk and
        # restarts the backend sees the change immediately. The worker
        # subprocess is respawned on adapter re-creation.
        model_facts: Dict[str, Dict[str, Any]] = self._model_facts
        if not isinstance(model_facts, dict):
            model_facts = {}

        results: List[Dict[str, Any]] = []

        for model in getattr(models, "data", None) or []:
            model_id = (
                getattr(model, "id_", None)
                or getattr(model, "id", None)
                or ""
            )
            if not model_id:
                continue

            model_id_str = str(model_id)

            # ----------------------------------------------------------
            # Model type filter.
            #
            # Sber's gRPC API declares ``type`` (chat / aicheck /
            # embedder). The REST API does not, but future versions
            # might. Skip non-chat models — they cannot serve
            # /chat/completions.
            # ----------------------------------------------------------
            model_type = getattr(model, "type", None)
            if isinstance(model_type, str) and model_type.lower() in (
                "aicheck", "embedder", "embedding"
            ):
                logger.debug(
                    "Skipping non-chat model %r (type=%r).",
                    model_id_str,
                    model_type,
                )
                continue

            # ----------------------------------------------------------
            # Resolve model_facts entry (may be empty).
            # ----------------------------------------------------------
            facts = model_facts.get(model_id_str)
            if not isinstance(facts, dict):
                facts = {}

            # ----------------------------------------------------------
            # Context window and output budget.
            #
            # Prefer model_facts (authoritative from docs); fall back to
            # a live ``context_length`` attribute if the SDK ever starts
            # returning one (defensive: today it does not).
            # ----------------------------------------------------------
            ctx = facts.get("max_context_length")
            if not isinstance(ctx, int) or ctx <= 0:
                ctx = getattr(model, "context_length", None)
            max_out = facts.get("max_output_tokens")
            if not isinstance(max_out, int) or max_out <= 0:
                max_out = None

            # ----------------------------------------------------------
            # Capabilities: facts override → package features fallback.
            # ----------------------------------------------------------
            def _bool_from(facts_key: str, fallback: bool) -> bool:
                val = facts.get(facts_key)
                return bool(val) if isinstance(val, bool) else fallback

            vision_supported = _bool_from("vision", default_vision)
            tools_supported = _bool_from("tools", default_tools)
            reasoning_supported = _bool_from("reasoning", default_reasoning)
            audio_supported = _bool_from("audio", default_audio)

            # Provenance:
            #   * Facts present → "provider_catalog" (declared by the
            #     package author based on official docs).
            #   * Facts absent → also "provider_catalog" but the value
            #     came from the coarse package-wide features block. The
            #     source label stays the same because both come from the
            #     manifest; the UI does not need to distinguish them.
            capability_sources: Dict[str, str] = {
                "vision": "provider_catalog",
                "trained_for_tool_use": "provider_catalog",
                "reasoning": "provider_catalog",
            }

            entry: Dict[str, Any] = {
                "id": model_id_str,
                "display_name": getattr(model, "name", None) or model_id_str,
                "type": "llm",
                "capabilities": {
                    "vision": vision_supported,
                    "trained_for_tool_use": tools_supported,
                    "reasoning": reasoning_supported,
                },
                "capability_sources": capability_sources,
            }

            if isinstance(ctx, int) and ctx > 0:
                entry["max_context_length"] = ctx
            if isinstance(max_out, int) and max_out > 0:
                entry["max_output_tokens"] = max_out

            # Audio is reported via modalities (not a capability bool).
            modalities_input = ["text"]
            if vision_supported:
                modalities_input.append("image")
            if audio_supported:
                modalities_input.append("audio")
            if modalities_input != ["text"]:
                entry["modalities"] = {
                    "input": modalities_input,
                    "output": ["text"],
                    "source": "provider_catalog",
                }

            if reasoning_supported:
                # GigaChat models ALWAYS think internally — there is
                # no toggle exposed by the SDK or the API. Sber only
                # advertises "reasoning" as a marketing term, not as
                # a controllable parameter. Use "automatic" so the UI
                # shows "always on" (no toggle) rather than a fake
                # Enable/Disable button.
                entry["capability_details"] = {
                    "reasoning": {
                        "mode": "automatic",
                        "state": "supported",
                    }
                }

            results.append(entry)

        logger.info(
            "GigaChatAdapter[discover]: %d model(s) returned, "
            "%d enriched from model_facts.",
            len(results),
            sum(1 for m in results if m["id"] in model_facts),
        )
        return results

    # ------------------------------------------------------------------
    # validate_connection()
    # ------------------------------------------------------------------

    async def validate_connection(
        self, connection: Optional[ConnectionContext] = None
    ) -> Dict[str, Any]:
        started = time.time()
        try:
            await self._client.aget_models()
            latency_ms = round((time.time() - started) * 1000.0, 2)
            return {
                "ok": True,
                "message": "GigaChat API reachable.",
                "latency_ms": latency_ms,
            }
        except GigaAuthError as exc:
            return {
                "ok": False,
                "message": f"Authentication failed: {exc}",
                "latency_ms": round((time.time() - started) * 1000.0, 2),
                "code": "auth_failed",
                "status_code": 401,
            }
        except Exception as exc:
            # ------------------------------------------------------------------
            # Surface the real error to the UI.
            #
            # ``str(exc)`` is empty for several common failures
            # (asyncio.TimeoutError, socket.gaierror with empty args,
            # custom exceptions). The UI showed "Connection Failed"
            # with no actionable message. The user could not tell
            # whether the issue was network, TLS, OAuth, or timeout.
            #
            # The message below includes the exception type and the
            # underlying __cause__ when present, so a single log line
            # is enough to diagnose.
            # ------------------------------------------------------------------
            err_msg = str(exc) or repr(exc) or "<empty exception>"
            cause = exc.__cause__ or exc.__context__
            cause_msg = ""
            if cause is not None:
                cause_msg = (
                    f" | caused by {type(cause).__name__}: "
                    f"{str(cause) or repr(cause) or '<empty>'}"
                )
            full_message = (
                f"{type(exc).__name__}: {err_msg}{cause_msg}"
            )
            return {
                "ok": False,
                "message": full_message,
                "latency_ms": round((time.time() - started) * 1000.0, 2),
            }

    # ------------------------------------------------------------------
    # get_model_metadata()
    # ------------------------------------------------------------------

    async def get_model_metadata(
        self, model_id: str
    ) -> Optional[Dict[str, Any]]:
        for model in await self.discover_models():
            if model["id"] == model_id:
                return model
        return None

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    async def close(self) -> None:
        # ------------------------------------------------------------------
        # DO NOT call self._client.aclose() here.
        #
        # The GigaChat SDK's aclose() has been observed to hang
        # indefinitely on Windows when the underlying httpx client is
        # in a half-closed SSL state. The hang is INSIDE OpenSSL's
        # synchronous SSL_shutdown() call, which means
        # ``asyncio.wait_for`` cannot cancel it — the timeout fires
        # but the async wrapper has no way to interrupt C code.
        #
        # Consequences of NOT skipping:
        #   * Worker subprocess hangs on shutdown.
        #   * Supervisor.shutdown() blocks waiting for the worker.
        #   * The main backend event loop is blocked because
        #     supervisor.shutdown() runs on it.
        #   * Only Ctrl+C escapes (hard exit).
        #
        # Why skipping is SAFE:
        #   * This adapter runs inside an isolated worker subprocess.
        #     The subprocess is SIGTERM'd, then SIGKILL'd by the
        #     supervisor after a timeout.
        #   * The OS reclaims all TCP connections, file descriptors,
        #     and SSL state when the process dies. Nothing leaks.
        #   * The GigaChat SDK does NOT buffer any data that needs
        #     flushing — all state is server-side.
        #   * There is no graceful-close contract with the provider
        #     API that requires a TCP FIN handshake.
        #
        # If a future adapter needs graceful close (e.g. flushing an
        # audit log), it can call aclose() with a synchronous-worker
        # guard (e.g. run it in a thread that can be abandoned).
        # ------------------------------------------------------------------
        logger.debug(
            "GigaChatAdapter.close(): skipping aclose() to avoid "
            "Windows SSL shutdown hang. The worker subprocess will be "
            "terminated by the supervisor."
        )
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _upload_image_bytes(
        self,
        image_bytes: bytes,
        mime_hint: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        """
        Upload an image to GigaChat and return its file_id, or None on failure.

        Two responsibilities beyond a bare ``aupload_file`` call:

        1. **Detect the real MIME type.** GigaChat rejects
           ``application/octet-stream`` with a 400. We inspect magic
           bytes (and, as fallback, the data-URI hint) to derive the
           correct type, then pass it to the SDK so the multipart
           Content-Type header is ``image/png`` (etc.) rather than
           the SDK's default.

        2. **Cross-SDK parameter compatibility.** Different
           ``gigachat`` SDK versions expose the MIME hint under
           different kwarg names (``content_type`` on 0.2.x+,
           ``filename`` on some older builds, none at all on
           ancient versions). We introspect the bound method once
           and dispatch to whichever kwarg the installed SDK
           actually accepts.

        Caching: the first successful upload of any given image is
        cached by SHA-256 prefix for the lifetime of the adapter
        instance. Repeated mentions of the same image (in successive
        assistant turns within one chat, or multiple parts of the same
        message) reuse the file_id instead of re-uploading.

        Supported formats (per GigaChat API):
            * image/png
            * image/jpeg
            * image/tiff
            * image/bmp

        WebP and GIF are detected but rejected by the API; we log a
        clear warning and skip so the conversation continues text-only
        rather than failing the whole run.

        Never raises. Any per-image failure is logged and skipped, so
        a single bad image never breaks a multimodal turn.
        """
        import hashlib as _hashlib
        import inspect as _inspect
        from io import BytesIO

        # ----------------------------------------------------------
        # Step 1 — resolve the real MIME type.
        # ----------------------------------------------------------
        mime_type = _detect_image_mime(image_bytes, hint=mime_hint)

        # ----------------------------------------------------------
        # Step 2 — fast-fail on formats GigaChat does not accept.
        #
        # Doing this client-side produces a clean, actionable log line
        # instead of the generic `400 File format ... is not supported`
        # from the API. The user sees exactly which format was rejected
        # and can convert the image before retrying.
        # ----------------------------------------------------------
        _GIGA_SUPPORTED = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}
        if mime_type not in _GIGA_SUPPORTED:
            logger.warning(
                "GigaChatAdapter: image format %r is not supported by "
                "GigaChat (supported: %s). Skipping upload — the "
                "conversation will continue without this image.",
                mime_type,
                sorted(_GIGA_SUPPORTED),
            )
            return None

        # ----------------------------------------------------------
        # Step 3 — cache check.
        # ----------------------------------------------------------
        cache_key = _hashlib.sha256(image_bytes).hexdigest()[:16]
        if cache_key in self._uploaded_image_cache:
            file_id, cached_mime = self._uploaded_image_cache[cache_key]
            logger.debug(
                "GigaChatAdapter: image cache hit (%s -> %s, %s)",
                cache_key,
                file_id,
                cached_mime,
            )
            return (file_id, cached_mime)

        # ----------------------------------------------------------
        # Step 4 — upload via the SDK's 3-tuple form.
        #
        # The SDK's ``aupload_file`` signature (verified against the
        # installed version) is::
        #
        #     aupload_file(
        #         file: Union[
        #             IO[bytes], bytes, str,
        #             Tuple[Optional[str], IO[bytes]],
        #             Tuple[Optional[str], IO[bytes], Optional[str]],
        #             Tuple[Optional[str], IO[bytes], Optional[str],
        #                   Mapping[str, str]],
        #         ],
        #         purpose: Literal['general', 'assistant'] = 'general',
        #     )
        #
        # The 3-tuple form ``(filename, content, content_type)`` is the
        # ONLY way to pass a MIME type explicitly. This matches the
        # httpx multipart contract — GigaChat SDK forwards the tuple
        # verbatim to httpx's ``files=`` parameter.
        #
        # Without the MIME, the SDK sends ``application/octet-stream``
        # (the default for raw bytes) and the server rejects with::
        #
        #   400 File format application/octet-stream is not supported
        #
        # That is the exact failure observed in production before this
        # fix — the octet-stream error appeared even though the caller
        # had a perfectly valid PNG.
        #
        # ``purpose='general'`` is correct for vision: the file is
        # attached per-message (via ``Messages.attachments``) and used
        # for image understanding. ``purpose='assistant'`` is for files
        # that must persist across all turns of a single assistant
        # thread — not our use case.
        # ----------------------------------------------------------
        _EXT_BY_MIME = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/tiff": "tiff",
            "image/bmp": "bmp",
        }
        ext = _EXT_BY_MIME.get(mime_type, "png")
        filename = f"image.{ext}"

        try:
            uploaded = await self._client.aupload_file(
                (filename, BytesIO(image_bytes), mime_type),
                # ``purpose`` selects the file's lifetime scope:
                #
                #   * ``general``   — one-off files for the current
                #                     chat thread. This is the ONLY
                #                     purpose that accepts images
                #                     for vision analysis. GigaChat
                #                     rejects PNG/JPEG uploads under
                #                     ``assistant`` with:
                #                       400 File format image/png is
                #                       not supported
                #                     (verified in production).
                #
                #   * ``assistant`` — persistent files bound to the
                #                     assistant thread across all
                #                     turns. NOT for images.
                #
                # Files uploaded with ``general`` are usable in
                # ``Messages.attachments`` for the current chat
                # session, which is exactly what vision needs.
                purpose="general",
            )
        except Exception as exc:
            # Log the exact tuple we passed so a future SDK change is
            # diagnosable from a single log line.
            logger.warning(
                "GigaChatAdapter: image upload failed "
                "(filename=%s, mime=%s, bytes=%d): %s",
                filename,
                mime_type,
                len(image_bytes),
                exc,
            )
            return None

        # ----------------------------------------------------------
        # Step 6 — extract file_id and cache.
        # ----------------------------------------------------------
        file_id = (
            getattr(uploaded, "id_", None)
            or getattr(uploaded, "id", None)
        )
        if not file_id:
            logger.warning(
                "GigaChatAdapter: upload returned no file_id (got %r)",
                type(uploaded).__name__,
            )
            return None

        file_id_str = str(file_id)
        self._uploaded_image_cache[cache_key] = (file_id_str, mime_type)
        logger.info(
            "GigaChatAdapter: uploaded image (%d bytes, %s) -> file_id=%s",
            len(image_bytes),
            mime_type,
            file_id_str,
        )
        return (file_id_str, mime_type)

    async def _content_to_text_and_attachments(
        self, content: Any
    ) -> tuple[str, List[tuple[str, str]]]:
        """
        Split a message's content into (text, image_file_ids).

        Handles all three shapes ``compiled.messages[].content`` can take:

          * ``str`` — returned as (text, []).
          * ``list`` of OpenAI-style parts:
              - ``{"type": "text", "text": "..."}`` → appended to text.
              - ``{"type": "image_url", "image_url": {"url": "data:..."}}``
                → base64-decoded, uploaded to GigaChat, file_id appended.
              - non-data-URI URLs (``http://``, ``artifact://``) → skipped
                with a warning (SDK ``aupload_file`` only accepts bytes).
          * any other shape → coerced via str().

        Never raises. Any per-part failure is logged and skipped so the
        conversation continues with whatever parts succeeded. This mirrors
        the failure-safety contract of the rest of the adapter.
        """
        import base64 as _b64

        if isinstance(content, str):
            return content, []

        if not isinstance(content, list):
            return str(content or ""), []

        text_fragments: List[str] = []
        attachments: List[tuple[str, str]] = []

        for part in content:
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")

            if ptype == "text":
                txt = part.get("text", "")
                if isinstance(txt, str) and txt:
                    text_fragments.append(txt)

            elif ptype == "image_url":
                image_payload = part.get("image_url")
                if isinstance(image_payload, dict):
                    url = image_payload.get("url")
                else:
                    url = image_payload

                if not isinstance(url, str) or not url:
                    continue

                if not url.startswith("data:"):
                    logger.warning(
                        "GigaChatAdapter: skipping non-data-URI image "
                        "URL %r (GigaChat requires inline bytes).",
                        url[:80],
                    )
                    continue

                # Parse the data URI: "data:<mime>;base64,<payload>"
                # The MIME is optional (some callers emit "data:;base64,..")
                # so we treat it as a hint rather than a requirement.
                try:
                    header, _, data = url.partition(",")
                    # header = "data:image/png;base64"
                    # Extract MIME from "image/png" part
                    mime_hint: Optional[str] = None
                    if header.startswith("data:"):
                        meta = header[len("data:"):]
                        mime_segment = meta.split(";", 1)[0].strip()
                        if mime_segment.startswith("image/"):
                            mime_hint = mime_segment.lower()
                    raw_bytes = _b64.b64decode(data)
                except Exception as exc:
                    logger.warning(
                        "GigaChatAdapter: could not decode data URI: %s",
                        exc,
                    )
                    continue

                uploaded = await self._upload_image_bytes(
                    raw_bytes, mime_hint=mime_hint
                )
                if uploaded:
                    attachments.append(uploaded)

        return " ".join(text_fragments), attachments

    def _has_images(self, compiled: CompiledRequest) -> bool:
        """
        Return True iff any message carries an image_url part.

        Used to choose between the two request-building paths:
        * Text-only → the old ``Chat``/``Messages`` path (works, tools work)
        * Multimodal → the new ``ChatCompletionRequest``/``ChatMessage``
          path (only this path actually passes image pixels to the
          model)
        """
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    async def _build_chat_completion_request(
        self, compiled: CompiledRequest, stream: bool = False
    ) -> ChatCompletionRequest:
        """
        Build a v2 ``ChatCompletionRequest`` for a multimodal turn.

        ``target="vision"`` is set on every ``ChatContentFile`` —
        without it, GigaChat treats the file as a plain attachment and
        the model reports "I cannot see any images attached".
        """
        # Warn (once per call) when the caller sends an image to a
        # model that is NOT documented as vision-capable. This is a
        # silent failure otherwise: GigaChat returns a clean "I cannot
        # see any images" response and the user has no idea that the
        # model itself doesn't support vision.
        _facts = self._model_facts.get(compiled.model, {}) or {}
        if _facts.get("vision") is False:
            logger.warning(
                "GigaChatAdapter: model %r is documented as text-only "
                "(vision=False in model_facts). The image will be "
                "uploaded but the model will not analyze it. Use "
                "GigaChat-2-Max or GigaChat-2-Pro for vision.",
                compiled.model,
            )

        # Multimodal request builder — the ONLY path that actually
        # delivers image content to the model. The old
        # ``Messages.attachments`` interface carries the file_id as
        # metadata but never forwards the image bytes.
        messages: List[ChatMessage] = []

        # Map tool_call_id -> function name (same rationale as the old
        # path — GigaChat's FUNCTION role wants the function name).
        tool_call_id_to_name: Dict[str, str] = {}
        for msg in compiled.messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id")
                        fn = tc.get("function") or {}
                        fn_name = fn.get("name") if isinstance(fn, dict) else None
                        if tc_id and fn_name:
                            tool_call_id_to_name[str(tc_id)] = str(fn_name)

        # System prompt
        if compiled.system_prompt:
            messages.append(
                ChatMessage(
                    role="system",
                    content=[ChatContentPart(text=compiled.system_prompt)],
                )
            )

        for msg in compiled.messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")

            content_text, image_attachments = (
                await self._content_to_text_and_attachments(raw_content)
            )

            # Tool results (from a previous turn that used tools)
            if role == "tool":
                tc_id = msg.get("tool_call_id") or ""
                fn_name = tool_call_id_to_name.get(str(tc_id), "unknown")

                raw_result = str(content_text or "")
                try:
                    json.loads(raw_result)
                    parsed_result: Any = raw_result
                except (ValueError, TypeError):
                    parsed_result = {"result": raw_result}

                messages.append(
                    ChatMessage(
                        role="function",
                        content=[
                            ChatContentPart(
                                function_result=ChatFunctionResult(
                                    name=fn_name,
                                    result=parsed_result,
                                )
                            )
                        ],
                    )
                )
                continue

            # Assistant messages with tool_calls (from a previous turn)
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                first_tc = tool_calls[0] if tool_calls else None
                if first_tc and isinstance(first_tc, dict):
                    fn = first_tc.get("function") or {}
                    fn_name = fn.get("name") if isinstance(fn, dict) else None
                    raw_args = fn.get("arguments") if isinstance(fn, dict) else None
                    if isinstance(raw_args, str):
                        try:
                            args_dict: Any = json.loads(raw_args)
                        except Exception:
                            args_dict = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args_dict = raw_args
                    else:
                        args_dict = {}

                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=[ChatContentPart(text=str(content_text or ""))],
                            function_call=ChatFunctionCall(
                                name=str(fn_name or "unknown"),
                                partial_arguments=args_dict,
                            ),
                        )
                    )
                    continue

            # Regular user / assistant message, possibly with images
            content_parts: List[ChatContentPart] = []
            if content_text:
                content_parts.append(ChatContentPart(text=str(content_text)))
            if image_attachments:
                # ----------------------------------------------------------
                # ``target="vision"`` is REQUIRED for GigaChat to treat
                # the attached file as an image input for vision
                # analysis. Without it, the file is silently treated as
                # a plain attachment (metadata only) and the model
                # reports "I cannot see any images attached".
                #
                # This matches Sber's Chat API v2 spec::
                #
                #   {"files": [{"id": "<file_id>", "target": "vision"}]}
                #
                # Any other value ("document", "audio", ...) would mark
                # the file for a different pipeline and the image
                # analysis would still be skipped.
                # ----------------------------------------------------------
                files_list = [
                    ChatContentFile(id_=fid, mime=mime, target="vision")
                    for fid, mime in image_attachments
                ]
                content_parts.append(ChatContentPart(files=files_list))
            if not content_parts:
                content_parts = [ChatContentPart(text="")]

            messages.append(
                ChatMessage(
                    role=role if role in ("user", "assistant", "system") else "user",
                    content=content_parts,
                )
            )

        # ------------------------------------------------------------------
        # [FIX-MIXED-MODE] GigaChat v2 does NOT support tool calling.
        # When an image is present, tools are silently discarded by the
        # SDK. Without an explicit note, the model hallucinates tool
        # calls or loops trying to invoke tools. Append a directive so
        # it responds with plain text and explains what the user should
        # do next.
        # ------------------------------------------------------------------
        if compiled.tools:
            logger.warning(
                "GigaChatAdapter: tools are ignored in multimodal (v2) mode. "
                "The agent will not be able to call tools this turn."
            )
            sys_warning = (
                "\n\n[SYSTEM NOTE: Tool calling is disabled because an image was "
                "provided. You cannot use tools this turn. Respond with plain "
                "text. If you need to perform a tool action (e.g. edit a file), "
                "explain what needs to be done so the user can send a follow-up "
                "text-only request.]"
            )
            has_sys = False
            for msg in messages:
                if msg.role == "system":
                    if msg.content:
                        msg.content[0].text = (msg.content[0].text or "") + sys_warning
                    has_sys = True
                    break
            if not has_sys:
                messages.insert(
                    0,
                    ChatMessage(
                        role="system",
                        content=[ChatContentPart(text=sys_warning.strip())]
                    )
                )

        # Build ChatModelOptions only with non-None values. GigaChat's
        # Pydantic schema accepts None, but serialized `null` fields may
        # be rejected server-side. Passing an empty ChatModelOptions
        # (or None) tells GigaChat to use its own defaults.
        _model_opts_kwargs: Dict[str, Any] = {}
        _temp = compiled.generation.get("temperature")
        if _temp is not None:
            _model_opts_kwargs["temperature"] = float(_temp)
        _max_tok = compiled.generation.get("max_tokens")
        if _max_tok is not None:
            _model_opts_kwargs["max_tokens"] = int(_max_tok)

        return ChatCompletionRequest(
            model=compiled.model,
            messages=messages,
            model_options=(
                ChatModelOptions(**_model_opts_kwargs)
                if _model_opts_kwargs
                else None
            ),
            stream=stream,
        )

    async def _build_chat_payload(
        self, compiled: CompiledRequest, stream: bool = False
    ) -> Chat:
        messages: List[Messages] = []

        # ------------------------------------------------------------------
        # First pass: build tool_call_id -> function_name map.
        #
        # GigaChat's FUNCTION role requires the function NAME, not the
        # tool_call_id that OpenAI's protocol uses. We need to resolve
        # this mapping before iterating the messages, so that tool_result
        # messages can carry the correct `name` field.
        # ------------------------------------------------------------------
        tool_call_id_to_name: Dict[str, str] = {}
        for msg in compiled.messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id")
                        fn = tc.get("function") or {}
                        fn_name = fn.get("name") if isinstance(fn, dict) else None
                        if tc_id and fn_name:
                            tool_call_id_to_name[str(tc_id)] = str(fn_name)

        if compiled.system_prompt:
            messages.append(
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=compiled.system_prompt,
                )
            )

        for msg in compiled.messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")

            # Multimodal: upload image parts to GigaChat and collect the
            # resulting file_ids. Returns plain text plus a list of
            # attachments (empty when there are no images). Handles all
            # content shapes and never raises.
            content, attachments = await self._content_to_text_and_attachments(
                raw_content
            )

            role_map = {
                "user": MessagesRole.USER,
                "assistant": MessagesRole.ASSISTANT,
                "system": MessagesRole.SYSTEM,
            }

            # --------------------------------------------------------------
            # Tool result messages (role="tool") -> GigaChat's FUNCTION role.
            #
            # GigaChat requires the function NAME on the message that
            # carries the result, not the tool_call_id.
            #
            # CRITICAL: GigaChat also requires the `content` of a FUNCTION
            # message to be VALID JSON. This is a hard divergence from
            # OpenAI, which accepts arbitrary strings (markdown, plain
            # text, etc.). If we send the tool's raw markdown output,
            # GigaChat's server-side json.loads() rejects it with:
            #
            #   422 INVALID_PARAMS: invalid function result for
            #   function <name> json string `<raw>`,
            #   error: JSON parse error at line 1 column 1:
            #   Invalid value.
            #
            # We wrap the tool's plain-text output in a JSON object so
            # that json.loads() always succeeds. If the content is
            # ALREADY valid JSON (rare for tool outputs, but possible
            # for tools that return structured data), we pass it
            # through unchanged to avoid double-wrapping.
            # --------------------------------------------------------------
            if role == "tool":
                tool_call_id = msg.get("tool_call_id") or ""
                fn_name = tool_call_id_to_name.get(
                    str(tool_call_id), "unknown"
                )

                raw_content = str(content or "")
                try:
                    # Accept content that is already valid JSON.
                    json.loads(raw_content)
                    function_content = raw_content
                except (ValueError, TypeError):
                    # Wrap plain-text output in a JSON object.
                    # The key name "result" is arbitrary — GigaChat
                    # only validates that the payload parses as JSON.
                    function_content = json.dumps(
                        {"result": raw_content},
                        ensure_ascii=False,
                    )

                messages.append(
                    Messages(
                        role=MessagesRole.FUNCTION,
                        content=function_content,
                        name=fn_name,
                    )
                )
                continue

            # --------------------------------------------------------------
            # Assistant messages that requested a tool call -> GigaChat's
            # function_call field. GigaChat accepts only ONE function_call
            # per assistant message (unlike OpenAI, which uses a
            # tool_calls array). We take the first call as a documented
            # limitation.
            # --------------------------------------------------------------
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                first_tc = tool_calls[0] if tool_calls else None
                if first_tc and isinstance(first_tc, dict):
                    fn = first_tc.get("function") or {}
                    fn_name = fn.get("name") if isinstance(fn, dict) else None
                    raw_args = (
                        fn.get("arguments") if isinstance(fn, dict) else None
                    )
                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except Exception:
                            args_dict = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args_dict = raw_args
                    else:
                        args_dict = {}

                    # ------------------------------------------------------------------
                    # GigaChat SDK requires `content: str` — it does NOT accept
                    # `None`. The previous code passed `None` when the assistant
                    # message contained only a function_call (no text), which
                    # is the NORMAL case after a tool is invoked. On the next
                    # step (when we re-send the conversation back to GigaChat
                    # with the tool result), the Pydantic validator rejected
                    # the message with:
                    #
                    #   ValidationError: content - Input should be a valid string
                    #   [input_value=None, input_type=NoneType]
                    #
                    # Passing an empty string is valid in GigaChat's protocol:
                    # the model understands "assistant requested a function
                    # call, produced no text" from `function_call` alone.
                    # ------------------------------------------------------------------
                    messages.append(
                        Messages(
                            role=MessagesRole.ASSISTANT,
                            content=str(content or ""),
                            function_call=FunctionCall(
                                name=str(fn_name or "unknown"),
                                arguments=args_dict,
                            ),
                        )
                    )
                    continue

            # Regular message. Attach image file_ids when the message
            # carried images. GigaChat's ``Messages.attachments`` accepts
            # a list of file_ids; when empty, omit the field entirely
            # (rather than passing []) to match the SDK's expected wire
            # format for text-only messages.
            messages.append(
                Messages(
                    role=role_map.get(role, MessagesRole.USER),
                    content=str(content or ""),
                    attachments=attachments if attachments else None,
                )
            )

        # ------------------------------------------------------------------
        # Convert compiled.tools (OpenAI shape) -> GigaChat Functions.
        #
        #   OpenAI: {"type": "function", "function": {name, description, parameters}}
        #   GigaChat: Function(name, description, parameters)
        # ------------------------------------------------------------------
        functions: Optional[List[Function]] = None
        if compiled.tools:
            functions = []
            for tool in compiled.tools:
                fn = tool.get("function") if "function" in tool else tool
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if not name:
                    continue

                # Sanitize the tool's JSON Schema parameters. GigaChat
                # rejects any `format` value outside {date, date-time,
                # time} with a 422 — see _sanitize_parameters_for_gigachat
                # above for the full rationale. This must run BEFORE
                # constructing the Function object, because the SDK
                # validates parameters at serialization time.
                raw_params = fn.get("parameters") or {}
                sanitized_params = _sanitize_parameters_for_gigachat(raw_params)

                functions.append(
                    Function(
                        name=str(name),
                        description=str(fn.get("description") or ""),
                        parameters=sanitized_params,
                    )
                )
            if not functions:
                functions = None

        # ------------------------------------------------------------------
        # Convert compiled.tool_choice -> GigaChat function_call.
        #
        # GigaChat function_call accepts: "auto", "none", or {"name": "..."}.
        # GigaChat does NOT support OpenAI's "required"; we map it to "auto".
        # ------------------------------------------------------------------
        function_call: Optional[Any] = None
        if functions and compiled.tool_choice:
            choice_type = compiled.tool_choice.get("type")
            if choice_type == "auto":
                function_call = "auto"
            elif choice_type == "none":
                function_call = "none"
            elif choice_type == "function":
                fn_data = compiled.tool_choice.get("function") or {}
                fn_name = fn_data.get("name")
                if fn_name:
                    function_call = {"name": str(fn_name)}
            elif choice_type == "required":
                function_call = "auto"

        return Chat(
            messages=messages,
            model=compiled.model,
            temperature=compiled.generation.get("temperature", 0.7),
            max_tokens=compiled.generation.get("max_tokens"),
            functions=functions,
            function_call=function_call,
            # stream=True must be set at CONSTRUCTION time. Some gigachat
            # SDK versions ignore the runtime-set attribute when
            # serializing the body, causing buffered (non-streamed)
            # responses.
            stream=stream,
        )

    def _convert_response(
        self, result: Any, model: str
    ) -> LLMResponse:
        """
        Convert either a v1 ``ChatCompletion`` or a v2
        ``ChatCompletionResponse`` into ``LLMResponse``.

        The two shapes differ:
            * v1:  ``result.choices[0].message`` (with ``content: str``)
            * v2:  ``result.messages[0]`` (with
                    ``content: List[ChatContentPart]``)
        """
        usage = getattr(result, "usage", None)
        message = None
        finish_reason_raw = None

        # --- v1 shape ---
        choices = getattr(result, "choices", None)
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            finish_reason_raw = getattr(choice, "finish_reason", None)

        # --- v2 shape (fallback if no v1 ``choices``) ---
        if message is None:
            v2_messages = getattr(result, "messages", None)
            if v2_messages:
                message = v2_messages[0]
                finish_reason_raw = getattr(message, "finish_reason", None)

        # Neither shape matched — the server returned an unexpected
        # payload. Return an empty response rather than crash.
        if message is None:
            logger.warning(
                "GigaChatAdapter: response has neither v1 `choices` "
                "nor v2 `messages` (got %r).",
                type(result).__name__,
            )
            return LLMResponse(
                content="",
                tool_calls=[],
                finish_reason="stop",
                model=model,
                raw=result,
                usage=TokenUsage(
                    input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                    total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                    source=UsageSource.PROVIDER if usage else UsageSource.UNKNOWN,
                ),
            )

        # ------------------------------------------------------------------
        # Content extraction — supports BOTH response shapes:
        #
        #   * v1 ChatCompletion  → choice.message is ``Messages``,
        #                          content is a plain ``str``.
        #   * v2 ChatCompletionResponse → choice.message is ``ChatMessage``,
        #                          content is ``List[ChatContentPart]``.
        #
        # We normalize to a single string that the caller sees. The raw
        # response is preserved in ``raw`` for downstream diagnostics.
        # ------------------------------------------------------------------
        content = ""
        if message is not None:
            raw_content = getattr(message, "content", None)
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                fragments: List[str] = []
                for part in raw_content:
                    if isinstance(part, dict):
                        txt = part.get("text")
                    else:
                        txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt:
                        fragments.append(txt)
                content = "".join(fragments)
            elif raw_content is not None:
                content = str(raw_content)

        # ------------------------------------------------------------------
        # Extract function_call from the message. GigaChat returns
        # `arguments` already parsed as a dict (unlike OpenAI, which
        # returns a JSON string). Handle both v1 and v2 shapes:
        #
        #   v1 Messages.function_call      → .name, .arguments
        #   v2 ChatMessage.function_call   → .name, .partial_arguments
        # ------------------------------------------------------------------
        tool_calls: List[ToolCall] = []
        fn_call = (
            getattr(message, "function_call", None)
            if message is not None
            else None
        )
        if fn_call is not None:
            fn_name = getattr(fn_call, "name", None)
            fn_args = (
                getattr(fn_call, "arguments", None)
                or getattr(fn_call, "partial_arguments", None)
            )
            if fn_name:
                if isinstance(fn_args, dict):
                    args_dict = fn_args
                elif isinstance(fn_args, str):
                    try:
                        args_dict = json.loads(fn_args)
                    except Exception:
                        args_dict = {"raw": fn_args}
                else:
                    args_dict = {}

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=str(fn_name),
                        arguments=args_dict,
                        type="function",
                    )
                )

        # Map GigaChat's "function_call" finish_reason to OpenAI's
        # "tool_calls" so downstream code treats both uniformly.
        finish_reason = finish_reason_raw or "stop"
        if finish_reason == "function_call":
            finish_reason = "tool_calls"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                source=UsageSource.PROVIDER if usage else UsageSource.UNKNOWN,
            ),
            model=model,
            raw=result,
        )

    def _map_exception(self, exc: Exception) -> ProviderError:
        status = getattr(exc, "status_code", None)
        if status == 401:
            return ProviderAuthError(str(exc), provider="gigachat", status_code=401)
        if status == 429:
            return ProviderRateLimitError(str(exc), provider="gigachat", status_code=429)
        if status == 400:
            return ProviderInvalidRequestError(str(exc), provider="gigachat", status_code=400)
        if status in (500, 502, 503, 504):
            return ProviderUnavailableError(str(exc), provider="gigachat", status_code=status)
        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError(str(exc), provider="gigachat", phase="request")
        if isinstance(exc, ConnectionError):
            return ProviderConnectionError(str(exc), provider="gigachat")
        return ProviderError(str(exc), provider="gigachat")
