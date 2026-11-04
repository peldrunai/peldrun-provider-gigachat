# package/gigachat_peldrun/adapter.py
"""
GigaChat adapter for PELDRUN.

Translates PELDRUN's LLMRequest into GigaChat's wire protocol and back.
Runs inside an isolated worker subprocess managed by peldrun.llm.plugin_host.
"""

from __future__ import annotations
import json
import uuid
import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from gigachat import GigaChat
from gigachat.exceptions import AuthenticationError as GigaAuthError
from gigachat.exceptions import GigaChatException
from gigachat.models import (
    Chat,
    ChatCompletion,
    Function,
    FunctionCall,
    Messages,
    MessagesRole,
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

        # Extract custom connection fields from extra_params.
        # `_peldrun_plugin` is reserved for the descriptor - never read it.
        extra = dict(self._connection.extra_params or {})
        extra.pop("_peldrun_plugin", None)

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

        # When verify_ssl is False, the GigaChat SDK's OAuth code path
        # ignores the `verify_ssl_certs=False` flag: it creates its own
        # httpx.AsyncClient() for the OAuth POST without passing
        # `verify=False`, so verification falls back to the system cert
        # store, which does not include the Russian Trusted Root CA —
        # producing `[SSL: CERTIFICATE_VERIFY_FAILED]` on
        # `ngw.devices.sberbank.ru:9443`.
        #
        # To bypass verification reliably for both the OAuth call and
        # the main API client, we monkey-patch httpx.AsyncClient.__init__
        # BEFORE creating the GigaChat instance. Every httpx client
        # created afterward — including the SDK's internal OAuth client
        # and its main chat client — will use verify=False.
        #
        # This is safe in a worker subprocess because:
        #   * The worker only runs THIS adapter (isolated venv).
        #   * No other code in the process needs real TLS verification.
        #   * The patch is applied per-worker, not globally to the
        #     main PELDRUN process.
        # When verify_ssl is False, the GigaChat SDK still verifies TLS
        # because it uses httpx.Client (SYNC) for OAuth, and in some
        # versions creates pre-configured httpx.AsyncHTTPTransport
        # instances whose SSL context overrides verify=. Patching only
        # AsyncClient is not enough.
        #
        # We patch ALL FOUR httpx entry points:
        #   * httpx.AsyncClient.__init__         (async client)
        #   * httpx.Client.__init__              (sync client — used by
        #                                         GigaChat SDK for OAuth!)
        #   * httpx.AsyncHTTPTransport.__init__  (async transport)
        #   * httpx.HTTPTransport.__init__       (sync transport)
        #
        # Each patch is idempotent (guarded by a marker attribute) so a
        # re-instantiation in the same worker does not stack patches.
        if not verify_ssl:
            import httpx as _httpx

            def _patch_httpx_class(cls, label):
                original = cls.__init__
                if getattr(original, "_peldrun_noverify_patched", False):
                    return
                is_transport = "Transport" in cls.__name__

                def _patched_init(self, *args, **kwargs):
                    if is_transport and args:
                        # Transport classes accept ``verify`` as the
                        # first positional arg. Override it in place
                        # to avoid "multiple values for keyword
                        # argument 'verify'".
                        args = (False,) + args[1:]
                        kwargs.pop("verify", None)
                    else:
                        kwargs["verify"] = False
                    original(self, *args, **kwargs)

                _patched_init._peldrun_noverify_patched = True  # type: ignore[attr-defined]
                cls.__init__ = _patched_init  # type: ignore[method-assign]
                logger.warning(
                    "GigaChatAdapter: verify_ssl=False — patched %s "
                    "to force verify=False.",
                    label,
                )

            _patch_httpx_class(_httpx.AsyncClient, "httpx.AsyncClient")
            _patch_httpx_class(_httpx.Client, "httpx.Client")
            _patch_httpx_class(_httpx.AsyncHTTPTransport, "httpx.AsyncHTTPTransport")
            _patch_httpx_class(_httpx.HTTPTransport, "httpx.HTTPTransport")

        # GigaChat SDK handles OAuth token acquisition internally.
        #
        # Important: when `verify_ssl` is False, do NOT pass
        # `ca_bundle_file` — the SDK uses
        # `verify = verify_ssl_certs or ca_bundle_file`, so a non-None
        # `ca_bundle_file` OVERRIDES the explicit `verify_ssl_certs=False`
        # and forces certificate verification against that bundle. If the
        # bundle is incomplete (e.g., the Russian Root CA without the
        # intermediate CA), the call fails with
        # "certificate verify failed: self-signed certificate in
        # certificate chain".
        #
        # When `verify_ssl` is True, we pass the CA bundle so the SDK
        # uses it for verification instead of certifi.
        if verify_ssl:
            self._client = GigaChat(
                credentials=credentials,
                scope=scope,
                verify_ssl_certs=True,
                ca_bundle_file=ca_bundle_path,
                model=str(extra.get("model") or "GigaChat"),
                timeout=float(self._connection.timeout or 60.0),
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

        logger.debug(
            "GigaChatAdapter ready: scope=%s verify_ssl=%s ca_bundle=%s",
            scope,
            verify_ssl,
            ca_bundle_path or "<system>",
        )

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        compiled = compile_request(request, self._profile)
        try:
            chat = self._build_chat_payload(compiled)
            result: ChatCompletion = await self._client.achat(chat)
        except GigaAuthError as exc:
            raise ProviderAuthError(str(exc), provider="gigachat") from exc
        except GigaChatException as exc:
            raise self._map_exception(exc) from exc
        except Exception as exc:
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

        try:
            # stream=True must be set at CONSTRUCTION time, not after.
            # Older gigachat SDK versions ignored the runtime-set
            # attribute during body serialization, sending a buffered
            # (non-streamed) request to the server.
            chat = self._build_chat_payload(compiled, stream=True)

            async for chunk in self._client.astream(chat):
                # Capture usage first (some providers send usage-only chunks).
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    captured_usage = TokenUsage(
                        input_tokens=getattr(chunk_usage, "prompt_tokens", None),
                        output_tokens=getattr(chunk_usage, "completion_tokens", None),
                        total_tokens=getattr(chunk_usage, "total_tokens", None),
                        source=UsageSource.PROVIDER,
                        estimated=False,
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Content delta
                if delta.content:
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.CONTENT_DELTA,
                        content_delta=delta.content,
                        sequence=seq,
                    )

                # ------------------------------------------------------
                # Function call in stream. GigaChat typically emits the
                # entire function_call in a SINGLE delta chunk (unlike
                # OpenAI which streams name + arguments incrementally).
                # Surface it as TOOL_CALL_START + TOOL_CALL_COMPLETE.
                # ------------------------------------------------------
                delta_fc = getattr(delta, "function_call", None)
                if delta_fc is not None:
                    fn_name = getattr(delta_fc, "name", None)
                    fn_args = getattr(delta_fc, "arguments", None)
                    if fn_name:
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        seq += 1
                        yield LLMStreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call_id=tool_id,
                            tool_name=str(fn_name),
                            sequence=seq,
                        )

                        if isinstance(fn_args, dict):
                            args_dict = fn_args
                        elif isinstance(fn_args, str):
                            try:
                                args_dict = json.loads(fn_args)
                            except Exception:
                                args_dict = {"raw": fn_args}
                        else:
                            args_dict = {}

                        seq += 1
                        yield LLMStreamEvent(
                            type=StreamEventType.TOOL_CALL_COMPLETE,
                            tool_call_id=tool_id,
                            tool_name=str(fn_name),
                            tool_arguments_complete=args_dict,
                            sequence=seq,
                        )

                # Finish reason terminates the stream.
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    # Map GigaChat's "function_call" to OpenAI convention.
                    if finish_reason == "function_call":
                        finish_reason = "tool_calls"
                    seq += 1
                    yield LLMStreamEvent(
                        type=StreamEventType.MESSAGE_COMPLETE,
                        finish_reason=finish_reason,
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

    # ------------------------------------------------------------------
    # discover_models()
    # ------------------------------------------------------------------

    async def discover_models(
        self, connection: Optional[ConnectionContext] = None
    ) -> List[Dict[str, Any]]:
        try:
            models = await self._client.aget_models()
        except Exception as exc:
            logger.warning("GigaChat model discovery failed: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for model in getattr(models, "data", None) or []:
            model_id = (
                getattr(model, "id_", None)
                or getattr(model, "id", None)
                or ""
            )
            if not model_id:
                continue

            # ------------------------------------------------------------------
            # Declare capabilities from the package's own feature flags.
            #
            # The GigaChat API's `/models` endpoint does NOT expose a
            # `capabilities` field — it only returns id, name, and
            # context_length. Every capability fact this adapter can
            # provide therefore comes from `manifest.json::features`.
            #
            # Two options exist for surfacing those facts:
            #
            #   (a) Hardcode them here based on Sber's public docs.
            #   (b) Read them from PROFILE.features, which the plugin
            #       host populates from manifest.features at construction
            #       time.
            #
            # Option (b) is chosen because it keeps a single source of
            # truth (manifest.json) and lets the package vendor ship a
            # new manifest version to add a feature without touching
            # Python code.
            #
            # Before this fix, only `trained_for_tool_use: True` was
            # hardcoded here. Vision was silently dropped even though
            # manifest.features.vision is true, and Reasoning was left
            # ambiguous instead of explicitly false. Result: the UI
            # showed no Vision badge for vision-capable GigaChat models
            # (e.g. GigaChat-2-Max with image input support).
            # ------------------------------------------------------------------
            feats = getattr(self._profile, "features", None)
            vision_supported = bool(getattr(feats, "vision", False))
            tools_supported = bool(getattr(feats, "tool_calling", False))
            reasoning_supported = bool(getattr(feats, "reasoning", False))

            capabilities: Dict[str, bool] = {
                "vision": vision_supported,
                "trained_for_tool_use": tools_supported,
                "reasoning": reasoning_supported,
            }

            # Every capability fact above comes from the manifest, not
            # from a live query. Record the correct provenance so the
            # resolver and the UI can distinguish "declared by the
            # package author" from "reported by the running server".
            capability_sources: Dict[str, str] = {
                "vision": "provider_catalog",
                "trained_for_tool_use": "provider_catalog",
                "reasoning": "provider_catalog",
            }

            # Reasoning, when supported, uses GigaChat's on/off toggle
            # (`reasoning_effort` is not exposed; the SDK treats
            # reasoning as a binary). Only emit details when the
            # capability is actually declared, so the resolver does
            # not fabricate a control mode for a model that has no
            # reasoning at all.
            entry: Dict[str, Any] = {
                "id": str(model_id),
                "display_name": getattr(model, "name", None),
                "max_context_length": getattr(model, "context_length", None),
                "type": "llm",
                "capabilities": capabilities,
                "capability_sources": capability_sources,
            }
            if reasoning_supported:
                entry["capability_details"] = {
                    "reasoning": {
                        "mode": "toggle",
                        "state": "supported",
                    }
                }

            results.append(entry)
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
            return {
                "ok": False,
                "message": str(exc),
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
        try:
            await self._client.aclose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chat_payload(
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
            content = msg.get("content", "")

            # Flatten multimodal content to text (SDK v0.2 limitation).
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
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

            # Regular message.
            messages.append(
                Messages(
                    role=role_map.get(role, MessagesRole.USER),
                    content=str(content or ""),
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
                functions.append(
                    Function(
                        name=str(name),
                        description=str(fn.get("description") or ""),
                        parameters=fn.get("parameters") or {},
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
        self, result: ChatCompletion, model: str
    ) -> LLMResponse:
        if not result.choices:
            return LLMResponse(
                content="",
                tool_calls=[],
                finish_reason="stop",
                model=model,
                raw=result,
            )

        choice = result.choices[0]
        content = choice.message.content or ""
        usage = getattr(result, "usage", None)

        # ------------------------------------------------------------------
        # Extract function_call from the message. GigaChat returns
        # `arguments` already parsed as a dict (unlike OpenAI, which
        # returns a JSON string). Handle both defensively.
        # ------------------------------------------------------------------
        tool_calls: List[ToolCall] = []
        fn_call = getattr(choice.message, "function_call", None)
        if fn_call is not None:
            fn_name = getattr(fn_call, "name", None)
            fn_args = getattr(fn_call, "arguments", None)
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
        finish_reason = choice.finish_reason or "stop"
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
