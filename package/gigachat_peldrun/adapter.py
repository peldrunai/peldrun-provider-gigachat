# package/gigachat_peldrun/adapter.py
"""
GigaChat adapter for PELDRUN.

Translates PELDRUN's LLMRequest into GigaChat's wire protocol and back.
Runs inside an isolated worker subprocess managed by peldrun.llm.plugin_host.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from gigachat import GigaChat
from gigachat.exceptions import AuthenticationError as GigaAuthError
from gigachat.exceptions import GigaChatException
from gigachat.models import Chat, ChatCompletion, Messages, MessagesRole

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
        scope = str(extra.get("scope") or "PERS")
        ca_bundle_path = extra.get("ca_bundle_path") or None
        verify_ssl = bool(extra.get("verify_ssl", True))

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

        # GigaChat SDK handles OAuth token acquisition internally.
        self._client = GigaChat(
            credentials=credentials,
            scope=scope,
            verify_ssl_certs=verify_ssl,
            ca_bundle_file=ca_bundle_path,
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
            chat = self._build_chat_payload(compiled)
            chat.stream = True

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

                # Finish reason terminates the stream.
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
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
            results.append({
                "id": str(model_id),
                "display_name": getattr(model, "name", None),
                "max_context_length": getattr(model, "context_length", None),
                "type": "llm",
            })
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

    def _build_chat_payload(self, compiled: CompiledRequest) -> Chat:
        messages: List[Messages] = []

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
            messages.append(
                Messages(
                    role=role_map.get(role, MessagesRole.USER),
                    content=str(content or ""),
                )
            )

        return Chat(
            messages=messages,
            model=compiled.model,
            temperature=compiled.generation.get("temperature", 0.7),
            max_tokens=compiled.generation.get("max_tokens"),
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

        return LLMResponse(
            content=content,
            tool_calls=[],
            finish_reason=choice.finish_reason or "stop",
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