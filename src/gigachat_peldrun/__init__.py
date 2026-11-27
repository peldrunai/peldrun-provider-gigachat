# package/gigachat_peldrun/__init__.py
"""
GigaChat provider adapter for PELDRUN.

This package implements the ProviderAdapter contract for the GigaChat
(Sber) LLM service. It translates PELDRUN's LLMRequest into GigaChat's
wire protocol and parses the responses back into LLMResponse objects.

Entry point is declared in manifest.json as:
    gigachat_peldrun.adapter:GigaChatAdapter

The adapter runs inside an isolated worker subprocess managed by
peldrun.llm.plugin_host. It communicates with the supervisor over a
JSON-Lines IPC protocol on stdin/stdout.

See https://developers.sber.ru/docs/ru/gigachat/guides/main for the
upstream GigaChat API documentation.
"""

__version__ = "1.0.0"
__author__ = "PELDRUN Community"

# The adapter class is exposed here for convenience, but PELDRUN imports
# it via the fully-qualified entry_point path. Importing it here is not
# required and would pull in SDK dependencies at package-import time,
# which we intentionally avoid to keep `import gigachat_peldrun` cheap.
#
# from .adapter import GigaChatAdapter  # noqa: F401

__all__ = ["__version__", "__author__"]