"""
Pluggable LLM backend so the same agent code can run:
  - fully offline with a deterministic template (default, zero API keys needed)
  - against Anthropic's API (if ANTHROPIC_API_KEY is set)
  - against OpenAI's API (if OPENAI_API_KEY is set)

This mirrors a common real-world agent-engineering pattern: never hard-code
a single model vendor into your agent logic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class TemplateProvider(LLMProvider):
    """No external calls, no API key. Deterministic and always available.

    This is intentionally simple: it composes the final summary from the
    structured facts it's given rather than free-generating prose, which
    also sidesteps hallucination risk for a project with no real
    safety-eval pipeline behind it.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "[template-mode summary]\n"
            + user_prompt
            + "\n\n(Set ANTHROPIC_API_KEY or OPENAI_API_KEY to generate "
            "richer natural-language prose via a real LLM.)"
        )


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # local import: optional dependency

        self._client = OpenAI()
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""


def get_llm_provider() -> LLMProvider:
    """Selects a provider based on environment variables, with safe fallback."""

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except Exception:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except Exception:
            pass
    return TemplateProvider()
