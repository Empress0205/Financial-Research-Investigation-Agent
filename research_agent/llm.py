"""Thin wrapper around the Groq chat API.

The rest of the code never touches the Groq SDK directly. It only sees
`LLMResponse` and `ToolCall`. That gives us two things:
  1. Tests can swap in a fake LLM with the same `chat()` method.
  2. If we change provider later, only this file changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """Raised when the LLM call fails (network, auth, rate limit, bad request).

    `kind` lets the agent react differently to different failures:
      "too_large"    the request has too many tokens -> shrink the context and retry
      "rate_limited" too many requests -> the SDK already waited and retried; give up
      "other"        anything else
    """

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str         # the API links our tool result back to this request by id
    name: str       # which tool the model wants
    arguments: str  # raw JSON string, exactly as the model wrote it (may be malformed!)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0

    def to_assistant_message(self) -> dict[str, Any]:
        """Convert back to the message format the API expects in the history."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    """Anything with this method can drive the agent (real Groq or a test fake)."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_retries: int = 4) -> None:
        import groq  # imported here so tests never need the SDK configured

        self._groq = groq
        # The SDK retries rate-limit (429) and server errors itself, with backoff.
        # The free tier has a small tokens-per-minute limit, so we allow a few more retries.
        self._client = groq.Groq(api_key=api_key, max_retries=max_retries)
        self.model = model
        self.temperature = temperature

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice  # "auto" = model decides, "none" = must answer in text

        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._groq.APIError as exc:  # base class of every Groq API error
            status = getattr(exc, "status_code", None)
            kind = {413: "too_large", 429: "rate_limited"}.get(status, "other")
            raise LLMError(f"{type(exc).__name__}: {exc}", kind=kind) from exc

        message = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (message.tool_calls or [])
        ]
        total_tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(content=message.content, tool_calls=tool_calls, total_tokens=total_tokens)
