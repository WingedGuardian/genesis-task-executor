"""OpenAI provider — wraps AsyncOpenAI for the LLMProvider protocol."""

from __future__ import annotations

import os

from genesis_task_executor.types import ToolCall, ToolResponse


class OpenAIProvider:
    """LLM provider using the OpenAI API (or any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install genesis-task-executor[openai]"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")

        kwargs: dict = {"api_key": key}
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.environ["OPENAI_BASE_URL"]

        self._client = AsyncOpenAI(**kwargs)
        self._model = model or os.environ.get("TASK_EXECUTOR_MODEL", "gpt-4o-mini")

    async def complete(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
    ) -> str:
        """Return text completion."""
        kwargs: dict = {"model": self._model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        """Return completion with optional tool calls."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                # Handle models that wrap args in array
                if isinstance(args, list):
                    args = args[0] if args else {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return ToolResponse(
            content=msg.content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason or "",
        )
