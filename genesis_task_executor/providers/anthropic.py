"""Anthropic provider — wraps AsyncAnthropic for the LLMProvider protocol."""

from __future__ import annotations

import json
import os

from genesis_task_executor.types import ToolCall, ToolResponse


def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool definitions to Anthropic format."""
    converted = []
    for tool in tools:
        func = tool.get("function", tool)
        converted.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _convert_messages_to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-style messages to Anthropic format.

    Returns (system_prompt, messages).
    """
    system_prompt = None
    converted = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            system_prompt = msg.get("content", "")
            continue

        if role == "tool":
            # Anthropic uses tool_result content blocks
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # Convert tool_calls to Anthropic tool_use blocks
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                func = tc.get("function", tc)
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })
            converted.append({"role": "assistant", "content": content_blocks})
            continue

        converted.append({"role": role, "content": msg.get("content", "")})

    return system_prompt, converted


class AnthropicProvider:
    """LLM provider using the Anthropic API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install genesis-task-executor[anthropic]"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._client = AsyncAnthropic(api_key=key)
        self._model = model or os.environ.get(
            "TASK_EXECUTOR_MODEL", "claude-sonnet-4-20250514",
        )

    async def complete(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
    ) -> str:
        """Return text completion."""
        system, conv_messages = _convert_messages_to_anthropic(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": conv_messages,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        """Return completion with optional tool calls."""
        system, conv_messages = _convert_messages_to_anthropic(messages)
        anthropic_tools = _convert_tools_to_anthropic(tools)

        kwargs: dict = {
            "model": self._model,
            "messages": conv_messages,
            "tools": anthropic_tools,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        content = "\n".join(text_parts) if text_parts else None

        return ToolResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
        )
