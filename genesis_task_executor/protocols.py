"""Protocol interfaces for pluggable backends.

All protocols have NoOp default implementations that gracefully degrade.
Only ``LLMProvider`` is required — everything else is optional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from genesis_task_executor.types import ToolResponse

# ---------------------------------------------------------------------------
# LLM Provider (required)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Async LLM provider for completions and tool use."""

    async def complete(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
    ) -> str:
        """Return text completion."""
        ...

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        """Return completion that may include tool calls."""
        ...


# ---------------------------------------------------------------------------
# Optional adapters (NoOp defaults)
# ---------------------------------------------------------------------------


@runtime_checkable
class WebSearcher(Protocol):
    """Web search for research and due diligence."""

    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        """Return [{title, url, snippet}]."""
        ...


@runtime_checkable
class KnowledgeStore(Protocol):
    """Knowledge recall for due diligence."""

    async def recall(self, query: str, *, limit: int = 5) -> list[dict]:
        """Return [{content, source, score}]."""
        ...


@runtime_checkable
class ProcedureStore(Protocol):
    """Procedural memory for workaround search."""

    async def find_relevant(
        self, context_tags: list[str], *, limit: int = 3,
    ) -> list[dict] | None:
        """Return [{approach, confidence}] or None if no matches."""
        ...


@runtime_checkable
class EventCallback(Protocol):
    """Optional event callback for observability."""

    def on_event(self, event_type: str, data: dict) -> None:
        """Called on lifecycle events (transition, step_complete, etc.)."""
        ...


# ---------------------------------------------------------------------------
# NoOp implementations
# ---------------------------------------------------------------------------


class NoOpWebSearcher:
    """Returns empty results — recovery degrades gracefully."""

    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        return []


class NoOpKnowledgeStore:
    """Returns empty results — due diligence skipped."""

    async def recall(self, query: str, *, limit: int = 5) -> list[dict]:
        return []


class NoOpProcedureStore:
    """Returns None — workaround layer skipped."""

    async def find_relevant(
        self, context_tags: list[str], *, limit: int = 3,
    ) -> list[dict] | None:
        return None


class NoOpEventCallback:
    """Silently discards events."""

    def on_event(self, event_type: str, data: dict) -> None:
        pass
