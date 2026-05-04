"""Shared fixtures for genesis-task-executor test suite."""

from __future__ import annotations

import json
from collections import deque

import aiosqlite
import pytest

from genesis_task_executor.db import init_schema
from genesis_task_executor.types import ToolResponse

# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------


class MockLLMProvider:
    """LLM provider that returns configurable canned responses.

    Usage::

        provider = MockLLMProvider()
        provider.enqueue("first response")
        provider.enqueue("second response")
        result = await provider.complete([...])  # → "first response"
        result = await provider.complete([...])  # → "second response"
        result = await provider.complete([...])  # → "default response"

    For tool use::

        provider.enqueue_tool_response(ToolResponse(content="done"))
    """

    def __init__(self, default_response: str = "default response") -> None:
        self._default = default_response
        self._responses: deque[str] = deque()
        self._tool_responses: deque[ToolResponse] = deque()
        self.complete_calls: list[dict] = []
        self.tool_calls_log: list[dict] = []

    def enqueue(self, response: str) -> None:
        """Queue a response for the next complete() call."""
        self._responses.append(response)

    def enqueue_many(self, responses: list[str]) -> None:
        """Queue multiple responses."""
        self._responses.extend(responses)

    def enqueue_tool_response(self, response: ToolResponse) -> None:
        """Queue a ToolResponse for the next complete_with_tools() call."""
        self._tool_responses.append(response)

    def enqueue_many_tool_responses(self, responses: list[ToolResponse]) -> None:
        """Queue multiple ToolResponse objects."""
        self._tool_responses.extend(responses)

    async def complete(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
    ) -> str:
        self.complete_calls.append({
            "messages": messages,
            "json_mode": json_mode,
        })
        if self._responses:
            return self._responses.popleft()
        return self._default

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        self.tool_calls_log.append({
            "messages": messages,
            "tools": tools,
        })
        if self._tool_responses:
            return self._tool_responses.popleft()
        return ToolResponse(content=self._default)


# ---------------------------------------------------------------------------
# Mock supporting stores
# ---------------------------------------------------------------------------


class MockWebSearcher:
    """WebSearcher that returns configurable results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []
        self.queries: list[str] = []

    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        self.queries.append(query)
        return self._results[:max_results]


class MockKnowledgeStore:
    """KnowledgeStore that returns configurable results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []
        self.queries: list[str] = []

    async def recall(self, query: str, *, limit: int = 5) -> list[dict]:
        self.queries.append(query)
        return self._results[:limit]


class MockProcedureStore:
    """ProcedureStore that returns configurable results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results
        self.queries: list[list[str]] = []

    async def find_relevant(
        self, context_tags: list[str], *, limit: int = 3,
    ) -> list[dict] | None:
        self.queries.append(context_tags)
        return self._results


class MockEventCallback:
    """EventCallback that records all events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Create a MockLLMProvider with default response."""
    return MockLLMProvider()


@pytest.fixture
async def temp_db():
    """Create an in-memory SQLite database with schema initialized."""
    database = await aiosqlite.connect(":memory:")
    database.row_factory = aiosqlite.Row
    await init_schema(database)
    yield database
    await database.close()


@pytest.fixture
def mock_web_searcher() -> MockWebSearcher:
    return MockWebSearcher()


@pytest.fixture
def mock_knowledge_store() -> MockKnowledgeStore:
    return MockKnowledgeStore()


@pytest.fixture
def mock_procedure_store() -> MockProcedureStore:
    return MockProcedureStore()


@pytest.fixture
def mock_event_callback() -> MockEventCallback:
    return MockEventCallback()


@pytest.fixture
def sample_plan() -> dict:
    """A minimal valid plan dict."""
    return {
        "goal": "Test task goal",
        "steps": [
            {
                "idx": 0,
                "type": "analysis",
                "description": "Analyze the input",
                "success_criterion": "Input analyzed and summarized",
            },
        ],
    }


@pytest.fixture
def sample_plan_json(sample_plan) -> str:
    return json.dumps(sample_plan)


@pytest.fixture
def sample_step() -> dict:
    """A single step dict for testing."""
    return {
        "idx": 0,
        "type": "analysis",
        "description": "Analyze the input data",
        "success_criterion": "Data analyzed and summarized",
    }


@pytest.fixture
def sample_step_result():
    """A completed StepResult for testing."""
    from genesis_task_executor.types import StepResult
    return StepResult(
        idx=0,
        status="completed",
        result="Analysis complete: found 3 key patterns",
        cost_usd=0.01,
        duration_s=2.5,
        artifacts=["/tmp/analysis.md"],
    )


@pytest.fixture
def sample_failed_step_result():
    """A failed StepResult for testing."""
    from genesis_task_executor.types import StepResult
    return StepResult(
        idx=0,
        status="failed",
        result="ERROR: could not connect to API endpoint",
        cost_usd=0.005,
        duration_s=1.0,
    )


def make_plan_review_approved(confidence: float = 0.9) -> str:
    """Generate an approved plan review JSON response."""
    return json.dumps({
        "approved": True,
        "confidence": confidence,
        "issues": [],
        "revised_steps": None,
    })


def make_plan_review_rejected(issues: list[str] | None = None) -> str:
    """Generate a rejected plan review JSON response."""
    return json.dumps({
        "approved": False,
        "confidence": 0.3,
        "issues": issues or ["Plan is too vague"],
        "revised_steps": None,
    })


def make_verify_accept(confidence: float = 0.85) -> str:
    """Generate an accept verification JSON response."""
    return json.dumps({
        "verdict": "accept",
        "confidence": confidence,
        "issues": [],
        "feedback": "All steps completed correctly",
    })


def make_verify_reject(reason: str = "Incomplete") -> str:
    """Generate a reject verification JSON response."""
    return json.dumps({
        "verdict": "reject",
        "confidence": 0.3,
        "issues": [reason],
        "feedback": reason,
    })


def make_step_completed_response(result: str = "Step completed successfully") -> str:
    """Generate a step-completed JSON block for LLM response."""
    return json.dumps({
        "status": "completed",
        "result": result,
        "artifacts": [],
    })


def make_step_failed_response(error: str = "Something went wrong") -> str:
    """Generate a step-failed JSON block for LLM response."""
    return json.dumps({
        "status": "failed",
        "result": error,
    })


def make_step_blocked_response(blocker: str = "Need API key") -> str:
    """Generate a step-blocked JSON block for LLM response."""
    return json.dumps({
        "status": "blocked",
        "blocker_description": blocker,
    })


def make_retrospective_response() -> str:
    """Generate a retrospective JSON response."""
    return json.dumps({
        "summary": "Task executed successfully",
        "went_well": ["All steps completed"],
        "went_wrong": [],
        "learnings": ["Approach was sound"],
        "efficiency_notes": [],
    })
