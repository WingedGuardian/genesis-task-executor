"""Type definitions for the task executor system.

Defines lifecycle phases, step types, result objects, execution traces,
and the canonical state transition table. Ported from Genesis's production
executor with zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Task phases (state machine states)
# ---------------------------------------------------------------------------


class TaskPhase(StrEnum):
    """Lifecycle phases for a task."""

    PENDING = "pending"
    REVIEWING = "reviewing"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    DELIVERING = "delivering"
    RETROSPECTIVE = "retrospective"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[TaskPhase, set[TaskPhase]] = {
    TaskPhase.PENDING: {TaskPhase.REVIEWING, TaskPhase.FAILED, TaskPhase.CANCELLED},
    TaskPhase.REVIEWING: {
        TaskPhase.PLANNING,
        TaskPhase.BLOCKED,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.PLANNING: {
        TaskPhase.EXECUTING,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.EXECUTING: {
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.PAUSED,
        TaskPhase.BLOCKED,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.PAUSED: {
        TaskPhase.EXECUTING,
        TaskPhase.CANCELLED,
    },
    TaskPhase.VERIFYING: {
        TaskPhase.EXECUTING,
        TaskPhase.SYNTHESIZING,
        TaskPhase.BLOCKED,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.SYNTHESIZING: {
        TaskPhase.DELIVERING,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.DELIVERING: {
        TaskPhase.RETROSPECTIVE,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    },
    TaskPhase.RETROSPECTIVE: {
        TaskPhase.COMPLETED,
        TaskPhase.FAILED,
    },
    TaskPhase.BLOCKED: {
        TaskPhase.REVIEWING,
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.CANCELLED,
    },
    # Terminal states
    TaskPhase.COMPLETED: set(),
    TaskPhase.FAILED: set(),
    TaskPhase.CANCELLED: set(),
}

TERMINAL_PHASES = frozenset({TaskPhase.COMPLETED, TaskPhase.FAILED, TaskPhase.CANCELLED})


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""


def validate_transition(from_phase: TaskPhase, to_phase: TaskPhase) -> None:
    """Raise InvalidTransitionError if the transition is not allowed."""
    allowed = VALID_TRANSITIONS.get(from_phase, set())
    if to_phase not in allowed:
        raise InvalidTransitionError(
            f"Invalid transition: {from_phase.value} -> {to_phase.value}. "
            f"Allowed: {sorted(p.value for p in allowed)}"
        )


# ---------------------------------------------------------------------------
# Step types with default timeouts
# ---------------------------------------------------------------------------


class StepType(StrEnum):
    """Classification of task steps for routing and timeout configuration."""

    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    EXTERNAL = "external"

    @property
    def default_timeout_s(self) -> int:
        """Default timeout in seconds for this step type."""
        return _STEP_TIMEOUTS.get(self, 600)


_STEP_TIMEOUTS: dict[StepType, int] = {
    StepType.RESEARCH: 3600,
    StepType.CODE: 3600,
    StepType.ANALYSIS: 1800,
    StepType.SYNTHESIS: 1800,
    StepType.VERIFICATION: 3600,
    StepType.EXTERNAL: 3600,
}


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of executing a single task step."""

    idx: int
    status: str  # "completed", "blocked", "failed"
    result: str
    cost_usd: float = 0.0
    session_id: str | None = None
    model_used: str = ""
    duration_s: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    blocker_description: str | None = None


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    """Full trace of a task execution for retrospective analysis."""

    task_id: str
    initiated_by: str
    user_request: str
    plan: list[str] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    quality_gate: dict = field(default_factory=dict)
    total_cost_usd: float = 0.0
    retrospective_notes: str = ""


# ---------------------------------------------------------------------------
# Recovery result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkaroundResult:
    """Result of a procedural workaround search."""

    found: bool
    approach: str | None = None


@dataclass(frozen=True)
class ResearchResult:
    """Result of a deep research session investigating a blocker."""

    found: bool
    approach: str | None = None
    sources: list[str] = field(default_factory=list)
    clues: str | None = None
    concrete_blockers: list[str] = field(default_factory=list)
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Tool response (from LLM tool use)
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResponse:
    """Response from an LLM completion with tool use."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
