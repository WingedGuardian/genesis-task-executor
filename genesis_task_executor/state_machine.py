"""State machine utilities for task lifecycle management.

Wraps the transition table from types.py with convenience functions
for phase queries and lifecycle analysis.
"""

from __future__ import annotations

from genesis_task_executor.types import (
    TERMINAL_PHASES,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    TaskPhase,
    validate_transition,
)

__all__ = [
    "TaskPhase",
    "VALID_TRANSITIONS",
    "TERMINAL_PHASES",
    "InvalidTransitionError",
    "validate_transition",
    "is_terminal",
    "is_active",
    "is_resumable",
    "allowed_transitions",
]


def is_terminal(phase: TaskPhase) -> bool:
    """Check if a phase is terminal (no transitions out)."""
    return phase in TERMINAL_PHASES


def is_active(phase: TaskPhase) -> bool:
    """Check if a phase represents active work."""
    return phase in {
        TaskPhase.REVIEWING,
        TaskPhase.PLANNING,
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.SYNTHESIZING,
        TaskPhase.DELIVERING,
        TaskPhase.RETROSPECTIVE,
    }


def is_resumable(phase: TaskPhase) -> bool:
    """Check if a task in this phase can be resumed after a crash."""
    return phase in {
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.BLOCKED,
        TaskPhase.PAUSED,
    }


def allowed_transitions(phase: TaskPhase) -> set[TaskPhase]:
    """Return the set of valid next phases from the given phase."""
    return VALID_TRANSITIONS.get(phase, set())
