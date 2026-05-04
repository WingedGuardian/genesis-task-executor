"""Tests for genesis_task_executor.state_machine module."""

from __future__ import annotations

import pytest

from genesis_task_executor.state_machine import (
    allowed_transitions,
    is_active,
    is_resumable,
    is_terminal,
)
from genesis_task_executor.types import TaskPhase


class TestIsTerminal:

    def test_completed_is_terminal(self):
        assert is_terminal(TaskPhase.COMPLETED) is True

    def test_failed_is_terminal(self):
        assert is_terminal(TaskPhase.FAILED) is True

    def test_cancelled_is_terminal(self):
        assert is_terminal(TaskPhase.CANCELLED) is True

    def test_pending_is_not_terminal(self):
        assert is_terminal(TaskPhase.PENDING) is False

    def test_executing_is_not_terminal(self):
        assert is_terminal(TaskPhase.EXECUTING) is False


class TestIsActive:

    ACTIVE_PHASES = [
        TaskPhase.REVIEWING,
        TaskPhase.PLANNING,
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.SYNTHESIZING,
        TaskPhase.DELIVERING,
        TaskPhase.RETROSPECTIVE,
    ]

    INACTIVE_PHASES = [
        TaskPhase.PENDING,
        TaskPhase.PAUSED,
        TaskPhase.COMPLETED,
        TaskPhase.BLOCKED,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    ]

    @pytest.mark.parametrize("phase", ACTIVE_PHASES)
    def test_active_phases(self, phase):
        assert is_active(phase) is True

    @pytest.mark.parametrize("phase", INACTIVE_PHASES)
    def test_inactive_phases(self, phase):
        assert is_active(phase) is False


class TestIsResumable:

    RESUMABLE = [
        TaskPhase.EXECUTING,
        TaskPhase.VERIFYING,
        TaskPhase.BLOCKED,
        TaskPhase.PAUSED,
    ]

    NOT_RESUMABLE = [
        TaskPhase.PENDING,
        TaskPhase.REVIEWING,
        TaskPhase.PLANNING,
        TaskPhase.SYNTHESIZING,
        TaskPhase.DELIVERING,
        TaskPhase.RETROSPECTIVE,
        TaskPhase.COMPLETED,
        TaskPhase.FAILED,
        TaskPhase.CANCELLED,
    ]

    @pytest.mark.parametrize("phase", RESUMABLE)
    def test_resumable_phases(self, phase):
        assert is_resumable(phase) is True

    @pytest.mark.parametrize("phase", NOT_RESUMABLE)
    def test_not_resumable_phases(self, phase):
        assert is_resumable(phase) is False


class TestAllowedTransitions:

    def test_pending_transitions(self):
        result = allowed_transitions(TaskPhase.PENDING)
        assert TaskPhase.REVIEWING in result
        assert TaskPhase.FAILED in result
        assert TaskPhase.CANCELLED in result

    def test_terminal_returns_empty(self):
        assert allowed_transitions(TaskPhase.COMPLETED) == set()

    def test_returns_set(self):
        result = allowed_transitions(TaskPhase.EXECUTING)
        assert isinstance(result, set)
