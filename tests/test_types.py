"""Tests for genesis_task_executor.types module.

Covers TaskPhase enum, VALID_TRANSITIONS table, validate_transition(),
StepType with timeouts, and all dataclass result types.
"""

from __future__ import annotations

import pytest

from genesis_task_executor.types import (
    TERMINAL_PHASES,
    VALID_TRANSITIONS,
    ExecutionTrace,
    InvalidTransitionError,
    ResearchResult,
    StepResult,
    StepType,
    TaskPhase,
    ToolCall,
    ToolResponse,
    WorkaroundResult,
    validate_transition,
)

# ---------------------------------------------------------------------------
# TaskPhase enum
# ---------------------------------------------------------------------------


class TestTaskPhase:
    """TaskPhase has exactly 13 values with correct string representations."""

    ALL_PHASES = [
        ("PENDING", "pending"),
        ("REVIEWING", "reviewing"),
        ("PLANNING", "planning"),
        ("EXECUTING", "executing"),
        ("PAUSED", "paused"),
        ("VERIFYING", "verifying"),
        ("SYNTHESIZING", "synthesizing"),
        ("DELIVERING", "delivering"),
        ("RETROSPECTIVE", "retrospective"),
        ("COMPLETED", "completed"),
        ("BLOCKED", "blocked"),
        ("FAILED", "failed"),
        ("CANCELLED", "cancelled"),
    ]

    def test_phase_count(self):
        assert len(TaskPhase) == 13

    @pytest.mark.parametrize("attr,value", ALL_PHASES)
    def test_phase_value(self, attr: str, value: str):
        phase = TaskPhase[attr]
        assert phase.value == value
        assert str(phase) == value

    def test_phase_is_str_enum(self):
        """TaskPhase inherits StrEnum so it can be used as a string."""
        assert isinstance(TaskPhase.PENDING, str)
        assert TaskPhase.PENDING == "pending"

    def test_phase_from_value(self):
        assert TaskPhase("executing") is TaskPhase.EXECUTING

    def test_phase_invalid_value(self):
        with pytest.raises(ValueError):
            TaskPhase("nonexistent")


# ---------------------------------------------------------------------------
# VALID_TRANSITIONS table
# ---------------------------------------------------------------------------


class TestValidTransitions:

    def test_every_phase_has_entry(self):
        """Every TaskPhase must appear as a key in VALID_TRANSITIONS."""
        for phase in TaskPhase:
            assert phase in VALID_TRANSITIONS, f"{phase} missing from VALID_TRANSITIONS"

    def test_terminal_phases_have_empty_sets(self):
        for phase in TERMINAL_PHASES:
            assert VALID_TRANSITIONS[phase] == set(), (
                f"Terminal phase {phase} should have empty transition set"
            )

    def test_terminal_phases_are_correct(self):
        assert frozenset({
            TaskPhase.COMPLETED,
            TaskPhase.FAILED,
            TaskPhase.CANCELLED,
        }) == TERMINAL_PHASES

    def test_pending_can_reach_reviewing(self):
        assert TaskPhase.REVIEWING in VALID_TRANSITIONS[TaskPhase.PENDING]

    def test_executing_self_transition(self):
        """EXECUTING can transition to itself (next step)."""
        assert TaskPhase.EXECUTING in VALID_TRANSITIONS[TaskPhase.EXECUTING]

    def test_paused_can_resume_to_executing(self):
        assert TaskPhase.EXECUTING in VALID_TRANSITIONS[TaskPhase.PAUSED]

    def test_blocked_can_return_to_executing(self):
        assert TaskPhase.EXECUTING in VALID_TRANSITIONS[TaskPhase.BLOCKED]

    def test_blocked_can_return_to_reviewing(self):
        assert TaskPhase.REVIEWING in VALID_TRANSITIONS[TaskPhase.BLOCKED]

    def test_retrospective_cannot_cancel(self):
        """RETROSPECTIVE can only go to COMPLETED or FAILED — no cancel."""
        assert TaskPhase.CANCELLED not in VALID_TRANSITIONS[TaskPhase.RETROSPECTIVE]

    def test_every_non_terminal_can_reach_failed_or_cancelled(self):
        """Every non-terminal phase has at least FAILED or CANCELLED as a target."""
        for phase in TaskPhase:
            if phase in TERMINAL_PHASES:
                continue
            targets = VALID_TRANSITIONS[phase]
            can_fail = TaskPhase.FAILED in targets
            can_cancel = TaskPhase.CANCELLED in targets
            assert can_fail or can_cancel, (
                f"{phase} has no path to FAILED or CANCELLED"
            )

    def test_transition_values_are_sets(self):
        for phase, targets in VALID_TRANSITIONS.items():
            assert isinstance(targets, set), f"{phase} targets should be a set"

    def test_all_transition_targets_are_valid_phases(self):
        for phase, targets in VALID_TRANSITIONS.items():
            for target in targets:
                assert isinstance(target, TaskPhase), (
                    f"Target {target!r} from {phase} is not a TaskPhase"
                )


# ---------------------------------------------------------------------------
# validate_transition
# ---------------------------------------------------------------------------


class TestValidateTransition:

    def test_valid_transition_passes(self):
        """No exception for a valid transition."""
        validate_transition(TaskPhase.PENDING, TaskPhase.REVIEWING)

    def test_valid_transition_executing_to_verifying(self):
        validate_transition(TaskPhase.EXECUTING, TaskPhase.VERIFYING)

    def test_invalid_transition_raises(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(TaskPhase.PENDING, TaskPhase.COMPLETED)

    def test_invalid_transition_from_terminal(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(TaskPhase.COMPLETED, TaskPhase.PENDING)

    def test_invalid_transition_error_message(self):
        with pytest.raises(InvalidTransitionError, match="pending -> completed"):
            validate_transition(TaskPhase.PENDING, TaskPhase.COMPLETED)

    def test_self_transition_executing(self):
        """EXECUTING → EXECUTING is explicitly allowed."""
        validate_transition(TaskPhase.EXECUTING, TaskPhase.EXECUTING)

    def test_self_transition_pending_invalid(self):
        """PENDING → PENDING is not in the table."""
        with pytest.raises(InvalidTransitionError):
            validate_transition(TaskPhase.PENDING, TaskPhase.PENDING)


# ---------------------------------------------------------------------------
# StepType enum
# ---------------------------------------------------------------------------


class TestStepType:

    EXPECTED_TYPES = ["research", "code", "analysis", "synthesis", "verification", "external"]

    def test_step_type_count(self):
        assert len(StepType) == 6

    @pytest.mark.parametrize("value", EXPECTED_TYPES)
    def test_step_type_exists(self, value: str):
        st = StepType(value)
        assert st.value == value

    def test_research_timeout(self):
        assert StepType.RESEARCH.default_timeout_s == 3600

    def test_code_timeout(self):
        assert StepType.CODE.default_timeout_s == 3600

    def test_analysis_timeout(self):
        assert StepType.ANALYSIS.default_timeout_s == 1800

    def test_synthesis_timeout(self):
        assert StepType.SYNTHESIS.default_timeout_s == 1800

    def test_verification_timeout(self):
        assert StepType.VERIFICATION.default_timeout_s == 3600

    def test_external_timeout(self):
        assert StepType.EXTERNAL.default_timeout_s == 3600

    def test_all_timeouts_are_positive(self):
        for st in StepType:
            assert st.default_timeout_s > 0


# ---------------------------------------------------------------------------
# StepResult dataclass
# ---------------------------------------------------------------------------


class TestStepResult:

    def test_construction_minimal(self):
        r = StepResult(idx=0, status="completed", result="done")
        assert r.idx == 0
        assert r.status == "completed"
        assert r.result == "done"
        assert r.cost_usd == 0.0
        assert r.session_id is None
        assert r.model_used == ""
        assert r.duration_s == 0.0
        assert r.artifacts == []
        assert r.blocker_description is None

    def test_construction_full(self):
        r = StepResult(
            idx=3,
            status="blocked",
            result="waiting",
            cost_usd=0.05,
            session_id="sess-123",
            model_used="gpt-4o",
            duration_s=12.5,
            artifacts=["/tmp/out.txt"],
            blocker_description="Need credentials",
        )
        assert r.cost_usd == 0.05
        assert r.artifacts == ["/tmp/out.txt"]
        assert r.blocker_description == "Need credentials"

    def test_artifacts_default_is_independent(self):
        """Each instance gets its own list."""
        r1 = StepResult(idx=0, status="completed", result="a")
        r2 = StepResult(idx=1, status="completed", result="b")
        r1.artifacts.append("file.txt")
        assert r2.artifacts == []


# ---------------------------------------------------------------------------
# ExecutionTrace dataclass
# ---------------------------------------------------------------------------


class TestExecutionTrace:

    def test_construction(self):
        t = ExecutionTrace(task_id="t1", initiated_by="user", user_request="do stuff")
        assert t.task_id == "t1"
        assert t.plan == []
        assert t.step_results == []
        assert t.quality_gate == {}
        assert t.total_cost_usd == 0.0
        assert t.retrospective_notes == ""

    def test_mutable_fields(self):
        t = ExecutionTrace(task_id="t1", initiated_by="user", user_request="x")
        t.plan.append("step 1")
        t.step_results.append(StepResult(idx=0, status="completed", result="ok"))
        assert len(t.plan) == 1
        assert len(t.step_results) == 1


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestWorkaroundResult:

    def test_construction(self):
        w = WorkaroundResult(found=True, approach="retry with backoff")
        assert w.found is True
        assert w.approach == "retry with backoff"

    def test_frozen(self):
        w = WorkaroundResult(found=False)
        with pytest.raises(AttributeError):
            w.found = True  # type: ignore[misc]

    def test_default_approach_is_none(self):
        w = WorkaroundResult(found=False)
        assert w.approach is None


class TestResearchResult:

    def test_construction_found(self):
        r = ResearchResult(
            found=True,
            approach="Use API v2",
            sources=["https://docs.example.com"],
        )
        assert r.found is True
        assert r.approach == "Use API v2"
        assert r.sources == ["https://docs.example.com"]
        assert r.clues is None
        assert r.concrete_blockers == []
        assert r.session_id is None

    def test_construction_not_found(self):
        r = ResearchResult(
            found=False,
            clues="Might need a different auth flow",
            concrete_blockers=["Need CAPTCHA solver"],
        )
        assert r.found is False
        assert r.approach is None
        assert r.concrete_blockers == ["Need CAPTCHA solver"]

    def test_frozen(self):
        r = ResearchResult(found=True)
        with pytest.raises(AttributeError):
            r.found = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolCall / ToolResponse
# ---------------------------------------------------------------------------


class TestToolCall:

    def test_construction(self):
        tc = ToolCall(id="tc-1", name="read_file", arguments={"path": "/tmp/x"})
        assert tc.id == "tc-1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/x"}


class TestToolResponse:

    def test_construction_text_only(self):
        tr = ToolResponse(content="Hello")
        assert tr.content == "Hello"
        assert tr.tool_calls == []
        assert tr.stop_reason == ""

    def test_construction_with_tool_calls(self):
        tc = ToolCall(id="tc-1", name="read_file", arguments={"path": "/tmp/x"})
        tr = ToolResponse(content=None, tool_calls=[tc], stop_reason="tool_use")
        assert tr.content is None
        assert len(tr.tool_calls) == 1
        assert tr.stop_reason == "tool_use"
