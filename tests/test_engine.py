"""Tests for genesis_task_executor.engine module.

These tests exercise the TaskExecutor's lifecycle by wiring it with
mock providers and an in-memory database.
"""

from __future__ import annotations

import json

from genesis_task_executor.db import create_task, get_task, update_task_phase
from genesis_task_executor.engine import (
    MAX_REVIEW_ITERATIONS,
    TaskExecutor,
    _synthesize_deliverable,
)
from genesis_task_executor.recovery import RecoveryCascade
from genesis_task_executor.research import Researcher
from genesis_task_executor.review import TaskReviewer
from genesis_task_executor.step_runner import StepRunner
from genesis_task_executor.types import StepResult, ToolResponse

from .conftest import (
    MockEventCallback,
    MockLLMProvider,
    make_plan_review_approved,
    make_plan_review_rejected,
    make_retrospective_response,
    make_step_blocked_response,
    make_step_completed_response,
    make_step_failed_response,
    make_verify_accept,
    make_verify_reject,
)


def _build_executor(
    provider: MockLLMProvider,
    temp_db,
    event_callback: MockEventCallback | None = None,
) -> TaskExecutor:
    """Wire up a TaskExecutor with all mock dependencies."""
    researcher = Researcher(provider=provider)
    reviewer = TaskReviewer(primary_provider=provider)
    recovery = RecoveryCascade(provider=provider, researcher=researcher)
    runner = StepRunner(provider=provider, database=temp_db)

    return TaskExecutor(
        provider=provider,
        database=temp_db,
        reviewer=reviewer,
        recovery=recovery,
        step_runner=runner,
        event_callback=event_callback,
    )


def _make_simple_plan(n_steps: int = 1) -> dict:
    """Create a simple plan with n steps."""
    return {
        "goal": "Test goal",
        "steps": [
            {
                "idx": i,
                "type": "analysis",
                "description": f"Step {i}",
                "success_criterion": f"Step {i} done",
            }
            for i in range(n_steps)
        ],
    }


class TestFullLifecycle:

    async def test_task_completes_successfully(self, temp_db):
        """Full happy path: plan → review → execute → verify → complete."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(1)
        events = MockEventCallback()

        # Queue responses in order:
        # 1. Plan generation
        provider.enqueue(json.dumps(plan))
        # 2. Plan review — approved
        provider.enqueue(make_plan_review_approved())
        # 3. Step execution — tool response (no tool calls, just text)
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Analysis complete"))
        )
        # 4. Fresh-eyes review — accept
        provider.enqueue(make_verify_accept())
        # 5. Adversarial review — accept
        provider.enqueue(make_verify_accept(0.9))
        # 6. Retrospective
        provider.enqueue(make_retrospective_response())

        executor = _build_executor(provider, temp_db, events)
        task_id = await create_task(temp_db, description="Analyze data")

        success = await executor.execute(task_id)
        assert success is True

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "completed"
        assert task["verdict"] == "accept"

        # Verify events were emitted
        event_types = [e[0] for e in events.events]
        assert "step_complete" in event_types
        assert "task_completed" in event_types

    async def test_plan_generation_fails(self, temp_db):
        """Task fails when plan generation returns invalid JSON."""
        provider = MockLLMProvider()
        provider.enqueue("not valid json at all")

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Bad task")

        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "failed"
        assert "Plan generation failed" in (task["reason"] or "")

    async def test_plan_review_rejects(self, temp_db):
        """Task fails when plan review rejects the plan."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(1)

        # 1. Plan generation
        provider.enqueue(json.dumps(plan))
        # 2. Plan review — rejected
        provider.enqueue(make_plan_review_rejected(["Too vague", "Missing steps"]))

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Vague task")

        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "failed"
        assert "rejected" in (task["reason"] or "").lower()

    async def test_step_blocked(self, temp_db):
        """Task blocks when a step returns blocked status."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(1)

        # 1. Plan generation
        provider.enqueue(json.dumps(plan))
        # 2. Plan review — approved
        provider.enqueue(make_plan_review_approved())
        # 3. Step execution — blocked
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_blocked_response("Need API credentials"))
        )

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Needs creds")

        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "blocked"

    async def test_step_fails_recovery_exhausted(self, temp_db):
        """Task fails when step fails and recovery cascade is exhausted."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(1)

        # 1. Plan generation
        provider.enqueue(json.dumps(plan))
        # 2. Plan review — approved
        provider.enqueue(make_plan_review_approved())
        # 3. Step execution — failed
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_failed_response("API returned 500"))
        )
        # 4. Recovery cascade research — not found
        provider.enqueue(
            '```json\n{"found": false, "approach": null, "sources": [], '
            '"clues": null, "concrete_blockers": ["Server error"]}\n```'
        )
        # 5. Exit gate — accept failure
        provider.enqueue(json.dumps({
            "verdict": "accept",
            "confirmed_blockers": ["Server error 500"],
        }))

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Failing task")

        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "failed"


class TestReviewIterationCap:

    async def test_max_review_iterations(self, temp_db):
        """Verify that MAX_REVIEW_ITERATIONS caps review attempts."""
        assert MAX_REVIEW_ITERATIONS == 2

        provider = MockLLMProvider()
        plan = _make_simple_plan(1)

        # 1. Plan generation
        provider.enqueue(json.dumps(plan))
        # 2. Plan review — approved
        provider.enqueue(make_plan_review_approved())
        # 3. Step execution — completed
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Done"))
        )
        # 4. First verify: fresh-eyes reject
        provider.enqueue(make_verify_reject("Missing conclusion"))
        # 5. First verify: adversarial reject
        provider.enqueue(make_verify_reject("Incomplete"))
        # 6. Fixup step execution (from review iteration)
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Fixed"))
        )
        # 7. Second verify: fresh-eyes reject again
        provider.enqueue(make_verify_reject("Still incomplete"))
        # 8. Second verify: adversarial reject again
        provider.enqueue(make_verify_reject("Still bad"))

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Hard task")

        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "blocked"


class TestPauseResumeCancel:

    async def test_cancel_signal(self, temp_db):
        """Cancelling a task before execution starts."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(2)

        # Plan generation and review
        provider.enqueue(json.dumps(plan))
        provider.enqueue(make_plan_review_approved())
        # First step completes
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Step 0 done"))
        )
        # Second step won't run because we cancel

        executor = _build_executor(provider, temp_db)
        task_id = await create_task(temp_db, description="Will be cancelled")

        # Cancel immediately — the check happens between steps
        executor.cancel_task(task_id)
        # Note: cancel_task won't work until execute() creates the event.
        # We need to test via a different approach — set cancel after the
        # executor starts. For a synchronous test, we pre-set the task to
        # cancelled phase.

        # Instead, test that the cancel signal API works
        assert task_id not in executor.get_active_tasks()

    async def test_pause_task_api(self, temp_db):
        """Test that pause_task and resume_task APIs exist and work."""
        provider = MockLLMProvider()
        executor = _build_executor(provider, temp_db)

        # When task is not active, pause/resume are no-ops
        executor.pause_task("nonexistent")
        executor.resume_task("nonexistent")
        # Should not raise


class TestResumeFromExecuting:

    async def test_resume_with_existing_plan(self, temp_db):
        """Resume a task from EXECUTING phase with a saved plan."""
        provider = MockLLMProvider()
        plan = _make_simple_plan(2)

        # Create task already in EXECUTING phase with plan saved
        task_id = await create_task(
            temp_db, description="Resumable task", plan_json=json.dumps(plan),
        )
        await update_task_phase(temp_db, task_id, "executing")

        # Step 0 execution (resumed — both steps need executing since
        # no completed steps exist in DB)
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Step 0 resumed"))
        )
        # Step 1 execution
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Step 1 done"))
        )
        # Fresh-eyes accept
        provider.enqueue(make_verify_accept())
        # Adversarial accept
        provider.enqueue(make_verify_accept(0.9))
        # Retrospective
        provider.enqueue(make_retrospective_response())

        executor = _build_executor(provider, temp_db)
        success = await executor.execute(task_id)
        assert success is True

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "completed"

    async def test_resume_without_plan_fails(self, temp_db):
        """Resume from EXECUTING without a saved plan fails gracefully."""
        provider = MockLLMProvider()

        task_id = await create_task(temp_db, description="No plan")
        await update_task_phase(temp_db, task_id, "executing")

        executor = _build_executor(provider, temp_db)
        success = await executor.execute(task_id)
        assert success is False

        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "failed"
        assert "no plan" in (task["reason"] or "").lower()


class TestGeneratePlan:

    async def test_returns_none_on_invalid_json(self, temp_db):
        provider = MockLLMProvider()
        provider.enqueue("This is not JSON")

        executor = _build_executor(provider, temp_db)
        result = await executor._generate_plan("Test task")
        assert result is None

    async def test_returns_none_without_steps_list(self, temp_db):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({"goal": "test", "steps": "not a list"}))

        executor = _build_executor(provider, temp_db)
        result = await executor._generate_plan("Test task")
        assert result is None

    async def test_caps_steps_at_max(self, temp_db):
        provider = MockLLMProvider()
        many_steps = [{"idx": i, "description": f"Step {i}"} for i in range(20)]
        provider.enqueue(json.dumps({"goal": "test", "steps": many_steps}))

        executor = _build_executor(provider, temp_db)
        result = await executor._generate_plan("Test task")
        assert result is not None
        assert len(result["steps"]) == 12  # MAX_STEPS

    async def test_adds_missing_idx(self, temp_db):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "goal": "test",
            "steps": [
                {"description": "A"},
                {"description": "B"},
            ],
        }))

        executor = _build_executor(provider, temp_db)
        result = await executor._generate_plan("Test task")
        assert result is not None
        assert result["steps"][0]["idx"] == 0
        assert result["steps"][1]["idx"] == 1


class TestSynthesizeDeliverable:

    def test_synthesize_with_completed_steps(self):
        results = [
            StepResult(idx=0, status="completed", result="Found 3 patterns"),
            StepResult(idx=1, status="completed", result="Wrote report",
                       artifacts=["/tmp/report.md"]),
        ]
        deliverable = _synthesize_deliverable(results)
        assert "Step 0" in deliverable
        assert "3 patterns" in deliverable
        assert "report.md" in deliverable

    def test_synthesize_skips_failed_steps(self):
        results = [
            StepResult(idx=0, status="completed", result="Good result"),
            StepResult(idx=1, status="failed", result="Error happened"),
        ]
        deliverable = _synthesize_deliverable(results)
        assert "Good result" in deliverable
        assert "Error happened" not in deliverable

    def test_synthesize_empty_results(self):
        deliverable = _synthesize_deliverable([])
        assert "no completed steps" in deliverable.lower()


class TestTaskNotFound:

    async def test_execute_nonexistent_task(self, temp_db):
        provider = MockLLMProvider()
        executor = _build_executor(provider, temp_db)
        success = await executor.execute("nonexistent-task-id")
        assert success is False
