"""Tests for genesis_task_executor.dispatcher module."""

from __future__ import annotations

import json

import pytest

from genesis_task_executor.db import create_task, get_task, update_task_phase
from genesis_task_executor.dispatcher import TaskDispatcher
from genesis_task_executor.engine import TaskExecutor
from genesis_task_executor.recovery import RecoveryCascade
from genesis_task_executor.research import Researcher
from genesis_task_executor.review import TaskReviewer
from genesis_task_executor.step_runner import StepRunner
from genesis_task_executor.types import ToolResponse

from .conftest import (
    MockLLMProvider,
    make_plan_review_approved,
    make_retrospective_response,
    make_step_completed_response,
    make_verify_accept,
)


def _build_system(provider: MockLLMProvider, temp_db):
    """Build executor + dispatcher with mock provider."""
    researcher = Researcher(provider=provider)
    reviewer = TaskReviewer(primary_provider=provider)
    recovery = RecoveryCascade(provider=provider, researcher=researcher)
    runner = StepRunner(provider=provider, database=temp_db)

    executor = TaskExecutor(
        provider=provider,
        database=temp_db,
        reviewer=reviewer,
        recovery=recovery,
        step_runner=runner,
    )
    dispatcher = TaskDispatcher(executor=executor, database=temp_db)
    return executor, dispatcher


def _simple_plan() -> dict:
    return {
        "goal": "Test",
        "steps": [
            {"idx": 0, "type": "analysis", "description": "Do it",
             "success_criterion": "Done"},
        ],
    }


def _queue_happy_path(provider: MockLLMProvider) -> None:
    """Queue all responses needed for a successful task execution."""
    plan = _simple_plan()
    provider.enqueue(json.dumps(plan))              # Plan generation
    provider.enqueue(make_plan_review_approved())    # Plan review
    provider.enqueue_tool_response(                  # Step execution
        ToolResponse(content=make_step_completed_response(
            "Task completed successfully with full analysis results written"
        ))
    )
    provider.enqueue(make_verify_accept())           # Fresh-eyes
    provider.enqueue(make_verify_accept(0.9))        # Adversarial
    provider.enqueue(make_retrospective_response())  # Retrospective


class TestSubmit:

    async def test_submit_creates_task_returns_id(self, temp_db):
        provider = MockLLMProvider()
        _queue_happy_path(provider)

        _, dispatcher = _build_system(provider, temp_db)
        task_id = await dispatcher.submit("Test task")

        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID

        # Task should exist in DB
        task = await get_task(temp_db, task_id)
        assert task is not None
        assert task["description"] == "Test task"

    async def test_submit_empty_description_raises(self, temp_db):
        provider = MockLLMProvider()
        _, dispatcher = _build_system(provider, temp_db)

        with pytest.raises(ValueError, match="empty"):
            await dispatcher.submit("")

    async def test_submit_whitespace_description_raises(self, temp_db):
        provider = MockLLMProvider()
        _, dispatcher = _build_system(provider, temp_db)

        with pytest.raises(ValueError, match="empty"):
            await dispatcher.submit("   ")


class TestSubmitAndWait:

    async def test_submit_and_wait_returns_completed_task(self, temp_db):
        provider = MockLLMProvider()
        _queue_happy_path(provider)

        _, dispatcher = _build_system(provider, temp_db)
        result = await dispatcher.submit_and_wait("Complete this task")

        assert result["current_phase"] == "completed"
        assert result["description"] == "Complete this task"

    async def test_submit_and_wait_with_prebuilt_plan(self, temp_db):
        """When a prebuilt plan is provided, the engine skips plan generation
        but still transitions through REVIEWING and reviews the plan."""
        provider = MockLLMProvider()
        plan = _simple_plan()

        # No plan generation (plan already provided)
        # Plan review
        provider.enqueue(make_plan_review_approved())
        # Step execution (step_runner calls complete_with_tools)
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response(
                "Task completed with full analysis results written to output file"
            ))
        )
        # Verification: fresh-eyes (primary) + adversarial (secondary)
        # verify_deliverable calls _fresh_eyes_review then _adversarial_review
        provider.enqueue(make_verify_accept())      # fresh-eyes
        provider.enqueue(make_verify_accept(0.9))   # adversarial
        # Retrospective
        provider.enqueue(make_retrospective_response())

        _, dispatcher = _build_system(provider, temp_db)
        result = await dispatcher.submit_and_wait("Task with plan", plan=plan)

        assert result["current_phase"] == "completed"
        assert result["verdict"] == "accept"


class TestRecoverIncomplete:

    async def test_recovers_resumable_tasks(self, temp_db):
        """Tasks in EXECUTING phase are recovered."""
        provider = MockLLMProvider()
        executor, dispatcher = _build_system(provider, temp_db)

        plan = _simple_plan()
        task_id = await create_task(
            temp_db, description="Interrupted",
            plan_json=json.dumps(plan),
        )
        await update_task_phase(temp_db, task_id, "executing")

        # Queue responses for resumed execution
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Resumed"))
        )
        provider.enqueue(make_verify_accept())
        provider.enqueue(make_verify_accept(0.9))
        provider.enqueue(make_retrospective_response())

        recovered = await dispatcher.recover_incomplete()
        assert recovered == 1

    async def test_skips_non_resumable_phases(self, temp_db):
        """Tasks in PLANNING phase are not resumable."""
        provider = MockLLMProvider()
        executor, dispatcher = _build_system(provider, temp_db)

        task_id = await create_task(temp_db, description="Planning stage")
        await update_task_phase(temp_db, task_id, "planning")

        recovered = await dispatcher.recover_incomplete()
        assert recovered == 0

    async def test_skips_already_dispatched(self, temp_db):
        """Already-dispatched tasks are not recovered again."""
        provider = MockLLMProvider()
        executor, dispatcher = _build_system(provider, temp_db)

        plan = _simple_plan()
        task_id = await create_task(
            temp_db, description="Already running",
            plan_json=json.dumps(plan),
        )
        await update_task_phase(temp_db, task_id, "executing")

        # Mark as already dispatched
        dispatcher._dispatched.add(task_id)

        recovered = await dispatcher.recover_incomplete()
        assert recovered == 0


class TestTaskControl:

    async def test_pause_inactive_task_returns_false(self, temp_db):
        provider = MockLLMProvider()
        _, dispatcher = _build_system(provider, temp_db)

        result = dispatcher.pause_task("nonexistent")
        assert result is False

    async def test_resume_inactive_task_returns_false(self, temp_db):
        provider = MockLLMProvider()
        _, dispatcher = _build_system(provider, temp_db)

        result = dispatcher.resume_task("nonexistent")
        assert result is False

    async def test_cancel_inactive_task_returns_false(self, temp_db):
        provider = MockLLMProvider()
        _, dispatcher = _build_system(provider, temp_db)

        result = dispatcher.cancel_task("nonexistent")
        assert result is False
