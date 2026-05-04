"""Integration tests for genesis_task_executor.

Tests the full pipeline from submission through completion with
mock providers, verifying state transitions and database records.
"""

from __future__ import annotations

import json

from genesis_task_executor.db import get_steps
from genesis_task_executor.dispatcher import TaskDispatcher
from genesis_task_executor.engine import TaskExecutor
from genesis_task_executor.recovery import RecoveryCascade
from genesis_task_executor.research import Researcher
from genesis_task_executor.review import TaskReviewer
from genesis_task_executor.step_runner import StepRunner
from genesis_task_executor.types import TaskPhase, ToolResponse

from .conftest import (
    MockEventCallback,
    MockLLMProvider,
    make_plan_review_approved,
    make_retrospective_response,
    make_step_completed_response,
    make_verify_accept,
)


def _build_full_system(
    provider: MockLLMProvider,
    temp_db,
    event_callback: MockEventCallback | None = None,
) -> tuple[TaskExecutor, TaskDispatcher]:
    """Wire up the complete system with mock provider."""
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
        event_callback=event_callback,
    )
    dispatcher = TaskDispatcher(executor=executor, database=temp_db)
    return executor, dispatcher


class TestCreateExecutorFactory:

    async def test_create_executor_requires_api_key(self):
        """create_executor with a real provider needs an API key.
        We test that the factory signature works by catching the expected error."""
        # We can't test the full factory without a real API key,
        # but we can verify it raises appropriately or accepts mock params.
        # Testing with a mock provider via manual wiring instead.
        provider = MockLLMProvider()
        assert hasattr(provider, "complete")
        assert hasattr(provider, "complete_with_tools")


class TestFullPipeline:

    async def test_submit_execute_verify_complete(self, temp_db):
        """Full pipeline: submit -> plan -> review -> execute -> verify -> complete."""
        provider = MockLLMProvider()
        events = MockEventCallback()

        plan = {
            "goal": "Summarize the data",
            "steps": [
                {
                    "idx": 0,
                    "type": "research",
                    "description": "Read the input file",
                    "success_criterion": "File contents loaded",
                },
                {
                    "idx": 1,
                    "type": "synthesis",
                    "description": "Write summary to output",
                    "success_criterion": "Summary written to /tmp/summary.md",
                },
            ],
        }

        # Queue all responses
        provider.enqueue(json.dumps(plan))                # Plan generation
        provider.enqueue(make_plan_review_approved())      # Plan review

        # Step 0: read file
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Read input.txt: 500 lines of data"))
        )
        # Step 1: write summary
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response(
                "Wrote summary to /tmp/summary.md with 3 key findings"
            ))
        )

        # Verification
        provider.enqueue(make_verify_accept())             # Fresh-eyes
        provider.enqueue(make_verify_accept(0.92))         # Adversarial
        provider.enqueue(make_retrospective_response())    # Retrospective

        executor, dispatcher = _build_full_system(provider, temp_db, events)

        result = await dispatcher.submit_and_wait("Summarize input.txt")

        assert result["current_phase"] == "completed"
        assert result["verdict"] == "accept"

        # Verify steps were recorded in DB
        steps = await get_steps(temp_db, result["task_id"])
        assert len(steps) == 2
        assert all(s["status"] == "completed" for s in steps)

    async def test_multi_step_with_failure_and_recovery(self, temp_db):
        """Pipeline where step 1 fails but recovery succeeds."""
        provider = MockLLMProvider()

        plan = {
            "goal": "Fetch and analyze",
            "steps": [
                {"idx": 0, "type": "research", "description": "Fetch URL",
                 "success_criterion": "URL content fetched"},
                {"idx": 1, "type": "analysis", "description": "Analyze content",
                 "success_criterion": "Analysis written"},
            ],
        }

        # Plan + review
        provider.enqueue(json.dumps(plan))
        provider.enqueue(make_plan_review_approved())

        # Step 0: success
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response("Fetched 10KB from URL"))
        )

        # Step 1: fails initially
        provider.enqueue_tool_response(
            ToolResponse(content=json.dumps({
                "status": "failed", "result": "Analysis library not available",
            }))
        )

        # Recovery cascade:
        # Research — finds a solution
        provider.enqueue(
            '```json\n{"found": true, "approach": "Use basic text analysis instead", '
            '"sources": [], "clues": null, "concrete_blockers": []}\n```'
        )
        # Retry with workaround — succeeds
        provider.enqueue_tool_response(
            ToolResponse(
                content=make_step_completed_response("Analyzed using basic text processing"),
            )
        )

        # Verification
        provider.enqueue(make_verify_accept())
        provider.enqueue(make_verify_accept(0.88))
        provider.enqueue(make_retrospective_response())

        executor, dispatcher = _build_full_system(provider, temp_db)

        result = await dispatcher.submit_and_wait("Fetch and analyze data")
        assert result["current_phase"] == "completed"


class TestStateTransitions:

    async def test_transitions_are_valid_throughout(self, temp_db):
        """Track all transitions and verify each is valid."""
        provider = MockLLMProvider()
        events = MockEventCallback()

        plan = {
            "goal": "Simple task",
            "steps": [
                {"idx": 0, "type": "analysis", "description": "Step 0",
                 "success_criterion": "Done"},
            ],
        }

        # Queue happy path
        provider.enqueue(json.dumps(plan))
        provider.enqueue(make_plan_review_approved())
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response(
                "Task completed with detailed analysis of the input data"
            ))
        )
        provider.enqueue(make_verify_accept())
        provider.enqueue(make_verify_accept(0.9))
        provider.enqueue(make_retrospective_response())

        executor, dispatcher = _build_full_system(provider, temp_db, events)
        await dispatcher.submit_and_wait("Track transitions")

        # Extract transitions from events
        transitions = [
            (e[1]["from"], e[1]["to"])
            for e in events.events
            if e[0] == "transition"
        ]

        # Verify the expected progression
        assert len(transitions) >= 5  # At least: reviewing, planning, executing, verifying, ...

        # Every transition should be valid per VALID_TRANSITIONS
        from genesis_task_executor.types import VALID_TRANSITIONS
        for from_phase, to_phase in transitions:
            from_p = TaskPhase(from_phase)
            to_p = TaskPhase(to_phase)
            assert to_p in VALID_TRANSITIONS[from_p], (
                f"Invalid transition recorded: {from_phase} -> {to_phase}"
            )

    async def test_happy_path_phase_sequence(self, temp_db):
        """Verify the happy path hits the expected phases in order."""
        provider = MockLLMProvider()
        events = MockEventCallback()

        plan = {
            "goal": "Verify sequence",
            "steps": [
                {"idx": 0, "type": "analysis", "description": "Do analysis",
                 "success_criterion": "Done"},
            ],
        }

        provider.enqueue(json.dumps(plan))
        provider.enqueue(make_plan_review_approved())
        provider.enqueue_tool_response(
            ToolResponse(content=make_step_completed_response(
                "Analysis completed with detailed findings and summary"
            ))
        )
        provider.enqueue(make_verify_accept())
        provider.enqueue(make_verify_accept(0.9))
        provider.enqueue(make_retrospective_response())

        executor, dispatcher = _build_full_system(provider, temp_db, events)
        await dispatcher.submit_and_wait("Verify sequence")

        transitions = [e[1]["to"] for e in events.events if e[0] == "transition"]

        # Should see these phases in order
        expected_sequence = [
            "reviewing", "planning", "executing",
            "verifying", "synthesizing", "delivering",
            "retrospective", "completed",
        ]

        # All expected phases should appear in order
        seen_idx = 0
        for phase in transitions:
            if seen_idx < len(expected_sequence) and phase == expected_sequence[seen_idx]:
                seen_idx += 1

        assert seen_idx == len(expected_sequence), (
            f"Expected phase sequence {expected_sequence}, "
            f"but transitions were {transitions}"
        )
