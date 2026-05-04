"""Tests for genesis_task_executor.recovery module."""

from __future__ import annotations

import json

from genesis_task_executor.recovery import (
    MAX_EXIT_GATE_CYCLES,
    RecoveryCascade,
    run_recovery_cascade,
)
from genesis_task_executor.research import Researcher
from genesis_task_executor.types import ResearchResult, StepResult

from .conftest import (
    MockKnowledgeStore,
    MockLLMProvider,
    MockProcedureStore,
    MockWebSearcher,
)


def _make_cascade(
    provider: MockLLMProvider | None = None,
    procedure_results: list[dict] | None = None,
    web_results: list[dict] | None = None,
    knowledge_results: list[dict] | None = None,
) -> RecoveryCascade:
    """Helper to build a RecoveryCascade with mocks."""
    prov = provider or MockLLMProvider()
    researcher = Researcher(
        provider=prov,
        web_searcher=MockWebSearcher(web_results),
        knowledge_store=MockKnowledgeStore(knowledge_results),
    )
    return RecoveryCascade(
        provider=prov,
        researcher=researcher,
        procedure_store=MockProcedureStore(procedure_results),
    )


SAMPLE_STEP = {"idx": 0, "type": "code", "description": "Write a parser"}
SAMPLE_ERROR = "ERROR: ConnectionRefused to api.example.com"


class TestTryWorkaround:

    async def test_matching_procedure_returns_workaround(self):
        cascade = _make_cascade(
            procedure_results=[
                {"approach": "Use retry with exponential backoff", "confidence": 0.8},
            ],
        )
        result = await cascade.try_workaround(SAMPLE_STEP, SAMPLE_ERROR, [])
        assert result is not None
        assert result.found is True
        assert "retry" in result.approach.lower()

    async def test_no_matches_returns_none(self):
        cascade = _make_cascade(procedure_results=None)
        result = await cascade.try_workaround(SAMPLE_STEP, SAMPLE_ERROR, [])
        assert result is None

    async def test_empty_list_returns_none(self):
        """ProcedureStore returns empty list (distinct from None)."""
        cascade = _make_cascade()
        # MockProcedureStore with None returns None; override to empty list
        cascade._procedures = MockProcedureStore([])
        result = await cascade.try_workaround(SAMPLE_STEP, SAMPLE_ERROR, [])
        # Empty list is falsy, so returns None
        assert result is None

    async def test_approach_as_list(self):
        """When approach is a list of steps, it gets joined."""
        cascade = _make_cascade(
            procedure_results=[
                {"approach": ["Step 1: retry", "Step 2: validate"], "confidence": 0.7},
            ],
        )
        result = await cascade.try_workaround(SAMPLE_STEP, SAMPLE_ERROR, [])
        assert result is not None
        assert "Step 1" in result.approach
        assert "Step 2" in result.approach


class TestTryDueDiligence:

    async def test_relevant_results(self):
        """When web/knowledge search finds relevant results, returns context."""
        provider = MockLLMProvider()
        # Triage response (not "NOT_RELEVANT")
        provider.enqueue("The API endpoint moved to api-v2.example.com")

        cascade = _make_cascade(
            provider=provider,
            web_results=[{"title": "API Migration", "snippet": "v2 endpoint"}],
        )
        result = await cascade.try_due_diligence(SAMPLE_STEP, SAMPLE_ERROR)
        assert result is not None
        assert "api-v2" in result.lower() or "API" in result

    async def test_not_relevant(self):
        """When triage says NOT_RELEVANT, returns None."""
        provider = MockLLMProvider()
        provider.enqueue("NOT_RELEVANT")

        cascade = _make_cascade(
            provider=provider,
            web_results=[{"title": "Unrelated", "snippet": "cats"}],
        )
        result = await cascade.try_due_diligence(SAMPLE_STEP, SAMPLE_ERROR)
        assert result is None

    async def test_no_search_results(self):
        """When both searches return nothing, returns None without calling LLM."""
        provider = MockLLMProvider()
        cascade = _make_cascade(provider=provider)
        result = await cascade.try_due_diligence(SAMPLE_STEP, SAMPLE_ERROR)
        assert result is None
        # LLM should not have been called
        assert len(provider.complete_calls) == 0


class TestTryResearch:

    async def test_found_approach(self):
        provider = MockLLMProvider()
        provider.enqueue(
            '```json\n{"found": true, "approach": "Use API v2 endpoint", '
            '"sources": ["https://docs.example.com"], "clues": null, '
            '"concrete_blockers": []}\n```'
        )
        cascade = _make_cascade(provider=provider)
        result, approach = await cascade.try_research(
            SAMPLE_STEP, SAMPLE_ERROR, [],
        )
        assert result is not None
        assert result.found is True
        assert approach == "Use API v2 endpoint"

    async def test_not_found(self):
        provider = MockLLMProvider()
        provider.enqueue(
            '```json\n{"found": false, "approach": null, '
            '"sources": [], "clues": "Might need different auth", '
            '"concrete_blockers": ["Need OAuth2 token"]}\n```'
        )
        cascade = _make_cascade(provider=provider)
        result, approach = await cascade.try_research(
            SAMPLE_STEP, SAMPLE_ERROR, [],
        )
        assert result is not None
        assert result.found is False
        assert approach is None
        assert "OAuth2" in result.concrete_blockers[0]


class TestChallengeFailure:

    async def test_accept_verdict(self):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "verdict": "accept",
            "confirmed_blockers": ["API endpoint permanently removed"],
            "what_needs_to_change": "Need alternative data source",
        }))
        cascade = _make_cascade(provider=provider)

        research = ResearchResult(
            found=False,
            clues="Endpoint was deprecated",
            concrete_blockers=["API endpoint removed"],
        )
        result = await cascade.challenge_failure(
            SAMPLE_STEP, SAMPLE_ERROR, research, [],
        )
        assert result["verdict"] == "accept"
        assert "API endpoint" in result["confirmed_blockers"][0]

    async def test_reject_verdict_with_suggestion(self):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "verdict": "reject",
            "reason": "You haven't tried the v2 endpoint",
            "suggested_approach": "Try https://api-v2.example.com",
        }))
        cascade = _make_cascade(provider=provider)

        result = await cascade.challenge_failure(
            SAMPLE_STEP, SAMPLE_ERROR, None, [],
        )
        assert result["verdict"] == "reject"
        assert "suggested_approach" in result

    async def test_with_prior_rejections(self):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "verdict": "accept",
            "confirmed_blockers": ["Exhausted all options"],
        }))
        cascade = _make_cascade(provider=provider)

        prior = [
            {"reason": "Try v2", "suggested_approach": "Use v2"},
            {"reason": "Try cache", "suggested_approach": "Use cached data"},
        ]
        result = await cascade.challenge_failure(
            SAMPLE_STEP, SAMPLE_ERROR, None, prior,
        )
        assert result["verdict"] == "accept"


class TestRunRecoveryCascade:

    async def test_workaround_succeeds(self):
        """Layer 1 finds a workaround and retry succeeds."""
        provider = MockLLMProvider()
        cascade = _make_cascade(
            provider=provider,
            procedure_results=[
                {"approach": "Use fallback endpoint", "confidence": 0.9},
            ],
        )

        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        async def execute_step_fn(step, *, workaround=None):
            return StepResult(
                idx=0, status="completed", result="Fixed via workaround",
            )

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is not None
        assert recovered.status == "completed"
        assert reason == "workaround"

    async def test_due_diligence_succeeds(self):
        """Layer 2 finds context and retry succeeds."""
        provider = MockLLMProvider()
        # Due diligence triage response
        provider.enqueue("Use the new v2 API instead")

        cascade = _make_cascade(
            provider=provider,
            procedure_results=None,  # No workaround
            web_results=[{"title": "v2 API", "snippet": "new endpoint"}],
        )

        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        call_count = 0

        async def execute_step_fn(step, *, workaround=None):
            nonlocal call_count
            call_count += 1
            if workaround and "v2" in workaround.lower():
                return StepResult(idx=0, status="completed", result="Fixed via v2")
            return StepResult(idx=0, status="failed", result="Still broken")

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is not None
        assert reason == "due_diligence"

    async def test_all_fail_exit_gate_accepts(self):
        """All layers fail; exit gate accepts the failure."""
        provider = MockLLMProvider()
        # Due diligence: NOT_RELEVANT (no web/knowledge results, so no LLM call)
        # Research: not found
        provider.enqueue(
            '```json\n{"found": false, "approach": null, "sources": [], '
            '"clues": "Dead end", "concrete_blockers": ["API removed"]}\n```'
        )
        # Exit gate: accept
        provider.enqueue(json.dumps({
            "verdict": "accept",
            "confirmed_blockers": ["API permanently removed"],
        }))

        cascade = _make_cascade(provider=provider)

        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        async def execute_step_fn(step, *, workaround=None):
            return StepResult(idx=0, status="failed", result="Still fails")

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is None
        assert "exit gate accepted" in reason

    async def test_exit_gate_reject_then_succeed(self):
        """Exit gate rejects with a suggestion, retry succeeds."""
        provider = MockLLMProvider()
        # Research: not found
        provider.enqueue(
            '```json\n{"found": false, "approach": null, "sources": [], '
            '"clues": null, "concrete_blockers": []}\n```'
        )
        # Exit gate: reject with suggestion
        provider.enqueue(json.dumps({
            "verdict": "reject",
            "reason": "Try the backup API",
            "suggested_approach": "Use backup.api.example.com",
        }))

        cascade = _make_cascade(provider=provider)
        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        async def execute_step_fn(step, *, workaround=None):
            if workaround and "backup" in workaround.lower():
                return StepResult(idx=0, status="completed", result="Backup worked!")
            return StepResult(idx=0, status="failed", result="Nope")

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is not None
        assert "exit_gate_cycle" in reason

    async def test_exit_gate_cap(self):
        """Exit gate rejects MAX_EXIT_GATE_CYCLES times, then force-accepts."""
        provider = MockLLMProvider()
        # Research: not found
        provider.enqueue(
            '```json\n{"found": false, "approach": null, "sources": [], '
            '"clues": null, "concrete_blockers": []}\n```'
        )
        # Queue enough exit gate rejections to hit the cap
        for i in range(MAX_EXIT_GATE_CYCLES):
            provider.enqueue(json.dumps({
                "verdict": "reject",
                "reason": f"Try approach {i}",
                "suggested_approach": f"Approach {i}",
            }))

        cascade = _make_cascade(provider=provider)
        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        async def execute_step_fn(step, *, workaround=None):
            return StepResult(idx=0, status="failed", result="Still fails")

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is None
        assert "cap reached" in reason

    async def test_blocked_during_due_diligence_retry(self):
        """When retry returns blocked during due diligence, cascade returns None."""
        provider = MockLLMProvider()
        # Due diligence triage
        provider.enqueue("Try using API key authentication")

        cascade = _make_cascade(
            provider=provider,
            web_results=[{"title": "Auth docs", "snippet": "use api key"}],
        )
        failed_result = StepResult(idx=0, status="failed", result=SAMPLE_ERROR)

        async def execute_step_fn(step, *, workaround=None):
            return StepResult(
                idx=0, status="blocked", result="blocked",
                blocker_description="Need API key",
            )

        recovered, reason = await run_recovery_cascade(
            cascade, SAMPLE_STEP, failed_result,
            execute_step_fn=execute_step_fn,
            prior_step_results=[],
        )
        assert recovered is None
        assert "blocked" in reason

    async def test_max_exit_gate_cycles_constant(self):
        assert MAX_EXIT_GATE_CYCLES == 10
