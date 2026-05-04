"""Tests for genesis_task_executor.review module."""

from __future__ import annotations

import json

from genesis_task_executor.review import TaskReviewer, _programmatic_checks

from .conftest import MockLLMProvider


class TestReviewPlan:

    async def test_approved_plan(self):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "approved": True,
            "confidence": 0.9,
            "issues": [],
            "revised_steps": None,
        }))
        reviewer = TaskReviewer(primary_provider=provider)

        result = await reviewer.review_plan(
            "Write a summary",
            {"goal": "Summarize data", "steps": [{"idx": 0, "description": "Read data"}]},
        )
        assert result["approved"] is True
        assert result["confidence"] == 0.9

    async def test_rejected_plan(self):
        provider = MockLLMProvider()
        provider.enqueue(json.dumps({
            "approved": False,
            "confidence": 0.2,
            "issues": ["Steps are too vague", "Missing verification"],
            "revised_steps": None,
        }))
        reviewer = TaskReviewer(primary_provider=provider)

        result = await reviewer.review_plan("Vague task", {"goal": "?", "steps": []})
        assert result["approved"] is False
        assert len(result["issues"]) == 2

    async def test_plan_review_with_revised_steps(self):
        provider = MockLLMProvider()
        revised = [{"idx": 0, "description": "Better step"}]
        provider.enqueue(json.dumps({
            "approved": True,
            "confidence": 0.85,
            "issues": [],
            "revised_steps": revised,
        }))
        reviewer = TaskReviewer(primary_provider=provider)

        result = await reviewer.review_plan("Task", {"steps": [{"idx": 0}]})
        assert result["revised_steps"] == revised

    async def test_plan_review_error_returns_rejected(self):
        """When the provider raises, review_plan returns a safe rejected result."""
        provider = MockLLMProvider()

        async def raise_error(*a, **kw):
            raise RuntimeError("API down")

        provider.complete = raise_error
        reviewer = TaskReviewer(primary_provider=provider)

        result = await reviewer.review_plan("Task", {"steps": []})
        assert result["approved"] is False


class TestVerifyDeliverable:

    def _make_results(self, status="completed", result="Done successfully and verified"):
        return [
            {"idx": 0, "status": status, "result": result, "description": "Step 0"},
        ]

    async def test_both_gates_accept(self):
        primary = MockLLMProvider()
        secondary = MockLLMProvider()

        # Fresh-eyes (primary) accepts
        primary.enqueue(json.dumps({
            "verdict": "accept",
            "issues": [],
            "feedback": "Looks good",
        }))
        # Adversarial (secondary) accepts with high confidence
        secondary.enqueue(json.dumps({
            "verdict": "accept",
            "confidence": 0.9,
            "step_verdicts": [],
            "overall_reason": "All criteria met",
        }))

        reviewer = TaskReviewer(
            primary_provider=primary,
            secondary_provider=secondary,
            confidence_threshold=0.75,
        )
        result = await reviewer.verify_deliverable(
            "Write summary",
            {"goal": "Summarize"},
            self._make_results(),
        )
        assert result["verdict"] == "accept"

    async def test_fresh_eyes_rejects(self):
        primary = MockLLMProvider()
        secondary = MockLLMProvider()

        # Fresh-eyes rejects
        primary.enqueue(json.dumps({
            "verdict": "reject",
            "issues": ["Missing section 3"],
            "feedback": "Incomplete work",
        }))
        # Adversarial accepts (but doesn't matter)
        secondary.enqueue(json.dumps({
            "verdict": "accept",
            "confidence": 0.9,
            "step_verdicts": [],
            "overall_reason": "Looks fine to me",
        }))

        reviewer = TaskReviewer(
            primary_provider=primary,
            secondary_provider=secondary,
        )
        result = await reviewer.verify_deliverable(
            "Write full report",
            {"goal": "Full report"},
            self._make_results(),
        )
        assert result["verdict"] == "reject"

    async def test_adversarial_rejects(self):
        primary = MockLLMProvider()
        secondary = MockLLMProvider()

        # Fresh-eyes accepts
        primary.enqueue(json.dumps({
            "verdict": "accept",
            "issues": [],
            "feedback": "OK",
        }))
        # Adversarial rejects
        secondary.enqueue(json.dumps({
            "verdict": "reject",
            "confidence": 0.3,
            "step_verdicts": [{"idx": 0, "passed": False, "note": "No evidence"}],
            "overall_reason": "Insufficient evidence",
        }))

        reviewer = TaskReviewer(
            primary_provider=primary,
            secondary_provider=secondary,
        )
        result = await reviewer.verify_deliverable(
            "Research task",
            {"goal": "Research"},
            self._make_results(),
        )
        assert result["verdict"] == "reject"

    async def test_confidence_below_threshold_rejects(self):
        """Even if both accept, confidence below threshold means reject."""
        primary = MockLLMProvider()
        secondary = MockLLMProvider()

        primary.enqueue(json.dumps({
            "verdict": "accept", "issues": [], "feedback": "OK",
        }))
        secondary.enqueue(json.dumps({
            "verdict": "accept",
            "confidence": 0.5,  # Below 0.75 threshold
            "step_verdicts": [],
            "overall_reason": "Marginal",
        }))

        reviewer = TaskReviewer(
            primary_provider=primary,
            secondary_provider=secondary,
            confidence_threshold=0.75,
        )
        result = await reviewer.verify_deliverable(
            "Task", {"goal": "G"}, self._make_results(),
        )
        assert result["verdict"] == "reject"

    async def test_single_provider_used_for_both_gates(self):
        """When no secondary is provided, primary is used for both."""
        provider = MockLLMProvider()
        # First call: fresh-eyes
        provider.enqueue(json.dumps({
            "verdict": "accept", "issues": [], "feedback": "OK",
        }))
        # Second call: adversarial
        provider.enqueue(json.dumps({
            "verdict": "accept", "confidence": 0.9,
            "step_verdicts": [], "overall_reason": "Good",
        }))

        reviewer = TaskReviewer(primary_provider=provider)
        result = await reviewer.verify_deliverable(
            "Task", {"goal": "G"}, self._make_results(),
        )
        assert result["verdict"] == "accept"
        # Provider was called twice
        assert len(provider.complete_calls) == 2


class TestProgrammaticChecks:

    def test_empty_results_rejected(self):
        issues = _programmatic_checks("Task", [])
        assert any("No step results" in i for i in issues)

    def test_trivial_result_flagged(self):
        issues = _programmatic_checks("Task", [
            {"idx": 0, "status": "completed", "result": "ok"},  # < 10 chars
        ])
        assert any("empty or trivial" in i for i in issues)

    def test_valid_results_pass(self):
        issues = _programmatic_checks("Task", [
            {"idx": 0, "status": "completed", "result": "A" * 50},
        ])
        # Should have no "No step results" issue
        assert not any("No step results" in i for i in issues)
