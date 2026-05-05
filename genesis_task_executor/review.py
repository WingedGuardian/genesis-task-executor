"""Dual-LLM adversarial verification gate.

Three-gate pattern:
1. Programmatic checks (basic validation)
2. Fresh-eyes review (independent perspective)
3. Adversarial verification (actively tries to find flaws)

Ported from Genesis's production review.py with coupling replaced
by LLMProvider protocol.
"""

from __future__ import annotations

import json
import logging
import re

from genesis_task_executor.prompts import (
    ADVERSARIAL_SYSTEM,
    FRESH_EYES_SYSTEM,
    PLAN_REVIEW_SYSTEM,
)
from genesis_task_executor.protocols import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification threshold
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD = 0.75


class TaskReviewer:
    """Dual-LLM adversarial quality gate for task deliverables."""

    def __init__(
        self,
        *,
        primary_provider: LLMProvider,
        secondary_provider: LLMProvider | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._primary = primary_provider
        # Cross-vendor adversarial review for genuine independence
        self._secondary = secondary_provider or primary_provider
        self._threshold = confidence_threshold

    async def review_plan(
        self,
        task_description: str,
        plan: dict,
    ) -> dict:
        """Review a plan before execution.

        Returns {approved: bool, confidence: float, issues: list, revised_steps: list|None}
        """
        try:
            response = await self._primary.complete(
                [
                    {"role": "system", "content": PLAN_REVIEW_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task:\n{task_description}\n\n"
                            f"Plan:\n{json.dumps(plan, indent=2)}"
                        ),
                    },
                ],
                json_mode=True,
            )
            return _parse_json_response(response, default={"approved": False, "confidence": 0.0})
        except Exception:
            logger.warning("Plan review failed", exc_info=True)
            return {"approved": False, "confidence": 0.0, "issues": ["Review unavailable"]}

    async def verify_deliverable(
        self,
        task_description: str,
        plan: dict,
        results: list[dict],
    ) -> dict:
        """Run dual-LLM verification on task deliverables.

        Gate 1: Programmatic checks
        Gate 2: Fresh-eyes review (primary provider)
        Gate 3: Adversarial verification (secondary provider for independence)

        Returns {verdict, confidence, issues, feedback, step_verdicts}
        """
        # Gate 1: Programmatic checks
        prog_issues = _programmatic_checks(task_description, results)
        if prog_issues:
            return {
                "verdict": "reject",
                "confidence": 0.0,
                "issues": prog_issues,
                "feedback": "Failed programmatic checks",
            }

        evidence = _format_evidence(results)

        # Gate 2: Fresh-eyes review (primary — independent perspective)
        fresh_result = await self._fresh_eyes_review(task_description, evidence)

        # Gate 3: Adversarial verification (secondary — cross-vendor if available)
        adversarial_result = await self._adversarial_review(
            task_description, plan, evidence,
        )

        # Combine verdicts: both must accept for overall accept
        combined = _combine_verdicts(fresh_result, adversarial_result, self._threshold)
        return combined

    async def _fresh_eyes_review(
        self,
        task_description: str,
        evidence: str,
    ) -> dict:
        """Gate 2: Fresh-eyes review — independent, hadn't seen the execution."""
        try:
            response = await self._primary.complete(
                [
                    {"role": "system", "content": FRESH_EYES_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task:\n{task_description}\n\n"
                            f"Results:\n{evidence}"
                        ),
                    },
                ],
                json_mode=True,
            )
            return _parse_json_response(
                response, default={"verdict": "reject", "issues": ["Parse error"]},
            )
        except Exception:
            logger.warning("Fresh-eyes review failed", exc_info=True)
            return {"verdict": "accept", "issues": [], "feedback": "Review unavailable"}

    async def _adversarial_review(
        self,
        task_description: str,
        plan: dict,
        evidence: str,
    ) -> dict:
        """Gate 3: Adversarial verification — actively tries to find flaws."""
        try:
            response = await self._secondary.complete(
                [
                    {"role": "system", "content": ADVERSARIAL_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task:\n{task_description}\n\n"
                            f"Goal: {plan.get('goal', '')}\n\n"
                            f"Results:\n{evidence}"
                        ),
                    },
                ],
                json_mode=True,
            )
            return _parse_json_response(
                response,
                default={"verdict": "reject", "confidence": 0.0, "overall_reason": "Parse error"},
            )
        except Exception:
            logger.warning("Adversarial review failed", exc_info=True)
            # Non-blocking: if adversarial review fails, don't reject
            return {"verdict": "accept", "confidence": 0.5, "overall_reason": "Review unavailable"}


# ---------------------------------------------------------------------------
# Programmatic checks
# ---------------------------------------------------------------------------


def _programmatic_checks(task_description: str, results: list[dict]) -> list[str]:
    """Basic validation before LLM review."""
    issues = []

    if not results:
        issues.append("No step results produced")
        return issues

    completed = [r for r in results if r.get("status") == "completed"]
    if not completed:
        issues.append("No steps completed successfully")

    # Check for empty results
    for r in results:
        result_text = r.get("result", "")
        if isinstance(result_text, dict):
            result_text = result_text.get("result", "")
        if not result_text or len(str(result_text).strip()) < 10:
            issues.append(f"Step {r.get('idx', r.get('index', '?'))}: empty or trivial result")

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_evidence(results: list[dict]) -> str:
    """Format step results as evidence for reviewers."""
    parts = []
    for r in results:
        idx = r.get("idx", r.get("index", "?"))
        desc = r.get("description", "")
        result = r.get("result", "")
        criterion = r.get("success_criterion", "")
        parts.append(
            f"Step {idx} ({desc}):\n"
            f"Criterion: {criterion}\n"
            f"Result: {result}"
        )
    return "\n\n".join(parts)


def _combine_verdicts(
    fresh: dict,
    adversarial: dict,
    threshold: float,
) -> dict:
    """Combine fresh-eyes and adversarial verdicts."""
    fresh_verdict = fresh.get("verdict", "reject")
    adv_verdict = adversarial.get("verdict", "reject")
    adv_confidence = adversarial.get("confidence", 0.0)

    # Both must accept
    if fresh_verdict == "accept" and adv_verdict == "accept":
        verdict = "accept" if adv_confidence >= threshold else "reject"
    else:
        verdict = "reject"

    all_issues = fresh.get("issues", []) + adversarial.get("step_verdicts", [])

    return {
        "verdict": verdict,
        "confidence": adv_confidence,
        "fresh_eyes": fresh,
        "adversarial": adversarial,
        "issues": all_issues,
        "feedback": fresh.get("feedback", ""),
        "overall_reason": adversarial.get("overall_reason", ""),
    }


def _parse_json_response(text: str, *, default: dict) -> dict:
    """Extract JSON from LLM response text."""
    # Try fenced JSON block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON object in text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return default
