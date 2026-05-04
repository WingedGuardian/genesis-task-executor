"""4-layer failure recovery cascade.

When a task step fails, the recovery cascade tries increasingly
thorough investigation strategies before accepting failure:

Layer 1: Procedural workaround — search learned procedures for a known fix
Layer 2: Inline due diligence — quick parallel web + knowledge search
Layer 3: Deep research — full LLM investigation session
Layer 4: Adversarial exit gate — challenges "is this really unsolvable?"

Each layer is optional and degrades gracefully via NoOp protocols.
"""

from __future__ import annotations

import logging

from genesis_task_executor.prompts import EXIT_GATE_TEMPLATE, render_template
from genesis_task_executor.protocols import (
    LLMProvider,
    NoOpProcedureStore,
    ProcedureStore,
)
from genesis_task_executor.research import Researcher
from genesis_task_executor.review import _parse_json_response
from genesis_task_executor.types import ResearchResult, StepResult, WorkaroundResult

logger = logging.getLogger(__name__)

MAX_EXIT_GATE_CYCLES = 10


class RecoveryCascade:
    """4-layer failure recovery for task step execution."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        researcher: Researcher,
        procedure_store: ProcedureStore | None = None,
    ) -> None:
        self._provider = provider
        self._researcher = researcher
        self._procedures = procedure_store or NoOpProcedureStore()

    async def try_workaround(
        self,
        step: dict,
        error: str,
        prior_attempts: list[str],
    ) -> WorkaroundResult | None:
        """Layer 1: Search procedural memory for a known workaround."""
        # Build context tags from step metadata
        tags = [
            step.get("type", ""),
            step.get("description", "")[:50],
        ]
        # Extract keywords from error
        error_words = [w for w in error.split()[:10] if len(w) > 3]
        tags.extend(error_words[:5])

        try:
            matches = await self._procedures.find_relevant(tags, limit=3)
        except Exception:
            logger.debug("Workaround search failed", exc_info=True)
            return None

        if not matches:
            return None

        # Return best match
        best = matches[0]
        approach = best.get("approach", best.get("steps", ""))
        if isinstance(approach, list):
            approach = "\n".join(f"- {s}" for s in approach)

        return WorkaroundResult(found=True, approach=str(approach))

    async def try_due_diligence(
        self,
        step: dict,
        error: str,
    ) -> str | None:
        """Layer 2: Quick parallel web + knowledge search with LLM triage."""
        return await self._researcher.inline_due_diligence(step, error)

    async def try_research(
        self,
        step: dict,
        error: str,
        prior_attempts: list[str],
        *,
        due_diligence_results: str | None = None,
    ) -> tuple[ResearchResult | None, str | None]:
        """Layer 3: Deep research session.

        Returns (research_result, approach_to_try) — approach is the
        actionable suggestion if research found something.
        """
        result = await self._researcher.research(
            step, error, prior_attempts,
            due_diligence_results=due_diligence_results,
        )
        if result is None:
            return None, None

        if result.found and result.approach:
            return result, result.approach
        return result, None

    async def challenge_failure(
        self,
        step: dict,
        error: str,
        research_result: ResearchResult | None,
        prior_rejections: list[dict],
    ) -> dict:
        """Layer 4: Adversarial exit gate — challenges the failure claim.

        Returns {verdict: "accept"|"reject", reason, suggested_approach, ...}
        """
        # Build context for the exit gate
        research_conclusion = "(no research performed)"
        concrete_blockers = "(none identified)"
        if research_result:
            if research_result.clues:
                research_conclusion = research_result.clues
            if research_result.concrete_blockers:
                concrete_blockers = "\n".join(
                    f"- {b}" for b in research_result.concrete_blockers
                )

        prior_text = ""
        if prior_rejections:
            parts = []
            for i, rej in enumerate(prior_rejections, 1):
                parts.append(
                    f"Attempt {i}: {rej.get('reason', 'no reason')}\n"
                    f"Suggested: {rej.get('suggested_approach', 'none')}"
                )
            prior_text = "\n\n".join(parts)

        prompt = render_template(
            EXIT_GATE_TEMPLATE,
            step_description=step.get("description", ""),
            error_text=error,
            research_conclusion=research_conclusion,
            concrete_blockers=concrete_blockers,
            prior_rejections=prior_text or "(first attempt)",
        )

        try:
            response = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                json_mode=True,
            )
            return _parse_json_response(
                response,
                default={"verdict": "accept", "confirmed_blockers": [error[:200]]},
            )
        except Exception:
            logger.warning("Exit gate failed", exc_info=True)
            return {"verdict": "accept", "confirmed_blockers": [error[:200]]}


async def run_recovery_cascade(
    cascade: RecoveryCascade,
    step: dict,
    result: StepResult,
    *,
    execute_step_fn,
    prior_step_results: list[StepResult],
) -> tuple[StepResult | None, str]:
    """Run the full 4-layer recovery cascade for a failed step.

    Args:
        cascade: The RecoveryCascade instance
        step: The step dict that failed
        result: The failed StepResult
        execute_step_fn: Async callable to retry the step with new context
        prior_step_results: Results from prior steps (for context)

    Returns:
        (recovered_result, reason) — recovered_result is None if all
        recovery failed; reason describes what happened.
    """
    error = result.result
    prior_attempts: list[str] = []

    # Layer 1: Procedural workaround
    workaround = await cascade.try_workaround(step, error, prior_attempts)
    if workaround and workaround.found and workaround.approach:
        retry = await execute_step_fn(step, workaround=workaround.approach)
        if retry.status == "completed":
            return retry, "workaround"
        prior_attempts.append(f"Workaround: {workaround.approach}")

    # Layer 2: Inline due diligence
    dd_context = await cascade.try_due_diligence(step, error)
    if dd_context:
        retry = await execute_step_fn(step, workaround=dd_context)
        if retry.status == "completed":
            return retry, "due_diligence"
        if retry.status == "blocked":
            return None, f"blocked during due diligence: {retry.blocker_description}"
        prior_attempts.append(f"Due diligence: {dd_context[:200]}")

    # Layer 3: Deep research
    research_result, approach = await cascade.try_research(
        step, error, prior_attempts,
        due_diligence_results=dd_context,
    )
    if approach:
        retry = await execute_step_fn(step, workaround=approach)
        if retry.status == "completed":
            return retry, "research"
        if retry.status == "blocked":
            return None, f"blocked during research retry: {retry.blocker_description}"
        prior_attempts.append(f"Research: {approach[:200]}")

    # Layer 4: Exit gate loop
    prior_rejections: list[dict] = []
    for cycle in range(MAX_EXIT_GATE_CYCLES):
        decision = await cascade.challenge_failure(
            step, error, research_result, prior_rejections,
        )

        if decision.get("verdict") == "accept":
            # Exit gate agrees: failure is genuine
            blockers = decision.get("confirmed_blockers", [error[:200]])
            return None, f"exit gate accepted: {', '.join(blockers)}"

        # Exit gate rejected — try its suggestion
        prior_rejections.append(decision)
        suggested = decision.get("suggested_approach", "")
        if suggested:
            retry = await execute_step_fn(step, workaround=suggested)
            if retry.status == "completed":
                return retry, f"exit_gate_cycle_{cycle + 1}"
            if retry.status == "blocked":
                return None, f"blocked during exit gate retry: {retry.blocker_description}"
            # Update error for next cycle
            error = retry.result

    # Hit cap — force accept
    logger.warning("Exit gate cap reached for step %s", step.get("idx", "?"))
    return None, "exit gate cap reached"
