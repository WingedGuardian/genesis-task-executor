"""Step execution with LLM tool use.

Runs individual task steps by dispatching them to the LLM provider
with sandboxed tools (read_file, write_file, fetch_url).
"""

from __future__ import annotations

import json
import logging
import re
import time

import aiosqlite

from genesis_task_executor import db
from genesis_task_executor.prompts import STEP_EXECUTION_SYSTEM
from genesis_task_executor.protocols import LLMProvider
from genesis_task_executor.tools import TOOL_DEFINITIONS, dispatch_tool, parse_tool_arguments
from genesis_task_executor.types import StepResult

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8


class StepRunner:
    """Execute task steps via LLM with tool use."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        database: aiosqlite.Connection,
    ) -> None:
        self._provider = provider
        self._db = database

    async def execute_step(
        self,
        task_id: str,
        step: dict,
        prior_results: list[StepResult],
        *,
        workaround: str | None = None,
    ) -> StepResult:
        """Execute a single step and return the result.

        Args:
            task_id: Parent task ID
            step: Step dict with idx, type, description, success_criterion
            prior_results: Results from prior steps (for context)
            workaround: Optional workaround/context from recovery cascade
        """
        step_idx = step.get("idx", step.get("index", 0))
        step_type = step.get("type", "analysis")
        description = step.get("description", "")
        criterion = step.get("success_criterion", "complete as described")

        # Record step start
        step_id = await db.create_step(
            self._db,
            task_id=task_id,
            step_idx=step_idx,
            step_type=step_type,
            description=description,
        )

        start = time.monotonic()

        # Build context from prior results
        context_parts = []
        for r in prior_results[-5:]:  # Last 5 for context window
            context_parts.append(f"Step {r.idx}: {r.result[:500]}")
        context = "\n".join(context_parts) if context_parts else "(none)"

        # Build user message
        user_msg = (
            f"Step {step_idx + 1}: {description}\n"
            f"Success criterion: {criterion}\n"
            f"Prior context:\n{context}"
        )
        if workaround:
            user_msg += f"\n\nAdditional context/workaround:\n{workaround}"

        # Execute with tool use loop
        messages: list[dict] = [
            {"role": "system", "content": STEP_EXECUTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        all_evidence: list[str] = []

        for _round in range(MAX_TOOL_ROUNDS):
            response = await self._provider.complete_with_tools(
                messages, TOOL_DEFINITIONS,
            )

            if not response.tool_calls:
                # No more tool calls — LLM is done
                narrative = response.content or "(no output)"
                break

            # Process tool calls
            # Add assistant message with tool calls for conversation history
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                args = parse_tool_arguments(tc.arguments)
                tool_result = await dispatch_tool(tc.name, args)

                evidence = (
                    f"[{tc.name}({json.dumps(args)[:120]})] → {tool_result[:200]}"
                )
                all_evidence.append(evidence)

                # Record tool call in DB
                await db.record_tool_call(
                    self._db,
                    step_id=step_id,
                    tool_name=tc.name,
                    args_json=json.dumps(args),
                    result_text=tool_result[:4000],
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result[:8000],
                })
        else:
            narrative = "(step hit tool-use limit)"

        duration = time.monotonic() - start

        # Combine evidence and narrative
        if all_evidence:
            full_result = (
                "Tool calls:\n" + "\n".join(all_evidence) + "\n\n" + narrative
            )
        else:
            full_result = narrative

        # Parse the step output
        parsed = _parse_step_output(full_result)

        step_result = StepResult(
            idx=step_idx,
            status=parsed.get("status", "completed"),
            result=parsed.get("result", full_result[:4000]),
            duration_s=duration,
            artifacts=parsed.get("artifacts", []),
            blocker_description=parsed.get("blocker_description"),
        )

        # Update step in DB
        await db.update_step(
            self._db,
            step_id,
            status=step_result.status,
            result=step_result.result[:4000],
            cost_usd=step_result.cost_usd,
            model_used=step_result.model_used,
            artifacts=json.dumps(step_result.artifacts) if step_result.artifacts else None,
            blocker_description=step_result.blocker_description,
        )

        return step_result


def _parse_step_output(text: str) -> dict:
    """Extract structured output from step execution response."""
    # Try to find JSON block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
            if "status" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Fallback: infer from text
    text_lower = text.lower()
    if "error" in text_lower or "failed" in text_lower:
        return {"status": "failed", "result": text[:4000]}
    return {"status": "completed", "result": text[:4000]}
