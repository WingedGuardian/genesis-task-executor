"""Task executor engine — the core 9-phase lifecycle.

Drives a task through: PENDING → REVIEWING → PLANNING → EXECUTING →
VERIFYING → SYNTHESIZING → DELIVERING → RETROSPECTIVE → COMPLETED

Implements:
- Formal state transitions via validate_transition()
- 4-layer failure recovery cascade
- Checkpoint pause/resume with asyncio.Event
- Review iteration cap (max 2)
- Crash recovery (phase-aware resume)

Ported from Genesis's production engine.py with all coupling replaced
by protocol adapters.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime

import aiosqlite

from genesis_task_executor import db
from genesis_task_executor.prompts import (
    PLAN_SYSTEM,
    RETROSPECTIVE_SYSTEM,
)
from genesis_task_executor.protocols import EventCallback, LLMProvider, NoOpEventCallback
from genesis_task_executor.recovery import RecoveryCascade, run_recovery_cascade
from genesis_task_executor.review import TaskReviewer
from genesis_task_executor.step_runner import StepRunner
from genesis_task_executor.types import (
    ExecutionTrace,
    InvalidTransitionError,
    StepResult,
    TaskPhase,
    validate_transition,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_ITERATIONS = 2
MAX_STEPS = 12


class TaskExecutor:
    """State machine driving autonomous multi-step task execution.

    Usage::

        executor = TaskExecutor(
            provider=my_llm_provider,
            database=my_db,
            reviewer=my_reviewer,
            recovery=my_cascade,
            step_runner=my_runner,
        )
        result = await executor.execute(task_id)
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        database: aiosqlite.Connection,
        reviewer: TaskReviewer,
        recovery: RecoveryCascade,
        step_runner: StepRunner,
        event_callback: EventCallback | None = None,
    ) -> None:
        self._provider = provider
        self._db = database
        self._reviewer = reviewer
        self._recovery = recovery
        self._runner = step_runner
        self._events = event_callback or NoOpEventCallback()

        # Pause/cancel control
        self._pause_event: dict[str, asyncio.Event] = {}
        self._cancel_event: dict[str, asyncio.Event] = {}
        self._active_tasks: set[str] = set()

        # Semaphore for sequential execution
        self._semaphore = asyncio.Semaphore(1)
        self._semaphore_released: set[str] = set()

    async def execute(self, task_id: str) -> bool:
        """Execute a task through the full lifecycle.

        Returns True if task completed successfully, False otherwise.
        """
        self._active_tasks.add(task_id)
        self._pause_event[task_id] = asyncio.Event()
        self._cancel_event[task_id] = asyncio.Event()

        try:
            return await self._run_lifecycle(task_id)
        finally:
            self._active_tasks.discard(task_id)
            self._pause_event.pop(task_id, None)
            self._cancel_event.pop(task_id, None)

    async def _run_lifecycle(self, task_id: str) -> bool:
        """Core lifecycle: review → plan → execute → verify → deliver."""
        task = await db.get_task(self._db, task_id)
        if task is None:
            logger.error("Task %s not found", task_id)
            return False

        description = task["description"]
        plan_json = task.get("plan_json")
        current_phase = TaskPhase(task["current_phase"])

        # Determine if this is a resume (skip review/plan)
        is_resume = current_phase in {
            TaskPhase.EXECUTING, TaskPhase.VERIFYING, TaskPhase.BLOCKED,
        }

        # Build execution trace
        trace = ExecutionTrace(
            task_id=task_id,
            initiated_by="user",
            user_request=description,
        )

        plan: dict | None = None
        if plan_json:
            with contextlib.suppress(json.JSONDecodeError):
                plan = json.loads(plan_json)

        # --- REVIEWING ---
        if not is_resume:
            await self._transition(task_id, TaskPhase.REVIEWING)

            if plan is None:
                # Generate plan
                plan = await self._generate_plan(description)
                if plan is None:
                    await self._fail_task(task_id, "Plan generation failed")
                    return False

                # Save plan to DB
                await db.update_task_phase(
                    self._db, task_id, TaskPhase.REVIEWING.value,
                    plan_json=json.dumps(plan),
                )

            # Review plan
            review = await self._reviewer.review_plan(description, plan)
            if not review.get("approved", False):
                issues = review.get("issues", ["Plan not approved"])
                reason = "; ".join(str(i) for i in issues)
                await self._fail_task(task_id, f"Plan rejected: {reason}")
                return False

            if review.get("revised_steps"):
                plan["steps"] = review["revised_steps"]
                await db.update_task_phase(
                    self._db, task_id, TaskPhase.REVIEWING.value,
                    plan_json=json.dumps(plan),
                )

        elif plan is None:
            await self._fail_task(task_id, "Resume failed: no plan found")
            return False

        if await self._is_cancelled(task_id):
            return False

        # --- PLANNING ---
        if not is_resume:
            await self._transition(task_id, TaskPhase.PLANNING)

        steps = plan.get("steps", [])
        if not steps:
            await self._fail_task(task_id, "Plan has no steps")
            return False

        trace.plan = [s.get("description", "") for s in steps]

        # On resume, load completed steps from DB
        step_results: list[StepResult] = []
        if is_resume:
            existing_steps = await db.get_steps(self._db, task_id)
            for es in existing_steps:
                if es["status"] == "completed":
                    step_results.append(StepResult(
                        idx=es["step_idx"],
                        status="completed",
                        result=es.get("result", ""),
                        cost_usd=es.get("cost_usd", 0.0),
                    ))
            logger.info(
                "Resuming task %s with %d/%d steps completed",
                task_id, len(step_results), len(steps),
            )

        completed_indices = {r.idx for r in step_results}

        # --- EXECUTING ---
        await self._transition(task_id, TaskPhase.EXECUTING)

        for step in steps:
            step_idx = step.get("idx", step.get("index", 0))

            # Skip already-completed steps on resume
            if step_idx in completed_indices:
                continue

            if await self._is_cancelled(task_id):
                return False

            # Check pause
            if await self._check_pause(task_id):
                return False  # Paused, will resume later

            # Execute step
            result = await self._runner.execute_step(
                task_id, step, step_results,
            )
            step_results.append(result)
            trace.step_results.append(result)
            self._events.on_event("step_complete", {
                "task_id": task_id, "step_idx": step_idx, "status": result.status,
            })

            if result.status == "completed":
                continue

            if result.status == "blocked":
                await self._block_task(
                    task_id,
                    result.blocker_description or "Step blocked",
                )
                return False

            # Handle failure — run 4-layer recovery cascade
            if result.status == "failed":
                async def execute_retry(s, *, workaround=None):
                    return await self._runner.execute_step(
                        task_id, s, step_results, workaround=workaround,
                    )

                recovered, reason = await run_recovery_cascade(
                    self._recovery,
                    step,
                    result,
                    execute_step_fn=execute_retry,
                    prior_step_results=step_results,
                )

                if recovered is not None:
                    step_results[-1] = recovered
                    trace.step_results[-1] = recovered
                    logger.info(
                        "Step %d recovered via %s", step_idx, reason,
                    )
                    continue

                # All recovery exhausted
                await self._fail_task(
                    task_id,
                    f"Step {step_idx} failed after recovery cascade: {reason}",
                )
                return False

        # --- VERIFYING (review loop) ---
        deliverable = _synthesize_deliverable(step_results)

        review_passed = False
        for iteration in range(MAX_REVIEW_ITERATIONS):
            await self._transition(task_id, TaskPhase.VERIFYING)

            verify = await self._reviewer.verify_deliverable(
                description,
                plan,
                [{"idx": r.idx, "result": r.result, "status": r.status,
                  "description": steps[r.idx]["description"] if r.idx < len(steps) else "",
                  "success_criterion": steps[r.idx].get("success_criterion", "")
                  if r.idx < len(steps) else ""}
                 for r in step_results],
            )

            trace.quality_gate = verify

            if verify.get("verdict") == "accept":
                review_passed = True
                break

            # Review failed — can we iterate?
            if iteration < MAX_REVIEW_ITERATIONS - 1:
                feedback = verify.get("feedback", "") or verify.get("overall_reason", "")
                if feedback:
                    # Add a fixup step
                    fixup_step = {
                        "idx": len(steps),
                        "type": "verification",
                        "description": f"Address review feedback: {feedback[:500]}",
                        "success_criterion": "All review issues resolved",
                    }
                    await self._transition(task_id, TaskPhase.EXECUTING)
                    fixup_result = await self._runner.execute_step(
                        task_id, fixup_step, step_results,
                    )
                    step_results.append(fixup_result)
                    trace.step_results.append(fixup_result)
            else:
                logger.warning("Review iteration cap reached for task %s", task_id)

        if not review_passed:
            await self._block_task(task_id, "Review cap reached — needs manual review")
            return False

        # --- SYNTHESIZING ---
        await self._transition(task_id, TaskPhase.SYNTHESIZING)
        await db.update_task_phase(
            self._db, task_id, TaskPhase.SYNTHESIZING.value,
            outputs=deliverable[:10000],
        )

        # --- DELIVERING ---
        await self._transition(task_id, TaskPhase.DELIVERING)
        self._events.on_event("task_delivering", {
            "task_id": task_id, "deliverable": deliverable[:1000],
        })

        # --- RETROSPECTIVE ---
        await self._transition(task_id, TaskPhase.RETROSPECTIVE)
        retro = await self._run_retrospective(trace)
        trace.retrospective_notes = retro

        # --- COMPLETED ---
        await self._transition(task_id, TaskPhase.COMPLETED)
        total_cost = sum(r.cost_usd for r in step_results)
        await db.update_task_phase(
            self._db, task_id, TaskPhase.COMPLETED.value,
            finished_at=datetime.now(UTC).isoformat(),
            verdict="accept",
            confidence=verify.get("confidence", 0.0) if review_passed else 0.0,
            reason=verify.get("overall_reason", ""),
        )

        self._events.on_event("task_completed", {
            "task_id": task_id, "cost_usd": total_cost,
        })

        logger.info("Task %s completed successfully (cost: $%.4f)", task_id, total_cost)
        return True

    # ----- State management -----

    async def _transition(self, task_id: str, to_phase: TaskPhase) -> None:
        """Validate and persist a state transition."""
        task = await db.get_task(self._db, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        from_phase = TaskPhase(task["current_phase"])

        # Allow self-transitions for EXECUTING (next step)
        if from_phase == to_phase == TaskPhase.EXECUTING:
            return

        validate_transition(from_phase, to_phase)
        await db.update_task_phase(self._db, task_id, to_phase.value)

        self._events.on_event("transition", {
            "task_id": task_id,
            "from": from_phase.value,
            "to": to_phase.value,
        })

        logger.debug("Task %s: %s → %s", task_id, from_phase.value, to_phase.value)

    async def _fail_task(self, task_id: str, reason: str) -> None:
        """Transition task to FAILED state."""
        with contextlib.suppress(InvalidTransitionError):
            await self._transition(task_id, TaskPhase.FAILED)
        await db.update_task_phase(
            self._db, task_id, TaskPhase.FAILED.value,
            reason=reason,
            finished_at=datetime.now(UTC).isoformat(),
        )
        self._events.on_event("task_failed", {"task_id": task_id, "reason": reason})
        logger.error("Task %s failed: %s", task_id, reason)

    async def _block_task(self, task_id: str, blocker: str) -> None:
        """Transition task to BLOCKED state."""
        with contextlib.suppress(InvalidTransitionError):
            await self._transition(task_id, TaskPhase.BLOCKED)
        await db.update_task_phase(
            self._db, task_id, TaskPhase.BLOCKED.value,
            blockers=blocker,
        )
        self._events.on_event("task_blocked", {"task_id": task_id, "blocker": blocker})
        logger.warning("Task %s blocked: %s", task_id, blocker)

    # ----- Pause/cancel -----

    async def _check_pause(self, task_id: str) -> bool:
        """Check if task should pause. Returns True if cancelled during pause."""
        evt = self._pause_event.get(task_id)
        if evt and evt.is_set():
            await self._transition(task_id, TaskPhase.PAUSED)
            # Release semaphore so another task can run
            self._semaphore.release()
            self._semaphore_released.add(task_id)
            try:
                # Wait for resume or cancel
                while evt.is_set():
                    cancel = self._cancel_event.get(task_id)
                    if cancel and cancel.is_set():
                        return True  # Cancelled during pause
                    await asyncio.sleep(1)
                # Re-acquire semaphore
                await self._semaphore.acquire()
                self._semaphore_released.discard(task_id)
            except Exception:
                # Ensure semaphore stays balanced on error
                if task_id in self._semaphore_released:
                    await self._semaphore.acquire()
                    self._semaphore_released.discard(task_id)
                raise
            await self._transition(task_id, TaskPhase.EXECUTING)
            return False  # Resumed, continue
        return False

    async def _is_cancelled(self, task_id: str) -> bool:
        """Check if task has been cancelled."""
        evt = self._cancel_event.get(task_id)
        if evt and evt.is_set():
            await self._fail_task(task_id, "Cancelled by user")
            return True
        return False

    def pause_task(self, task_id: str) -> None:
        """Signal a running task to pause."""
        evt = self._pause_event.get(task_id)
        if evt:
            evt.set()

    def resume_task(self, task_id: str) -> None:
        """Signal a paused task to resume."""
        evt = self._pause_event.get(task_id)
        if evt:
            evt.clear()

    def cancel_task(self, task_id: str) -> None:
        """Signal a running task to cancel."""
        evt = self._cancel_event.get(task_id)
        if evt:
            evt.set()

    def get_active_tasks(self) -> set[str]:
        """Return IDs of currently executing tasks."""
        return set(self._active_tasks)

    # ----- Plan generation -----

    async def _generate_plan(self, description: str) -> dict | None:
        """Generate an execution plan from a task description."""
        try:
            response = await self._provider.complete(
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user", "content": description},
                ],
                json_mode=True,
            )
            plan = json.loads(response)
            if not isinstance(plan.get("steps"), list):
                return None
            # Cap steps
            plan["steps"] = plan["steps"][:MAX_STEPS]
            # Ensure idx fields
            for i, step in enumerate(plan["steps"]):
                step.setdefault("idx", i)
            return plan
        except Exception:
            logger.warning("Plan generation failed", exc_info=True)
            return None

    # ----- Retrospective -----

    async def _run_retrospective(self, trace: ExecutionTrace) -> str:
        """Analyze execution for lessons learned."""
        summary_parts = []
        for r in trace.step_results:
            summary_parts.append(f"Step {r.idx} ({r.status}): {r.result[:300]}")

        try:
            response = await self._provider.complete(
                [
                    {"role": "system", "content": RETROSPECTIVE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task: {trace.user_request}\n\n"
                            f"Steps:\n" + "\n".join(summary_parts) + "\n\n"
                            f"Quality gate: {json.dumps(trace.quality_gate)}"
                        ),
                    },
                ],
                json_mode=True,
            )
            return response
        except Exception:
            logger.warning("Retrospective failed", exc_info=True)
            return ""


def _synthesize_deliverable(results: list[StepResult]) -> str:
    """Combine step results into a deliverable summary."""
    parts = []
    for r in results:
        if r.status == "completed":
            parts.append(f"## Step {r.idx}\n\n{r.result}")
            if r.artifacts:
                parts.append(f"Artifacts: {', '.join(r.artifacts)}")
    return "\n\n".join(parts) if parts else "(no completed steps)"
