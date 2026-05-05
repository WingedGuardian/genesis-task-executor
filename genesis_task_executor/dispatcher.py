"""Task dispatcher — submission, validation, and crash recovery.

Handles task lifecycle management outside the core engine:
- Task submission with plan validation
- Sequential execution via semaphore
- Crash recovery for non-terminal tasks
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiosqlite

from genesis_task_executor import db
from genesis_task_executor.engine import TaskExecutor
from genesis_task_executor.state_machine import is_resumable
from genesis_task_executor.types import TaskPhase

logger = logging.getLogger(__name__)

# Required sections in a task plan (when provided as structured text)
REQUIRED_SECTIONS = {"requirements", "steps", "success criteria"}


class TaskDispatcher:
    """Manages task submission and sequential execution."""

    def __init__(
        self,
        *,
        executor: TaskExecutor,
        database: aiosqlite.Connection,
    ) -> None:
        self._executor = executor
        self._db = database
        self._dispatched: set[str] = set()
        self._background_tasks: set[asyncio.Task] = set()

    async def submit(
        self,
        description: str,
        *,
        plan: dict | None = None,
    ) -> str:
        """Submit a task for execution.

        Args:
            description: Task description
            plan: Optional pre-built plan dict. If None, the engine
                will generate one via LLM.

        Returns:
            Task ID
        """
        if not description.strip():
            raise ValueError("Task description cannot be empty")

        plan_json = json.dumps(plan) if plan else None
        task_id = await db.create_task(
            self._db,
            description=description,
            plan_json=plan_json,
        )

        logger.info("Task %s submitted: %s", task_id, description[:100])

        # Dispatch execution (non-blocking, prevent GC collection)
        task = asyncio.create_task(self._guarded_execute(task_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return task_id

    async def submit_and_wait(
        self,
        description: str,
        *,
        plan: dict | None = None,
    ) -> dict:
        """Submit a task and wait for completion.

        Returns the final task record.
        """
        if not description.strip():
            raise ValueError("Task description cannot be empty")

        plan_json = json.dumps(plan) if plan else None
        task_id = await db.create_task(
            self._db,
            description=description,
            plan_json=plan_json,
        )

        logger.info("Task %s submitted (blocking): %s", task_id, description[:100])
        await self._guarded_execute(task_id)

        return await db.get_task(self._db, task_id) or {"task_id": task_id, "status": "unknown"}

    async def recover_incomplete(self) -> int:
        """Recover tasks that were interrupted (crash recovery).

        Returns the number of tasks recovered.
        """
        non_terminal = await db.get_non_terminal_tasks(self._db)
        recovered = 0

        for task in non_terminal:
            task_id = task["task_id"]
            phase = TaskPhase(task["current_phase"])

            if task_id in self._dispatched:
                continue

            if not is_resumable(phase):
                logger.info(
                    "Task %s in phase %s — not resumable, skipping",
                    task_id, phase.value,
                )
                continue

            logger.info("Recovering task %s from phase %s", task_id, phase.value)
            task = asyncio.create_task(self._guarded_execute(task_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            recovered += 1

        return recovered

    async def _guarded_execute(self, task_id: str) -> None:
        """Execute with semaphore for sequential ordering."""
        if task_id in self._dispatched:
            return
        self._dispatched.add(task_id)

        await self._executor._semaphore.acquire()
        try:
            await self._executor.execute(task_id)
        except Exception:
            logger.error("Task %s execution error", task_id, exc_info=True)
            try:
                await db.update_task_phase(
                    self._db, task_id, TaskPhase.FAILED.value,
                    reason="Unhandled execution error",
                )
            except Exception:
                logger.error("Failed to update task state", exc_info=True)
        finally:
            if task_id not in self._executor._semaphore_released:
                self._executor._semaphore.release()
            self._dispatched.discard(task_id)

    # --- Task control ---

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task."""
        if task_id in self._executor.get_active_tasks():
            self._executor.pause_task(task_id)
            return True
        return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        if task_id in self._executor.get_active_tasks():
            self._executor.resume_task(task_id)
            return True
        return False

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._executor.get_active_tasks():
            self._executor.cancel_task(task_id)
            return True
        return False
