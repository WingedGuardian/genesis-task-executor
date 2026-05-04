"""Tests for genesis_task_executor.db module.

All tests use an in-memory aiosqlite database via the temp_db fixture.
"""

from __future__ import annotations

from genesis_task_executor.db import (
    create_step,
    create_task,
    get_non_terminal_tasks,
    get_steps,
    get_task,
    init_schema,
    list_tasks,
    record_tool_call,
    update_step,
    update_task_phase,
)


class TestInitSchema:

    async def test_creates_tables(self, temp_db):
        """init_schema creates tasks, steps, and tool_calls tables."""
        cursor = await temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert "tasks" in tables
        assert "steps" in tables
        assert "tool_calls" in tables

    async def test_creates_indexes(self, temp_db):
        cursor = await temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = [row[0] for row in await cursor.fetchall()]
        assert "idx_steps_task" in indexes
        assert "idx_tool_calls_step" in indexes
        assert "idx_tasks_phase" in indexes

    async def test_idempotent(self, temp_db):
        """Calling init_schema twice does not raise."""
        await init_schema(temp_db)  # Already called by fixture; call again


class TestCreateAndGetTask:

    async def test_create_returns_uuid(self, temp_db):
        task_id = await create_task(temp_db, description="Test task")
        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID format

    async def test_get_task_returns_dict(self, temp_db):
        task_id = await create_task(temp_db, description="Hello world")
        task = await get_task(temp_db, task_id)
        assert task is not None
        assert task["task_id"] == task_id
        assert task["description"] == "Hello world"
        assert task["current_phase"] == "pending"

    async def test_get_task_with_plan(self, temp_db):
        task_id = await create_task(
            temp_db, description="With plan", plan_json='{"steps": []}'
        )
        task = await get_task(temp_db, task_id)
        assert task["plan_json"] == '{"steps": []}'

    async def test_get_nonexistent_task(self, temp_db):
        result = await get_task(temp_db, "nonexistent-id")
        assert result is None


class TestUpdateTaskPhase:

    async def test_update_phase(self, temp_db):
        task_id = await create_task(temp_db, description="Phase test")
        await update_task_phase(temp_db, task_id, "reviewing")
        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "reviewing"

    async def test_update_with_extra_columns(self, temp_db):
        task_id = await create_task(temp_db, description="Extra cols")
        await update_task_phase(
            temp_db, task_id, "failed",
            reason="Something broke",
            confidence=0.1,
        )
        task = await get_task(temp_db, task_id)
        assert task["current_phase"] == "failed"
        assert task["reason"] == "Something broke"
        assert task["confidence"] == 0.1


class TestListTasks:

    async def test_list_all(self, temp_db):
        await create_task(temp_db, description="Task 1")
        await create_task(temp_db, description="Task 2")
        tasks = await list_tasks(temp_db)
        assert len(tasks) == 2

    async def test_list_with_phase_filter(self, temp_db):
        t1 = await create_task(temp_db, description="Pending")
        t2 = await create_task(temp_db, description="Will be reviewing")
        await update_task_phase(temp_db, t2, "reviewing")

        pending = await list_tasks(temp_db, phase="pending")
        assert len(pending) == 1
        assert pending[0]["task_id"] == t1

        reviewing = await list_tasks(temp_db, phase="reviewing")
        assert len(reviewing) == 1
        assert reviewing[0]["task_id"] == t2

    async def test_list_with_limit(self, temp_db):
        for i in range(5):
            await create_task(temp_db, description=f"Task {i}")
        tasks = await list_tasks(temp_db, limit=3)
        assert len(tasks) == 3

    async def test_list_empty(self, temp_db):
        tasks = await list_tasks(temp_db)
        assert tasks == []


class TestGetNonTerminalTasks:

    async def test_excludes_terminal(self, temp_db):
        t1 = await create_task(temp_db, description="Active")
        t2 = await create_task(temp_db, description="Done")
        t3 = await create_task(temp_db, description="Failed")
        await update_task_phase(temp_db, t2, "completed")
        await update_task_phase(temp_db, t3, "failed")

        non_terminal = await get_non_terminal_tasks(temp_db)
        ids = [t["task_id"] for t in non_terminal]
        assert t1 in ids
        assert t2 not in ids
        assert t3 not in ids

    async def test_empty_when_all_terminal(self, temp_db):
        t1 = await create_task(temp_db, description="Done")
        await update_task_phase(temp_db, t1, "completed")
        assert await get_non_terminal_tasks(temp_db) == []


class TestSteps:

    async def test_create_and_get_steps(self, temp_db):
        task_id = await create_task(temp_db, description="Step test")
        step_id = await create_step(
            temp_db,
            task_id=task_id,
            step_idx=0,
            step_type="analysis",
            description="Analyze data",
        )
        assert isinstance(step_id, str)

        steps = await get_steps(temp_db, task_id)
        assert len(steps) == 1
        assert steps[0]["step_id"] == step_id
        assert steps[0]["step_idx"] == 0
        assert steps[0]["description"] == "Analyze data"
        assert steps[0]["status"] == "pending"

    async def test_steps_ordered_by_idx(self, temp_db):
        task_id = await create_task(temp_db, description="Order test")
        await create_step(temp_db, task_id=task_id, step_idx=2, step_type="code", description="C")
        await create_step(
            temp_db, task_id=task_id, step_idx=0, step_type="research", description="A",
        )
        await create_step(
            temp_db, task_id=task_id, step_idx=1, step_type="analysis", description="B",
        )

        steps = await get_steps(temp_db, task_id)
        assert [s["step_idx"] for s in steps] == [0, 1, 2]

    async def test_update_step(self, temp_db):
        task_id = await create_task(temp_db, description="Update test")
        step_id = await create_step(
            temp_db, task_id=task_id, step_idx=0, step_type="code", description="Write code",
        )
        await update_step(
            temp_db, step_id,
            status="completed",
            result="Wrote 50 lines of code",
            cost_usd=0.02,
            model_used="gpt-4o",
            artifacts='["/tmp/output.py"]',
        )
        steps = await get_steps(temp_db, task_id)
        s = steps[0]
        assert s["status"] == "completed"
        assert s["result"] == "Wrote 50 lines of code"
        assert s["cost_usd"] == 0.02
        assert s["model_used"] == "gpt-4o"
        assert s["finished_at"] is not None


class TestRecordToolCall:

    async def test_record_and_verify(self, temp_db):
        task_id = await create_task(temp_db, description="Tool test")
        step_id = await create_step(
            temp_db, task_id=task_id, step_idx=0, step_type="code", description="Step",
        )
        call_id = await record_tool_call(
            temp_db,
            step_id=step_id,
            tool_name="read_file",
            args_json='{"path": "/tmp/test.txt"}',
            result_text="file contents here",
        )
        assert isinstance(call_id, str)
        assert len(call_id) == 36

        # Verify the tool call exists in DB
        cursor = await temp_db.execute(
            "SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        cols = [desc[0] for desc in cursor.description]
        data = dict(zip(cols, row, strict=False))
        assert data["tool_name"] == "read_file"
        assert data["step_id"] == step_id
