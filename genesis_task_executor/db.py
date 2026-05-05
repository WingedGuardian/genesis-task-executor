"""SQLite database for task audit trail.

Stores tasks, steps, and tool calls with full execution history.
Uses aiosqlite for async access.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import aiosqlite

DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    current_phase TEXT NOT NULL DEFAULT 'pending',
    plan_json TEXT,
    blockers TEXT,
    outputs TEXT,
    verdict TEXT,
    confidence REAL,
    reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    step_idx INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    cost_usd REAL DEFAULT 0.0,
    model_used TEXT DEFAULT '',
    session_id TEXT,
    artifacts TEXT,
    blocker_description TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES steps(step_id),
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_text TEXT,
    called_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_step ON tool_calls(step_id);
CREATE INDEX IF NOT EXISTS idx_tasks_phase ON tasks(current_phase);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


async def init_schema(db: aiosqlite.Connection) -> None:
    """Create tables if they don't exist."""
    await db.executescript(DDL)
    await db.commit()


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------


async def create_task(
    db: aiosqlite.Connection,
    *,
    description: str,
    plan_json: str | None = None,
) -> str:
    """Insert a new task and return its ID."""
    task_id = _uuid()
    now = _now()
    await db.execute(
        "INSERT INTO tasks (task_id, description, plan_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, description, plan_json, now, now),
    )
    await db.commit()
    return task_id


async def get_task(db: aiosqlite.Connection, task_id: str) -> dict | None:
    """Fetch a task by ID."""
    cursor = await db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row, strict=False))


_ALLOWED_EXTRA_COLS = frozenset({
    "plan_json", "blockers", "outputs", "verdict",
    "confidence", "reason", "finished_at",
})


async def update_task_phase(
    db: aiosqlite.Connection,
    task_id: str,
    phase: str,
    **extra: str | float | None,
) -> None:
    """Update the current phase and any extra columns."""
    sets = ["current_phase = ?", "updated_at = ?"]
    vals: list = [phase, _now()]
    for k, v in extra.items():
        if k not in _ALLOWED_EXTRA_COLS:
            raise ValueError(f"Disallowed column name: {k!r}")
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(task_id)
    await db.execute(
        f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = ?",  # noqa: S608
        vals,
    )
    await db.commit()


async def list_tasks(
    db: aiosqlite.Connection,
    *,
    phase: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List tasks, optionally filtered by phase."""
    if phase:
        cursor = await db.execute(
            "SELECT * FROM tasks WHERE current_phase = ? ORDER BY created_at DESC LIMIT ?",
            (phase, limit),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    rows = await cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row, strict=False)) for row in rows]


async def get_non_terminal_tasks(db: aiosqlite.Connection) -> list[dict]:
    """Fetch all tasks not in a terminal state (for crash recovery)."""
    cursor = await db.execute(
        "SELECT * FROM tasks WHERE current_phase NOT IN ('completed', 'failed', 'cancelled')"
    )
    rows = await cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row, strict=False)) for row in rows]


# ---------------------------------------------------------------------------
# Step CRUD
# ---------------------------------------------------------------------------


async def create_step(
    db: aiosqlite.Connection,
    *,
    task_id: str,
    step_idx: int,
    step_type: str,
    description: str,
) -> str:
    """Insert a new step and return its ID."""
    step_id = _uuid()
    await db.execute(
        "INSERT INTO steps (step_id, task_id, step_idx, step_type, description, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (step_id, task_id, step_idx, step_type, description, _now()),
    )
    await db.commit()
    return step_id


async def update_step(
    db: aiosqlite.Connection,
    step_id: str,
    *,
    status: str,
    result: str | None = None,
    cost_usd: float = 0.0,
    model_used: str = "",
    artifacts: str | None = None,
    blocker_description: str | None = None,
) -> None:
    """Update step with execution result."""
    await db.execute(
        "UPDATE steps SET status=?, result=?, cost_usd=?, model_used=?, "
        "artifacts=?, blocker_description=?, finished_at=? WHERE step_id=?",
        (status, result, cost_usd, model_used, artifacts, blocker_description, _now(), step_id),
    )
    await db.commit()


async def get_steps(db: aiosqlite.Connection, task_id: str) -> list[dict]:
    """Fetch all steps for a task, ordered by index."""
    cursor = await db.execute(
        "SELECT * FROM steps WHERE task_id = ? ORDER BY step_idx", (task_id,)
    )
    rows = await cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row, strict=False)) for row in rows]


# ---------------------------------------------------------------------------
# Tool call recording
# ---------------------------------------------------------------------------


async def record_tool_call(
    db: aiosqlite.Connection,
    *,
    step_id: str,
    tool_name: str,
    args_json: str,
    result_text: str,
) -> str:
    """Record a tool call and return its ID."""
    call_id = _uuid()
    await db.execute(
        "INSERT INTO tool_calls (call_id, step_id, tool_name, args_json, result_text, called_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (call_id, step_id, tool_name, args_json, result_text, _now()),
    )
    await db.commit()
    return call_id
