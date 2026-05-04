"""CLI entry point for genesis-task-executor."""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    console = None  # type: ignore[assignment]
    HAS_RICH = False

DATA_DIR = Path.home() / ".task_executor"
DB_PATH = DATA_DIR / "tasks.db"


def pr(msg: str, style: str = "") -> None:
    if HAS_RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def section(title: str) -> None:
    if HAS_RICH and console:
        console.rule(f"[bold]{title}")
    else:
        print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def list_tasks_sync() -> None:
    """List recent tasks from the SQLite database."""
    if not DB_PATH.exists():
        pr("No tasks yet.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT task_id, current_phase, verdict, confidence, created_at, description "
        "FROM tasks ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    conn.close()

    if not rows:
        pr("No tasks.")
        return

    if HAS_RICH and console:
        t = Table(title="Recent Tasks", show_lines=True)
        for col in ("ID", "Phase", "Verdict", "Conf", "Created", "Description"):
            t.add_column(col)
        for r in rows:
            t.add_row(
                r[0][:8], r[1], r[2] or "—",
                f"{r[3]:.0%}" if r[3] is not None else "—",
                r[4][:16], r[5][:60],
            )
        console.print(t)
    else:
        for r in rows:
            print(
                f"{r[0][:8]} | {r[1]:12} | {r[2] or '—':8} | "
                f"{r[4][:16]} | {r[5][:60]}"
            )


def show_task_sync(prefix: str) -> None:
    """Show details for a specific task."""
    if not DB_PATH.exists():
        pr(f"No task matching '{prefix}'", "bold red")
        return

    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT * FROM tasks WHERE task_id LIKE ?", (prefix + "%",)
    ).fetchone()

    if not row:
        pr(f"No task matching '{prefix}'", "bold red")
        conn.close()
        return

    cols = [desc[0] for desc in conn.execute("SELECT * FROM tasks LIMIT 0").description]
    task = dict(zip(cols, row, strict=False))

    pr(f"\nTask: {task['task_id']}")
    pr(f"Phase: {task['current_phase']} | Verdict: {task.get('verdict', '—')}")
    pr(f"Description: {task['description']}")
    if task.get("reason"):
        pr(f"Reason: {task['reason']}")

    steps = conn.execute(
        "SELECT step_idx, step_type, description, status, result "
        "FROM steps WHERE task_id=? ORDER BY step_idx",
        (task["task_id"],),
    ).fetchall()

    if steps:
        pr("\nSteps:")
        for s in steps:
            icon = "✓" if s[3] == "completed" else ("✗" if s[3] == "failed" else "⏳")
            pr(f"  {icon} Step {s[0] + 1} [{s[1]}]: {s[2]} — {s[3]}", "dim")

    conn.close()


async def run_task_async(
    task: str,
    *,
    provider_name: str,
    model: str | None,
) -> None:
    """Run a task using the async executor."""
    from genesis_task_executor import create_executor

    section("Task")
    pr(task)

    system = await create_executor(
        provider=provider_name,
        model=model,
    )

    try:
        section("Executing")
        result = await system.dispatcher.submit_and_wait(task)

        section("Result")
        phase = result.get("current_phase", "unknown")
        verdict = result.get("verdict", "—")
        confidence = result.get("confidence")

        if phase == "completed":
            pr(f"COMPLETED — Verdict: {verdict}", "bold green")
        elif phase == "failed":
            pr(f"FAILED — {result.get('reason', 'unknown')}", "bold red")
        else:
            pr(f"Phase: {phase} — {result.get('reason', '')}")

        if confidence is not None:
            pr(f"Confidence: {confidence:.0%}")

        pr(f"\nTask ID: {result.get('task_id', 'unknown')}", "dim")
    finally:
        await system.close()


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Genesis Task Executor — autonomous LLM task execution",
    )
    p.add_argument("task", nargs="?", help="Task description to execute")
    p.add_argument("--list-tasks", action="store_true", help="List recent tasks")
    p.add_argument("--task-id", help="Show details for a task by ID prefix")
    p.add_argument(
        "--provider", default=os.environ.get("TASK_EXECUTOR_PROVIDER", "openai"),
        help="LLM provider (openai or anthropic)",
    )
    p.add_argument(
        "--model", default=None,
        help="Model name (provider-specific default if not set)",
    )
    args = p.parse_args()

    if args.list_tasks:
        list_tasks_sync()
        return

    if args.task_id:
        show_task_sync(args.task_id)
        return

    if not args.task:
        try:
            from rich.prompt import Prompt
            task = Prompt.ask("[bold]Describe the task[/bold]")
        except ImportError:
            task = input("Task: ").strip()
        if not task:
            sys.exit("No task provided.")
    else:
        task = args.task

    asyncio.run(run_task_async(task, provider_name=args.provider, model=args.model))


if __name__ == "__main__":
    main()
