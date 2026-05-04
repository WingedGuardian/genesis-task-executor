"""Genesis Task Executor — autonomous LLM task execution with adversarial verification.

Quickstart::

    import asyncio
    from genesis_task_executor import create_executor

    async def main():
        system = await create_executor(provider="openai")
        result = await system.dispatcher.submit_and_wait(
            "Research top 3 Python async frameworks and write comparison to /tmp/compare.md"
        )
        print(result)
        await system.close()

    asyncio.run(main())
"""

from __future__ import annotations

from genesis_task_executor.types import (
    ExecutionTrace,
    ResearchResult,
    StepResult,
    StepType,
    TaskPhase,
    ToolCall,
    ToolResponse,
    WorkaroundResult,
)

__version__ = "1.0.0"
__all__ = [
    "create_executor",
    "ExecutorSystem",
    "TaskPhase",
    "StepType",
    "StepResult",
    "ExecutionTrace",
    "WorkaroundResult",
    "ResearchResult",
    "ToolCall",
    "ToolResponse",
]


from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from genesis_task_executor.dispatcher import TaskDispatcher
from genesis_task_executor.engine import TaskExecutor
from genesis_task_executor.recovery import RecoveryCascade
from genesis_task_executor.research import Researcher
from genesis_task_executor.review import TaskReviewer
from genesis_task_executor.step_runner import StepRunner


@dataclass
class ExecutorSystem:
    """Fully wired executor system."""

    executor: TaskExecutor
    dispatcher: TaskDispatcher
    db: aiosqlite.Connection

    async def close(self) -> None:
        """Clean up resources."""
        await self.db.close()


async def create_executor(
    *,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    secondary_provider: str | None = None,
    secondary_api_key: str | None = None,
    secondary_model: str | None = None,
    db_path: str | Path | None = None,
    web_searcher=None,
    knowledge_store=None,
    procedure_store=None,
    event_callback=None,
    confidence_threshold: float = 0.75,
) -> ExecutorSystem:
    """Create a fully wired executor system.

    Args:
        provider: Primary LLM provider ("openai" or "anthropic")
        api_key: API key for primary provider
        model: Model name for primary provider
        base_url: Base URL for OpenAI-compatible APIs
        secondary_provider: Optional cross-vendor provider for adversarial review
        secondary_api_key: API key for secondary provider
        secondary_model: Model name for secondary provider
        db_path: SQLite database path (default: ~/.task_executor/tasks.db)
        web_searcher: Optional WebSearcher for research layer
        knowledge_store: Optional KnowledgeStore for due diligence
        procedure_store: Optional ProcedureStore for workaround layer
        event_callback: Optional EventCallback for observability
        confidence_threshold: Minimum confidence to accept (default: 0.75)
    """
    from genesis_task_executor import db as db_module
    from genesis_task_executor.providers import get_provider

    # Resolve DB path
    if db_path is None:
        data_dir = Path.home() / ".task_executor"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "tasks.db"

    # Connect to DB
    database = await aiosqlite.connect(str(db_path))
    try:
        database.row_factory = aiosqlite.Row
        await db_module.init_schema(database)

        # Build primary provider
        primary = get_provider(provider, api_key=api_key, model=model, base_url=base_url)

        # Build secondary provider (for cross-vendor adversarial review)
        secondary = None
        if secondary_provider:
            secondary = get_provider(
                secondary_provider,
                api_key=secondary_api_key,
                model=secondary_model,
            )

        # Build components
        researcher = Researcher(
            provider=primary,
            web_searcher=web_searcher,
            knowledge_store=knowledge_store,
        )

        reviewer = TaskReviewer(
            primary_provider=primary,
            secondary_provider=secondary,
            confidence_threshold=confidence_threshold,
        )

        recovery = RecoveryCascade(
            provider=primary,
            researcher=researcher,
            procedure_store=procedure_store,
        )

        runner = StepRunner(
            provider=primary,
            database=database,
        )

        executor = TaskExecutor(
            provider=primary,
            database=database,
            reviewer=reviewer,
            recovery=recovery,
            step_runner=runner,
            event_callback=event_callback,
        )

        dispatcher = TaskDispatcher(
            executor=executor,
            database=database,
        )
    except Exception:
        await database.close()
        raise

    return ExecutorSystem(
        executor=executor,
        dispatcher=dispatcher,
        db=database,
    )
