# genesis-task-executor

Autonomous LLM task executor with formal state machine, 4-layer failure recovery, and adversarial verification.

## Features

- **13-phase formal state machine** with validated transitions
- **4-layer failure recovery cascade**: procedural workaround → inline due diligence → deep research → adversarial exit gate
- **Dual-LLM adversarial verification**: fresh-eyes review + adversarial verification
- **Checkpoint pause/resume** with asyncio.Event + semaphore coordination
- **Crash recovery** — phase-aware resume from any non-terminal state
- **SQLite audit trail** — full task/step/tool_call recording
- **Sandboxed tools** — read_file, write_file, fetch_url only (no code execution)
- **Multi-provider** — OpenAI + Anthropic (any OpenAI-compatible API via base_url)

## Install

```bash
pip install genesis-task-executor[openai]    # OpenAI provider
pip install genesis-task-executor[anthropic] # Anthropic provider
pip install genesis-task-executor[all]       # Everything
```

## Quickstart

### CLI

```bash
export OPENAI_API_KEY=sk-...
genesis-task-executor "Research top 3 Python async frameworks, write comparison to /tmp/compare.md"
genesis-task-executor --list-tasks
genesis-task-executor --task-id abc123
```

### Python

```python
import asyncio
from genesis_task_executor import create_executor

async def main():
    system = await create_executor(provider="openai")
    result = await system.dispatcher.submit_and_wait(
        "Research top 3 Python async frameworks and write comparison to /tmp/compare.md"
    )
    print(f"Status: {result['current_phase']}")
    print(f"Verdict: {result.get('verdict', 'N/A')}")
    await system.close()

asyncio.run(main())
```

### Cross-vendor adversarial review

```python
system = await create_executor(
    provider="openai",
    model="gpt-4o",
    secondary_provider="anthropic",
    secondary_model="claude-sonnet-4-20250514",
)
```

## Architecture

```
Task submitted
    │
    ▼
┌─────────────┐
│  REVIEWING   │──→ Plan generated + reviewed
└──────┬──────┘
       ▼
┌─────────────┐
│  PLANNING    │──→ Steps decomposed
└──────┬──────┘
       ▼
┌─────────────┐     ┌──────────────────────────────────┐
│  EXECUTING   │──→  │ For each step:                    │
│              │     │  1. Execute with tools             │
│              │     │  2. On failure → Recovery cascade: │
│              │     │     L1: Procedural workaround      │
│              │     │     L2: Inline due diligence       │
│              │     │     L3: Deep research session      │
│              │     │     L4: Adversarial exit gate      │
└──────┬──────┘     └──────────────────────────────────┘
       ▼
┌─────────────┐     ┌──────────────────────────────────┐
│  VERIFYING   │──→  │ Dual-LLM gate:                    │
│              │     │  1. Programmatic checks            │
│              │     │  2. Fresh-eyes review              │
│              │     │  3. Adversarial verification       │
└──────┬──────┘     └──────────────────────────────────┘
       ▼
┌─────────────┐
│ SYNTHESIZING │──→ Deliverable compiled
└──────┬──────┘
       ▼
┌─────────────┐
│ DELIVERING   │──→ Results delivered
└──────┬──────┘
       ▼
┌──────────────┐
│RETROSPECTIVE │──→ Lessons extracted
└──────┬───────┘
       ▼
┌─────────────┐
│  COMPLETED   │
└─────────────┘
```

## State Machine

13 phases with validated transitions. Terminal states: COMPLETED, FAILED, CANCELLED.

Special states:
- **PAUSED** — releases execution semaphore so another task can run
- **BLOCKED** — needs external input; can resume to REVIEWING, EXECUTING, or VERIFYING

## Recovery Cascade

When a step fails, four recovery layers fire in sequence:

| Layer | Strategy | Speed | Requires |
|-------|----------|-------|----------|
| L1 | Procedural workaround | Instant | `ProcedureStore` adapter |
| L2 | Inline due diligence | Seconds | `WebSearcher` + `KnowledgeStore` |
| L3 | Deep research | Minutes | LLM provider only |
| L4 | Exit gate | Seconds | LLM provider only |

All adapters are optional — layers degrade gracefully via NoOp defaults.

## Extending

### Custom tools

```python
from genesis_task_executor.tools import TOOL_DEFINITIONS

# Add your own tool definition
TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Does something custom",
        "parameters": {...}
    }
})
```

### Custom recovery adapters

```python
class MyWebSearcher:
    async def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        # Your search implementation
        return [{"title": "...", "url": "...", "snippet": "..."}]

system = await create_executor(
    provider="openai",
    web_searcher=MyWebSearcher(),
)
```

## License

MIT
