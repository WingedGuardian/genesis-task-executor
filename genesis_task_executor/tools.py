"""Sandboxed tool implementations for task step execution.

Three tools: read_file, write_file, fetch_url. No code execution,
no subprocess calls — safe by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

MAX_FETCH_BYTES = 50_000
FETCH_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_read_file(path: str) -> str:
    """Read a local file and return its contents."""
    p = Path(path).expanduser()
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"ERROR: not found: {path}"
    except PermissionError:
        return f"ERROR: permission denied: {path}"


def tool_write_file(path: str, content: str) -> str:
    """Write or overwrite a local file."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except PermissionError:
        return f"ERROR: permission denied: {path}"


async def tool_fetch_url(url: str) -> str:
    """HTTP GET a URL and return its text content."""
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True,
        ) as client:
            r = await client.get(url, headers={"User-Agent": "genesis-task-executor/1.0"})
            r.raise_for_status()
            text = r.text[:MAX_FETCH_BYTES]
            return text + ("\n[TRUNCATED]" if len(r.text) > MAX_FETCH_BYTES else "")
    except httpx.HTTPStatusError as e:
        return f"ERROR: HTTP {e.response.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool registry (for LLM function calling)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local file and return its contents.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "description": "File path to read"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a local file.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "HTTP GET a URL. Returns text content (max 50 KB).",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string", "description": "URL to fetch"}},
            },
        },
    },
]


async def dispatch_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name and return the result."""
    if name == "read_file":
        return tool_read_file(arguments["path"])
    if name == "write_file":
        return tool_write_file(arguments["path"], arguments["content"])
    if name == "fetch_url":
        return await tool_fetch_url(arguments["url"])
    return f"ERROR: unknown tool '{name}'"


def parse_tool_arguments(raw: str | dict) -> dict:
    """Parse tool arguments from string or dict."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        # Some models wrap args in an array
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {}
