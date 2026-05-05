"""Sandboxed tool implementations for task step execution.

Three tools: read_file, write_file, fetch_url. File tools are sandboxed
to a configurable directory. URL fetching blocks private/loopback ranges.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx

MAX_FETCH_BYTES = 50_000
FETCH_TIMEOUT = 15

# Default sandbox: ~/task_executor_workspace
# Set via set_sandbox() or TASK_EXECUTOR_SANDBOX env var
_sandbox_root: Path | None = None


def set_sandbox(path: str | Path | None) -> None:
    """Set the sandbox root for file tools. None disables sandboxing."""
    global _sandbox_root  # noqa: PLW0603
    _sandbox_root = Path(path).resolve() if path else None


def _resolve_sandboxed(path: str) -> Path:
    """Resolve a path within the sandbox. Raises ValueError if it escapes."""
    p = Path(path).expanduser()
    if _sandbox_root is not None:
        # If relative, resolve relative to sandbox
        if not p.is_absolute():
            p = _sandbox_root / p
        resolved = p.resolve()
        sandbox_resolved = _sandbox_root.resolve()
        sandbox_prefix = str(sandbox_resolved) + "/"
        if not str(resolved).startswith(sandbox_prefix) and resolved != sandbox_resolved:
            raise ValueError(
                f"Path escapes sandbox: {path} resolves to {resolved}, "
                f"sandbox is {sandbox_resolved}"
            )
        return resolved
    return p.resolve()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_read_file(path: str) -> str:
    """Read a local file within the sandbox."""
    try:
        p = _resolve_sandboxed(path)
    except ValueError as e:
        return f"ERROR: {e}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"ERROR: not found: {path}"
    except PermissionError:
        return f"ERROR: permission denied: {path}"


def tool_write_file(path: str, content: str) -> str:
    """Write or overwrite a file within the sandbox."""
    try:
        p = _resolve_sandboxed(path)
    except ValueError as e:
        return f"ERROR: {e}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except PermissionError:
        return f"ERROR: permission denied: {path}"


def _is_url_safe(url: str) -> tuple[bool, str]:
    """Check if a URL is safe to fetch (no SSRF)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname"

    # Resolve hostname and check for private/loopback IPs
    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Blocked private/loopback IP: {addr}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    return True, ""


async def tool_fetch_url(url: str) -> str:
    """HTTP GET a URL. Blocks private/loopback IPs to prevent SSRF."""
    safe, reason = _is_url_safe(url)
    if not safe:
        return f"ERROR: {reason}"
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
        path = arguments.get("path")
        if not path:
            return "ERROR: missing required argument 'path'"
        return tool_read_file(path)
    if name == "write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        if not path:
            return "ERROR: missing required argument 'path'"
        if content is None:
            return "ERROR: missing required argument 'content'"
        return tool_write_file(path, content)
    if name == "fetch_url":
        url = arguments.get("url")
        if not url:
            return "ERROR: missing required argument 'url'"
        return await tool_fetch_url(url)
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
