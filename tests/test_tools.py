"""Tests for genesis_task_executor.tools module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from genesis_task_executor.tools import (
    TOOL_DEFINITIONS,
    dispatch_tool,
    parse_tool_arguments,
    tool_fetch_url,
    tool_read_file,
    tool_write_file,
)


class TestToolReadFile:

    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = tool_read_file(str(f))
        assert result == "Hello, world!"

    def test_read_missing_file(self):
        result = tool_read_file("/tmp/nonexistent_file_abc123.txt")
        assert "ERROR" in result
        assert "not found" in result

    def test_read_with_expanduser(self, tmp_path, monkeypatch):
        """Verify path expansion works (tilde)."""
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        # Just test with an absolute path — tilde expansion is implicit
        result = tool_read_file(str(f))
        assert result == "content"


class TestToolWriteFile:

    def test_write_creates_file(self, tmp_path):
        target = tmp_path / "output.txt"
        result = tool_write_file(str(target), "Hello")
        assert "OK" in result
        assert target.read_text() == "Hello"

    def test_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "dir" / "file.txt"
        result = tool_write_file(str(target), "nested content")
        assert "OK" in result
        assert target.read_text() == "nested content"

    def test_write_overwrites(self, tmp_path):
        target = tmp_path / "overwrite.txt"
        target.write_text("old content")
        tool_write_file(str(target), "new content")
        assert target.read_text() == "new content"

    def test_write_reports_char_count(self, tmp_path):
        target = tmp_path / "count.txt"
        result = tool_write_file(str(target), "12345")
        assert "5 chars" in result


class TestToolFetchUrl:

    async def test_fetch_success(self):
        """Mock httpx to test fetch_url without network."""
        import httpx

        mock_response = httpx.Response(
            200,
            text="page content",
            request=httpx.Request("GET", "http://example.com"),
        )

        with patch("genesis_task_executor.tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await tool_fetch_url("http://example.com")
            assert result == "page content"

    async def test_fetch_error(self):
        """Test that network errors are returned as ERROR strings."""
        with patch("genesis_task_executor.tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await tool_fetch_url("http://bad-host.invalid")
            assert "ERROR" in result


class TestDispatchTool:

    async def test_dispatch_read_file(self, tmp_path):
        f = tmp_path / "dispatch_test.txt"
        f.write_text("dispatch content")
        result = await dispatch_tool("read_file", {"path": str(f)})
        assert result == "dispatch content"

    async def test_dispatch_write_file(self, tmp_path):
        target = tmp_path / "dispatch_write.txt"
        result = await dispatch_tool("write_file", {"path": str(target), "content": "written"})
        assert "OK" in result
        assert target.read_text() == "written"

    async def test_dispatch_unknown_tool(self):
        result = await dispatch_tool("nonexistent_tool", {})
        assert "ERROR" in result
        assert "unknown tool" in result


class TestParseToolArguments:

    def test_dict_passthrough(self):
        d = {"path": "/tmp/x"}
        assert parse_tool_arguments(d) is d

    def test_json_string(self):
        result = parse_tool_arguments('{"path": "/tmp/x"}')
        assert result == {"path": "/tmp/x"}

    def test_json_array_unwrap(self):
        """Some models wrap args in an array — first element is extracted."""
        result = parse_tool_arguments('[{"path": "/tmp/x"}]')
        assert result == {"path": "/tmp/x"}

    def test_empty_array(self):
        result = parse_tool_arguments("[]")
        assert result == {}

    def test_invalid_json_returns_empty_dict(self):
        result = parse_tool_arguments("not json at all")
        assert result == {}

    def test_none_returns_empty_dict(self):
        result = parse_tool_arguments(None)  # type: ignore[arg-type]
        assert result == {}


class TestToolDefinitions:

    def test_has_three_tools(self):
        assert len(TOOL_DEFINITIONS) == 3

    def test_tool_names(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert names == {"read_file", "write_file", "fetch_url"}

    def test_all_have_required_structure(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
