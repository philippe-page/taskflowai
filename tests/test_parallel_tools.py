"""Exercise public tools, Task dispatch, and real SDK HTTP request construction."""

import asyncio
import builtins
import json

import httpx
import pytest

from taskflowai import Agent, ParallelTools, Task, WebTools
from taskflowai.tools import ParallelTools as ToolsExport


@pytest.fixture
def mcp_http(monkeypatch):
    requests = []
    state = {"error": False, "text_only": False}

    async def send(client, request, **kwargs):
        body = json.loads(request.content) if request.content else None
        requests.append((request, body))
        if not body or "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif body["method"] == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif body["method"] == "tools/call":
            args = body["params"]["arguments"]
            data = {
                "results": [{"url": "https://example.com", "excerpts": ["Useful excerpt"]}],
                "session_id": args["session_id"],
                "warnings": [{"type": "warning", "message": "Fixture warning"}],
            }
            if body["params"]["name"] == "web_fetch":
                data["errors"] = [{"url": "https://example.com/missing", "error_type": "404"}]
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "Quota reached" if state["error"] else json.dumps(data),
                    }
                ],
                "isError": state["error"],
            }
            if not state["text_only"]:
                result["structuredContent"] = data
        else:
            raise AssertionError(f"Unexpected method: {body['method']}")
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return requests, state


def test_public_exports_and_task_dispatch(mcp_http, monkeypatch):
    requests, _ = mcp_http
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-used")
    assert ToolsExport is ParallelTools
    assert Agent(role="r", goal="g").tools is None
    assert WebTools.exa_search.__name__ == "exa_search"
    tools = ParallelTools()
    calls = iter(
        [
            {
                "tool_calls": [
                    {
                        "tool": "search",
                        "params": {
                            "objective": "Find documentation",
                            "search_queries": ["Python asyncio docs"],
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "tool": "fetch",
                        "params": {
                            "urls": ["https://example.com"],
                            "objective": "Read documentation",
                        },
                    }
                ]
            },
            {"status": "END_TOOL_USAGE"},
        ]
    )

    def model(system_prompt, user_prompt, **kwargs):
        if kwargs.get("require_json_output"):
            return next(calls)
        assert "Useful excerpt" in user_prompt
        assert "Fixture warning" in user_prompt
        assert "404" in user_prompt
        return "Answer citing https://example.com"

    events = []
    agent = Agent(
        role="researcher", goal="find sources", llm=model, tools={tools.search, tools.fetch}
    )
    result = Task.create(agent=agent, instruction="Research documentation", callback=events.append)
    assert result == "Answer citing https://example.com"
    executed = [e for e in events if e["type"] == "tool_call"]
    assert [e["tool"] for e in executed] == ["search", "fetch"]
    calls_sent = [b for _, b in requests if b and b.get("method") == "tools/call"]
    assert [b["params"]["name"] for b in calls_sent] == ["web_search", "web_fetch"]
    sessions = {b["params"]["arguments"]["session_id"] for b in calls_sent}
    assert len(sessions) == 1
    assert len(next(iter(sessions))) == 36
    for request, _ in requests:
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["User-Agent"] == (
            "TaskFlowAI/0.5.13 (https://github.com/philippe-page/taskflowai)"
        )
        assert "authorization" not in request.headers


def test_text_fallback_and_fetch_defaults(mcp_http):
    requests, state = mcp_http
    state["text_only"] = True
    result = ParallelTools().fetch(["https://example.com"])
    assert result["errors"][0]["error_type"] == "404"
    body = next(b for _, b in requests if b and b.get("method") == "tools/call")
    assert set(body["params"]["arguments"]) == {"urls", "session_id"}


def test_tool_error_does_not_return_success(mcp_http):
    _, state = mcp_http
    state["error"] = True
    with pytest.raises(Exception) as error:
        ParallelTools().search("Find docs", ["Python docs"])

    # AnyIO may wrap the server error when closing SDK task groups.
    def messages(exc):
        return str(exc) + " ".join(messages(child) for child in getattr(exc, "exceptions", []))

    assert "Quota reached" in messages(error.value)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite positive"):
        ParallelTools().search("Find docs", ["Python docs"], timeout=timeout)


def test_overall_timeout(monkeypatch):
    async def stalled_send(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(httpx.AsyncClient, "send", stalled_send)
    with pytest.raises(TimeoutError):
        ParallelTools().fetch(["https://example.com"], timeout=0.05)


def test_missing_optional_dependency(monkeypatch):
    original = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("MCP not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    with pytest.raises(ImportError, match=r"taskflowai\[parallel_tools\]"):
        ParallelTools().fetch(["https://example.com"])


def test_active_event_loop():
    async def invoke():
        with pytest.raises(RuntimeError, match="asyncio.to_thread"):
            ParallelTools().fetch(["https://example.com"])

    asyncio.run(invoke())
