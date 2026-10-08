# Copyright 2024 TaskFlowAI Contributors. Licensed under Apache License 2.0.

"""Opt-in, anonymous Parallel Search MCP tools for synchronous TaskflowAI tasks."""

import asyncio
import json
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ParallelTools:
    """Create one instance per conversation to share its anonymous MCP session ID.

    Install ``taskflowai[parallel_tools]`` and pass the bound ``search`` and
    ``fetch`` methods to ``Agent.tools``. No Parallel API key is read or sent.
    """

    def __init__(self) -> None:
        self._session_id = str(uuid4())

    def search(
        self, objective: str, search_queries: List[str], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Search the web for an objective using concise keyword queries.

        Args:
            objective: Specific information to find, with preferred sources if needed.
            search_queries: At least one related keyword query, ideally 3-6 words each.
            timeout: Maximum seconds for the complete MCP operation; defaults to 30.

        Returns:
            A dictionary with results (URLs, titles and excerpts), session_id and
            any server warnings. The anonymous free tier has rate limits.
        """
        return self._call(
            "web_search", {"objective": objective, "search_queries": search_queries}, timeout
        )

    def fetch(
        self, urls: List[str], objective: Optional[str] = None, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Fetch relevant excerpts from specific web pages through Parallel MCP.

        Args:
            urls: One to twenty HTTP/HTTPS URLs to read.
            objective: Optional description of what to extract, at most 200 characters.
            timeout: Maximum seconds for the complete MCP operation; defaults to 30.

        Returns:
            A dictionary with results (URLs, titles and excerpts), per-URL errors,
            session_id and any server warnings. Prefer search excerpts when sufficient.
        """
        arguments: Dict[str, Any] = {"urls": urls}
        if objective is not None:
            arguments["objective"] = objective
        return self._call("web_fetch", arguments, timeout)

    def _call(self, name: str, arguments: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("ParallelTools is synchronous; use asyncio.to_thread in async code")

        try:
            import anyio
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise ImportError(
                "Install Parallel tools with: pip install 'taskflowai[parallel_tools]'"
            ) from exc

        async def execute() -> Dict[str, Any]:
            with anyio.fail_after(timeout):
                async with streamablehttp_client(
                    "https://search.parallel.ai/mcp",
                    headers={
                        "User-Agent": "TaskFlowAI/0.5.13 (https://github.com/philippe-page/taskflowai)"
                    },
                    timeout=timeout,
                    sse_read_timeout=timeout,
                ) as (read, write, _):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=timeout)
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            name, {**arguments, "session_id": self._session_id}
                        )
                        text = "\n".join(
                            block.text for block in result.content if block.type == "text"
                        )
                        if result.isError:
                            raise RuntimeError(f"Parallel {name} failed: {text}")
                        if result.structuredContent is not None:
                            return result.structuredContent
                        data = json.loads(text)
                        if not isinstance(data, dict):
                            raise ValueError("Parallel MCP returned a non-object result")
                        return data

        return asyncio.run(execute())
