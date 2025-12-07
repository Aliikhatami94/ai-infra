"""Tests for MCP tool loading and cache helpers."""

import asyncio

import pytest

from ai_infra.mcp import tools as tools_module
from ai_infra.mcp.tools import (
    clear_mcp_cache,
    get_cache_stats,
    get_cached_tools,
    is_cached,
    load_mcp_tools_cached,
)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_mcp_cache()
    tools_module._locks.clear()
    yield
    clear_mcp_cache()
    tools_module._locks.clear()


@pytest.mark.asyncio
async def test_load_mcp_tools_cached_reuses_cache(monkeypatch):
    calls: list[str] = []

    class FakeConfig:
        def __init__(self, *, transport: str, url: str):
            self.transport = transport
            self.url = url

    class FakeClient:
        def __init__(self, configs):
            self.configs = configs

        async def list_tools(self):
            calls.append("called")
            return ["tool-a"]

    monkeypatch.setattr("ai_infra.mcp.client.MCPClient", FakeClient)
    monkeypatch.setattr("ai_infra.mcp.client.models.McpServerConfig", FakeConfig)

    url = "http://localhost:8000/mcp"
    first = await load_mcp_tools_cached(url)
    second = await load_mcp_tools_cached(url)

    assert first is second
    assert calls == ["called"]
    assert get_cached_tools(url) == first
    assert is_cached(url) is True


@pytest.mark.asyncio
async def test_load_mcp_tools_cached_force_refresh(monkeypatch):
    call_count = 0

    class FakeConfig:
        def __init__(self, *, transport: str, url: str):
            self.transport = transport
            self.url = url

    class FakeClient:
        def __init__(self, *args, **kwargs):
            # Accept arbitrary args to mirror real MCPClient signature
            pass

        async def list_tools(self):
            nonlocal call_count
            call_count += 1
            return [f"tool-{call_count}"]

    monkeypatch.setattr("ai_infra.mcp.client.MCPClient", FakeClient)
    monkeypatch.setattr("ai_infra.mcp.client.models.McpServerConfig", FakeConfig)

    url = "http://localhost:8000/mcp"
    first = await load_mcp_tools_cached(url)
    second = await load_mcp_tools_cached(url, force_refresh=True)

    assert first == ["tool-1"]
    assert second == ["tool-2"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_load_mcp_tools_cached_is_thread_safe(monkeypatch):
    call_count = 0

    class FakeConfig:
        def __init__(self, *, transport: str, url: str):
            self.transport = transport
            self.url = url

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_tools(self):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0)
            return ["tool"]

    monkeypatch.setattr("ai_infra.mcp.client.MCPClient", FakeClient)
    monkeypatch.setattr("ai_infra.mcp.client.models.McpServerConfig", FakeConfig)

    url = "http://localhost:8000/mcp"

    results = await asyncio.gather(
        load_mcp_tools_cached(url),
        load_mcp_tools_cached(url),
    )

    assert results[0] is results[1]
    assert call_count == 1


def test_clear_mcp_cache_and_stats():
    tools_module._cached_tools["one"] = ["a"]
    tools_module._cached_tools["two"] = ["b", "c"]

    stats = get_cache_stats()
    assert set(stats["cached_urls"]) == {"one", "two"}
    assert stats["cache_size"] == 2
    assert stats["total_tools"] == 3

    clear_mcp_cache("one")
    stats_after = get_cache_stats()
    assert stats_after["cached_urls"] == ["two"]
    assert stats_after["cache_size"] == 1

    clear_mcp_cache()
    assert get_cache_stats()["cache_size"] == 0
