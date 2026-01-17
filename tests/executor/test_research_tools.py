"""Tests for Phase 3.4: Research Capability.

Tests for research tools: web search, documentation lookup,
and package registry search.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_infra.executor.tools.research import (
    lookup_docs,
    search_packages,
    web_search,
)
from ai_infra.executor.tools.research.docs_lookup import (
    DocsLookupResult,
    DocsLookupTool,
    create_docs_lookup_tool,
)
from ai_infra.executor.tools.research.package_search import (
    PackageInfo,
    PackageSearchTool,
    create_package_search_tool,
)
from ai_infra.executor.tools.research.web_search import (
    WebSearchResult,
    WebSearchTool,
    create_web_search_tool,
)

# =============================================================================
# WebSearchTool Tests
# =============================================================================


class TestWebSearchResult:
    """Tests for WebSearchResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a search result."""
        result = WebSearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"

    def test_result_to_dict(self) -> None:
        """Test converting result to dict."""
        result = WebSearchResult(
            title="Test",
            url="https://test.com",
            snippet="Snippet",
        )
        data = result.to_dict()
        assert data["title"] == "Test"
        assert data["url"] == "https://test.com"
        assert data["snippet"] == "Snippet"


class TestWebSearchTool:
    """Tests for WebSearchTool class."""

    def test_tool_initialization_defaults(self) -> None:
        """Test tool initializes with defaults."""
        tool = WebSearchTool()
        assert tool._api_key is None or isinstance(tool._api_key, str)
        assert tool._backend == "auto"

    def test_tool_with_api_key(self) -> None:
        """Test tool with API key."""
        tool = WebSearchTool(api_key="test-key")
        assert tool._api_key == "test-key"

    def test_tool_with_backend_override(self) -> None:
        """Test tool with specific backend."""
        tool = WebSearchTool(backend="duckduckgo")
        assert tool._backend == "duckduckgo"

    def test_create_tool_function(self) -> None:
        """Test create_web_search_tool returns StructuredTool."""
        tool = create_web_search_tool()
        assert tool.name == "web_search"
        assert "Search the web" in tool.description

    def test_default_tool_instance_exists(self) -> None:
        """Test default web_search tool is available."""
        assert web_search is not None
        assert web_search.name == "web_search"

    @pytest.mark.asyncio
    async def test_search_auto_backend_no_key(self) -> None:
        """Test auto backend uses duckduckgo when no API key."""
        tool = WebSearchTool(api_key=None, backend="auto")

        # Mock the module-level function
        with patch("ai_infra.executor.tools.research.web_search._search_duckduckgo") as mock_ddg:
            mock_ddg.return_value = [
                WebSearchResult(
                    title="Test Result",
                    url="https://test.com",
                    snippet="Test snippet",
                )
            ]

            results = await tool.search("test query")
            mock_ddg.assert_called_once_with("test query", 5)
            assert len(results) == 1
            assert results[0].title == "Test Result"

    @pytest.mark.asyncio
    async def test_search_with_brave_backend(self) -> None:
        """Test Brave Search when API key is provided."""
        tool = WebSearchTool(api_key="test-key", backend="brave")

        with patch("ai_infra.executor.tools.research.web_search._search_brave") as mock_brave:
            mock_brave.return_value = [
                WebSearchResult(
                    title="Brave Result",
                    url="https://brave.com",
                    snippet="From Brave",
                )
            ]

            results = await tool.search("test query")
            mock_brave.assert_called_once_with("test query", 5, "test-key")
            assert len(results) == 1
            assert results[0].title == "Brave Result"


# =============================================================================
# DocsLookupTool Tests
# =============================================================================


class TestDocsLookupResult:
    """Tests for DocsLookupResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a docs result."""
        result = DocsLookupResult(
            package="fastapi",
            topic="authentication",
            content="FastAPI documentation content",
            source="known_docs",
            url="https://fastapi.tiangolo.com",
        )
        assert result.package == "fastapi"
        assert result.topic == "authentication"
        assert result.source == "known_docs"
        assert result.url == "https://fastapi.tiangolo.com"

    def test_result_to_dict(self) -> None:
        """Test converting result to dict."""
        result = DocsLookupResult(
            package="react",
            topic="hooks",
            content="React hooks documentation",
            source="known_docs",
            url="https://react.dev",
        )
        data = result.to_dict()
        assert data["package"] == "react"
        assert data["topic"] == "hooks"
        assert data["content"] == "React hooks documentation"
        assert data["source"] == "known_docs"


class TestDocsLookupTool:
    """Tests for DocsLookupTool class."""

    def test_tool_initialization(self) -> None:
        """Test tool initializes."""
        tool = DocsLookupTool()
        assert tool is not None

    def test_create_tool_function(self) -> None:
        """Test create_docs_lookup_tool returns StructuredTool."""
        tool = create_docs_lookup_tool()
        assert tool.name == "lookup_docs"
        assert "documentation" in tool.description.lower()

    def test_default_tool_instance_exists(self) -> None:
        """Test default lookup_docs tool is available."""
        assert lookup_docs is not None
        assert lookup_docs.name == "lookup_docs"

    @pytest.mark.asyncio
    async def test_lookup_known_package(self) -> None:
        """Test looking up a well-known package."""
        tool = DocsLookupTool()

        # FastAPI is in _KNOWN_DOCS
        result = await tool.lookup("fastapi")
        assert result is not None
        assert result.package == "fastapi"
        assert "fastapi" in result.url.lower() or "docs" in result.content.lower()

    @pytest.mark.asyncio
    async def test_lookup_with_topic(self) -> None:
        """Test looking up with a specific topic."""
        tool = DocsLookupTool()

        result = await tool.lookup("pydantic", topic="validation")
        assert result is not None
        assert result.package == "pydantic"
        assert result.topic == "validation"


# =============================================================================
# PackageSearchTool Tests
# =============================================================================


class TestPackageInfo:
    """Tests for PackageInfo dataclass."""

    def test_info_creation(self) -> None:
        """Test creating package info."""
        info = PackageInfo(
            name="requests",
            version="2.31.0",
            description="HTTP library",
            registry="pypi",
            url="https://pypi.org/project/requests/",
        )
        assert info.name == "requests"
        assert info.version == "2.31.0"
        assert info.description == "HTTP library"
        assert info.registry == "pypi"

    def test_info_to_dict(self) -> None:
        """Test converting info to dict."""
        info = PackageInfo(
            name="axios",
            version="1.6.0",
            description="HTTP client",
            registry="npm",
            url="https://www.npmjs.com/package/axios",
        )
        data = info.to_dict()
        assert data["name"] == "axios"
        assert data["version"] == "1.6.0"
        assert data["registry"] == "npm"


class TestPackageSearchTool:
    """Tests for PackageSearchTool class."""

    def test_tool_initialization(self) -> None:
        """Test tool initializes."""
        tool = PackageSearchTool()
        assert tool is not None

    def test_create_tool_function(self) -> None:
        """Test create_package_search_tool returns StructuredTool."""
        tool = create_package_search_tool()
        assert tool.name == "search_packages"
        assert "package" in tool.description.lower()

    def test_default_tool_instance_exists(self) -> None:
        """Test default search_packages tool is available."""
        assert search_packages is not None
        assert search_packages.name == "search_packages"

    @pytest.mark.asyncio
    async def test_search_pypi(self) -> None:
        """Test searching PyPI registry."""
        tool = PackageSearchTool()

        with patch("ai_infra.executor.tools.research.package_search._search_pypi") as mock_search:
            mock_search.return_value = [
                PackageInfo(
                    name="httpx",
                    version="0.25.0",
                    description="HTTP client",
                    registry="pypi",
                    url="https://pypi.org/project/httpx/",
                )
            ]

            results = await tool.search("http client", "pypi")
            mock_search.assert_called_once()
            assert len(results) == 1
            assert results[0].name == "httpx"

    @pytest.mark.asyncio
    async def test_search_npm(self) -> None:
        """Test searching npm registry."""
        tool = PackageSearchTool()

        with patch("ai_infra.executor.tools.research.package_search._search_npm") as mock_search:
            mock_search.return_value = [
                PackageInfo(
                    name="axios",
                    version="1.6.0",
                    description="HTTP client",
                    registry="npm",
                    url="https://www.npmjs.com/package/axios",
                )
            ]

            results = await tool.search("http client", "npm")
            mock_search.assert_called_once()
            assert len(results) == 1
            assert results[0].name == "axios"


# =============================================================================
# Integration Tests
# =============================================================================


class TestResearchToolsIntegration:
    """Integration tests for research tools module."""

    def test_all_tools_exported(self) -> None:
        """Test that all tools are properly exported."""
        from ai_infra.executor.tools.research import (
            lookup_docs,
            search_packages,
            web_search,
        )

        assert web_search is not None
        assert lookup_docs is not None
        assert search_packages is not None

    def test_tools_have_correct_interface(self) -> None:
        """Test tools have callable interface."""
        # Each tool should be a StructuredTool with name and description
        assert hasattr(web_search, "name")
        assert hasattr(web_search, "description")
        assert hasattr(web_search, "coroutine") or hasattr(web_search, "_run")

        assert hasattr(lookup_docs, "name")
        assert hasattr(lookup_docs, "description")
        assert hasattr(lookup_docs, "coroutine") or hasattr(lookup_docs, "_run")

        assert hasattr(search_packages, "name")
        assert hasattr(search_packages, "description")
        assert hasattr(search_packages, "coroutine") or hasattr(search_packages, "_run")

    def test_tools_can_be_used_with_agent(self) -> None:
        """Test tools can be collected for agent use."""
        tools = [web_search, lookup_docs, search_packages]
        assert len(tools) == 3

        # All tools should have names
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names
        assert "lookup_docs" in tool_names
        assert "search_packages" in tool_names


# =============================================================================
# ResearcherAgent Integration Tests
# =============================================================================


class TestResearcherAgentWithTools:
    """Tests for ResearcherAgent with research tools integration."""

    def test_researcher_has_research_tools(self) -> None:
        """Test ResearcherAgent includes research tools."""
        from ai_infra.executor.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        tools = agent._get_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names
        assert "lookup_docs" in tool_names
        assert "search_packages" in tool_names

    def test_researcher_prompt_mentions_tools(self) -> None:
        """Test ResearcherAgent prompt documents the tools."""
        from ai_infra.executor.agents.researcher import RESEARCHER_PROMPT

        assert "web_search" in RESEARCHER_PROMPT
        assert "lookup_docs" in RESEARCHER_PROMPT
        assert "search_packages" in RESEARCHER_PROMPT
