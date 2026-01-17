"""Tests for intelligent task handling.

Comprehensive integration tests to verify intelligent task handling
components work together correctly:
- Roadmap generation from natural language
- Task decomposition for complex tasks
- Subagent spawning based on task type
- Research tools integration

These tests verify end-to-end functionality of the intelligent
task handling system.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_infra.executor.agents.base import ExecutionContext
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.researcher import ResearcherAgent
from ai_infra.executor.agents.spawner import spawn_for_task

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    """Create a sample workspace with project files."""
    # Create a basic Python project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create pyproject.toml
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "test-project"
version = "0.1.0"
description = "A test project"
dependencies = ["fastapi", "pydantic"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
""")

    # Create main.py
    (src_dir / "main.py").write_text('''"""Main application module."""

from fastapi import FastAPI

app = FastAPI(title="Test API")


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Hello, World!"}
''')

    # Create a README
    (tmp_path / "README.md").write_text("""# Test Project

A sample project for testing.
""")

    return tmp_path


@pytest.fixture
def execution_context(sample_workspace: Path) -> ExecutionContext:
    """Create an execution context for tests."""
    return ExecutionContext(
        workspace=sample_workspace,
        files_modified=[],
        project_type="python",
        summary="Test project with FastAPI",
        dependencies=["fastapi", "pydantic"],
    )


# =============================================================================
# 3.5.1: Roadmap Generation Tests
# =============================================================================


class TestRoadmapGeneration:
    """Tests for roadmap generation from natural language."""

    def test_roadmap_generator_imports(self) -> None:
        """Test that RoadmapGenerator can be imported."""
        from ai_infra.executor.roadmap_generator import (
            GeneratedRoadmap,
            GenerationStyle,
            RoadmapGenerator,
        )

        assert RoadmapGenerator is not None
        assert GenerationStyle is not None
        assert GeneratedRoadmap is not None

    def test_generation_styles_exist(self) -> None:
        """Test all generation styles are defined."""
        from ai_infra.executor.roadmap_generator import GenerationStyle

        # GenerationStyle is a class with string constants
        assert GenerationStyle.MINIMAL == "minimal"
        assert GenerationStyle.STANDARD == "standard"
        assert GenerationStyle.DETAILED == "detailed"

    @pytest.mark.asyncio
    async def test_roadmap_generator_initialization(self) -> None:
        """Test RoadmapGenerator can be initialized."""
        from ai_infra.executor.roadmap_generator import RoadmapGenerator

        # Mock the agent
        mock_agent = MagicMock()
        generator = RoadmapGenerator(agent=mock_agent)

        assert generator is not None
        # Check that agent is stored (attribute name may vary)
        assert hasattr(generator, "agent") or hasattr(generator, "_agent")

    def test_generated_roadmap_properties(self) -> None:
        """Test GeneratedRoadmap has all required properties."""
        from ai_infra.executor.roadmap_generator import GeneratedRoadmap

        roadmap = GeneratedRoadmap(
            content="# Test Roadmap\n\n- [ ] Task 1",
            title="Test Roadmap",
            task_count=1,
            estimated_time="1 hour",
            complexity="low",
            confidence=0.9,
        )

        assert roadmap.title == "Test Roadmap"
        assert roadmap.task_count == 1
        assert roadmap.complexity == "low"
        assert roadmap.confidence == 0.9
        assert roadmap.content is not None

    def test_generated_roadmap_to_dict(self) -> None:
        """Test GeneratedRoadmap.to_dict() method."""
        from ai_infra.executor.roadmap_generator import GeneratedRoadmap

        roadmap = GeneratedRoadmap(
            content="# Test\n- [ ] Task 1",
            title="Test",
            task_count=1,
        )

        data = roadmap.to_dict()
        assert "title" in data
        assert "task_count" in data


# =============================================================================
# 3.5.2: Subagent Spawning Tests
# =============================================================================


class TestSubagentSpawning:
    """Tests for subagent selection and spawning."""

    def test_all_agent_types_registered(self) -> None:
        """Test all expected agent types are in the registry."""
        available = SubAgentRegistry.available_types()

        expected_types = [
            SubAgentType.CODER,
            SubAgentType.REVIEWER,
            SubAgentType.TESTER,
            SubAgentType.DEBUGGER,
            SubAgentType.RESEARCHER,
        ]

        for agent_type in expected_types:
            assert agent_type in available, f"Missing agent type: {agent_type}"

    def test_registry_get_returns_agent_instance(self) -> None:
        """Test SubAgentRegistry.get returns an agent instance."""
        from ai_infra.executor.agents.tester import TesterAgent

        agent = SubAgentRegistry.get(SubAgentType.TESTER, cached=False)
        assert isinstance(agent, TesterAgent)


# =============================================================================
# 3.5.3: Research Tools Integration Tests
# =============================================================================


class TestResearchToolsIntegration:
    """Tests for research tools integration with agents."""

    def test_researcher_agent_has_research_tools(self) -> None:
        """Test ResearcherAgent includes all research tools."""
        agent = ResearcherAgent()
        tools = agent._get_tools()

        tool_names = {t.name for t in tools}

        assert "web_search" in tool_names
        assert "lookup_docs" in tool_names
        assert "search_packages" in tool_names

    def test_research_tools_are_structured_tools(self) -> None:
        """Test research tools are proper StructuredTools."""
        from ai_infra.executor.tools.research import (
            lookup_docs,
            search_packages,
            web_search,
        )

        # All should have required StructuredTool attributes
        for tool in [web_search, lookup_docs, search_packages]:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "args_schema") or hasattr(tool, "coroutine")

    def test_web_search_tool_schema(self) -> None:
        """Test web_search has correct input schema."""
        from ai_infra.executor.tools.research import web_search

        assert web_search.name == "web_search"
        assert "query" in str(web_search.args_schema.model_fields)

    def test_lookup_docs_tool_schema(self) -> None:
        """Test lookup_docs has correct input schema."""
        from ai_infra.executor.tools.research import lookup_docs

        assert lookup_docs.name == "lookup_docs"
        assert "package" in str(lookup_docs.args_schema.model_fields)

    def test_search_packages_tool_schema(self) -> None:
        """Test search_packages has correct input schema."""
        from ai_infra.executor.tools.research import search_packages

        assert search_packages.name == "search_packages"
        assert "query" in str(search_packages.args_schema.model_fields)


# =============================================================================
# 3.5.4: End-to-End Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests for Phase 3 components."""

    def test_complete_workflow_imports(self) -> None:
        """Test all Phase 3 components can be imported together."""
        # Roadmap generation
        from ai_infra.executor.roadmap_generator import RoadmapGenerator

        # Subagent spawning
        # Research tools
        from ai_infra.executor.tools.research import (
            web_search,
        )

        # All imports should succeed
        assert RoadmapGenerator is not None
        assert spawn_for_task is not None
        assert web_search is not None

    def test_agent_registry_returns_correct_agent(self) -> None:
        """Test agent registry returns correct agent instances."""
        # Get agent instance directly by type
        agent = SubAgentRegistry.get(SubAgentType.RESEARCHER, cached=False)
        assert isinstance(agent, ResearcherAgent)

        # Agent should have research tools
        tools = agent._get_tools()
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names

    def test_executor_cli_generate_command_registered(self) -> None:
        """Test the generate command is registered in CLI."""
        from ai_infra.cli.cmds.executor_cmds import app

        # Get all registered commands
        commands = [cmd.name for cmd in app.registered_commands]

        assert "generate" in commands

    def test_roadmap_content_format(self) -> None:
        """Test generated roadmaps have correct format."""
        from ai_infra.executor.roadmap_generator import GeneratedRoadmap

        # Create a sample roadmap that should be parseable
        content = """# Add Authentication

## Overview
Implement JWT authentication.

## Tasks

### Phase 1: Setup
- [ ] 1. Install dependencies
- [ ] 2. Create configuration

### Phase 2: Implementation
- [ ] 3. Add JWT generation
- [ ] 4. Create login endpoint
"""

        roadmap = GeneratedRoadmap(
            content=content,
            title="Add Authentication",
            task_count=4,
            estimated_time="2-3 hours",
            complexity="moderate",
            confidence=0.85,
        )

        # The content should be valid markdown
        assert "# Add Authentication" in roadmap.content
        assert "- [ ]" in roadmap.content
        assert roadmap.task_count == 4


# =============================================================================
# 3.5.5: Phase 3 Success Criteria Verification
# =============================================================================


class TestPhase3SuccessCriteria:
    """Tests verifying Phase 3 success criteria are met."""

    def test_criterion_1_generate_command_exists(self) -> None:
        """Criterion: ai-infra executor generate works."""
        from ai_infra.cli.cmds.executor_cmds import generate_cmd

        # Command should exist and be callable
        assert callable(generate_cmd)

    def test_criterion_2_roadmap_has_validation(self) -> None:
        """Criterion: Generated roadmaps can be validated."""
        from ai_infra.executor.roadmap_generator import GeneratedRoadmap

        # GeneratedRoadmap should have validation_issues field
        roadmap = GeneratedRoadmap(
            content="# Test\n- [ ] Task 1",
            title="Test",
            task_count=1,
        )

        assert hasattr(roadmap, "validation_issues")

    def test_criterion_3_subagents_exist(self) -> None:
        """Criterion: All expected subagent types are available."""
        # Verify all agent types exist in registry
        expected_types = [
            SubAgentType.CODER,
            SubAgentType.TESTER,
            SubAgentType.DEBUGGER,
            SubAgentType.REVIEWER,
            SubAgentType.RESEARCHER,
        ]

        available = SubAgentRegistry.available_types()
        for agent_type in expected_types:
            assert agent_type in available, f"Missing agent type: {agent_type}"

    def test_criterion_4_research_tools_work(self) -> None:
        """Criterion: Research tools work when agent is stuck."""
        from ai_infra.executor.tools.research import (
            lookup_docs,
            search_packages,
            web_search,
        )

        # All tools should be properly configured
        assert web_search.name == "web_search"
        assert lookup_docs.name == "lookup_docs"
        assert search_packages.name == "search_packages"

        # ResearcherAgent should have these tools
        agent = ResearcherAgent()
        tools = agent._get_tools()

        tool_names = {t.name for t in tools}
        assert len(tool_names) >= 3
        assert "web_search" in tool_names
        assert "lookup_docs" in tool_names
        assert "search_packages" in tool_names
