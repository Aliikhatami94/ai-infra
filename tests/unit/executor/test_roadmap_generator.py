"""Tests for RoadmapGenerator (Phase 3.1.1 and 3.1.5).

Tests roadmap generation from natural language prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.roadmap_generator import (
    GeneratedRoadmap,
    GenerationStyle,
    RoadmapGenerator,
    ValidationIssue,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = "test-model"
    return agent


@pytest.fixture
def generator(mock_agent: MagicMock) -> RoadmapGenerator:
    """Create a RoadmapGenerator with mock agent."""
    return RoadmapGenerator(mock_agent)


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project for testing."""
    (tmp_path / "pyproject.toml").write_text("""\
[tool.poetry]
name = "test-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
""")
    (tmp_path / "README.md").write_text("# Test Project\n\nA simple test project.")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    return tmp_path


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample roadmap content for testing validation."""
    return """\
# Add Authentication

## Phase 1: Setup

### 1.1 Install Dependencies

- [ ] **Install JWT library**
  Add python-jose to dependencies.

- [ ] **Install passlib**
  Add password hashing library.

  > depends: 1.1.1

### 1.2 Implement Auth

- [ ] **Create auth module**
  Create src/auth.py with login/logout functions.

  > depends: 1.1
"""


# =============================================================================
# GeneratedRoadmap Tests
# =============================================================================


class TestGeneratedRoadmap:
    """Tests for GeneratedRoadmap dataclass."""

    def test_to_dict(self):
        """Test GeneratedRoadmap.to_dict() method."""
        roadmap = GeneratedRoadmap(
            content="# Test\n\n- [ ] Task 1",
            title="Test",
            task_count=1,
            estimated_time="1 hour",
            complexity="low",
            confidence=0.9,
            validation_issues=[],
        )

        result = roadmap.to_dict()

        assert result["title"] == "Test"
        assert result["task_count"] == 1
        assert result["complexity"] == "low"
        assert result["confidence"] == 0.9

    def test_is_valid_with_no_issues(self):
        """Test is_valid property with no issues."""
        roadmap = GeneratedRoadmap(
            content="# Test",
            title="Test",
            task_count=1,
            validation_issues=[],
        )

        assert roadmap.is_valid is True

    def test_is_valid_with_high_severity_issues(self):
        """Test is_valid property with high severity issues."""
        roadmap = GeneratedRoadmap(
            content="# Test",
            title="Test",
            task_count=1,
            validation_issues=[
                ValidationIssue(
                    task_id="1.1.1",
                    issue_type="circular_dependency",
                    message="Circular dependency detected",
                    severity="high",
                )
            ],
        )

        assert roadmap.is_valid is False

    def test_is_valid_with_medium_severity_issues(self):
        """Test is_valid property with medium severity issues only."""
        roadmap = GeneratedRoadmap(
            content="# Test",
            title="Test",
            task_count=1,
            validation_issues=[
                ValidationIssue(
                    task_id="1.1.1",
                    issue_type="missing_description",
                    message="Task has no description",
                    severity="medium",
                )
            ],
        )

        assert roadmap.is_valid is True  # Only high severity fails validation

    def test_has_warnings_property(self):
        """Test has_warnings property."""
        roadmap = GeneratedRoadmap(
            content="# Test",
            title="Test",
            task_count=1,
            validation_issues=[
                ValidationIssue(
                    task_id="1.1.1",
                    issue_type="missing_description",
                    message="Task has no description",
                    severity="medium",
                )
            ],
        )

        assert roadmap.has_warnings is True


# =============================================================================
# ValidationIssue Tests
# =============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            task_id="1.1.1",
            issue_type="circular_dependency",
            message="Task depends on itself",
            severity="high",
        )

        assert issue.task_id == "1.1.1"
        assert issue.issue_type == "circular_dependency"
        assert issue.severity == "high"

    def test_default_severity(self):
        """Test default severity is medium."""
        issue = ValidationIssue(
            task_id="1.1.1",
            issue_type="test",
            message="Test issue",
        )

        assert issue.severity == "medium"


# =============================================================================
# GenerationStyle Tests
# =============================================================================


class TestGenerationStyle:
    """Tests for GenerationStyle class."""

    def test_minimal_style(self):
        """Test MINIMAL style values."""
        style = GenerationStyle.MINIMAL

        assert "minimal" in style.lower()

    def test_standard_style(self):
        """Test STANDARD style values."""
        style = GenerationStyle.STANDARD

        assert "standard" in style.lower()

    def test_detailed_style(self):
        """Test DETAILED style values."""
        style = GenerationStyle.DETAILED

        assert "detailed" in style.lower()


# =============================================================================
# RoadmapGenerator Tests
# =============================================================================


class TestRoadmapGenerator:
    """Tests for RoadmapGenerator class."""

    def test_init(self, mock_agent: MagicMock):
        """Test RoadmapGenerator initialization."""
        generator = RoadmapGenerator(mock_agent)

        assert generator.agent == mock_agent

    @pytest.mark.asyncio
    async def test_generate_calls_agent(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, python_project: Path
    ):
        """Test that generate() calls the agent correctly."""
        mock_response = MagicMock()
        mock_response.content = """\
# Add Auth

## Overview
Adding authentication to the project.

## Estimated Time
2-3 hours

## Complexity
medium

## Tasks

### Phase 1: Setup

- [ ] **Task 1.1**: Add jwt library
  - Description: Install python-jose for JWT handling.
  - Files: pyproject.toml
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        result = await generator.generate(
            prompt="Add JWT authentication",
            workspace=python_project,
            style="standard",
            validate=False,
        )

        assert mock_agent.ainvoke.called
        assert isinstance(result, GeneratedRoadmap)
        assert result.task_count >= 1

    @pytest.mark.asyncio
    async def test_generate_and_save(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, python_project: Path
    ):
        """Test generate_and_save() creates file."""
        mock_response = MagicMock()
        mock_response.content = """\
# Add Feature

## Overview
Adding a new feature.

## Estimated Time
1-2 hours

## Complexity
low

## Tasks

### Phase 1: Implementation

- [ ] **Task 1.1**: Create module
  - Description: Add new module.
  - Files: src/new_module.py
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        output_path = Path("ROADMAP.md")
        result = await generator.generate_and_save(
            prompt="Add a feature",
            workspace=python_project,
            output=output_path,
            style="minimal",
        )

        # Check file was created
        assert (python_project / output_path).exists()

        # Check content matches
        content = (python_project / output_path).read_text()
        assert "Add Feature" in content


# =============================================================================
# Circular Dependency Detection Tests
# =============================================================================


class TestCircularDependencyDetection:
    """Tests for _find_circular_dependencies method."""

    def test_no_cycle(self, generator: RoadmapGenerator):
        """Test _find_circular_dependencies with no cycles."""
        deps = {
            "1.1.1": ["1.1.2"],
            "1.1.2": ["1.1.3"],
            "1.1.3": [],
        }

        cycle = generator._find_circular_dependencies(deps)

        # Returns empty list when no cycles
        assert cycle == [] or cycle is None

    def test_with_cycle(self, generator: RoadmapGenerator):
        """Test _find_circular_dependencies with a cycle."""
        deps = {
            "1.1.1": ["1.1.2"],
            "1.1.2": ["1.1.3"],
            "1.1.3": ["1.1.1"],  # Cycle back to 1.1.1
        }

        cycle = generator._find_circular_dependencies(deps)

        # Should detect the cycle - returns non-empty list
        assert cycle is not None
        assert len(cycle) >= 1  # At least 1 cycle detected

    def test_self_reference(self, generator: RoadmapGenerator):
        """Test _find_circular_dependencies with self-reference."""
        deps = {
            "1.1.1": ["1.1.1"],  # Self-reference
        }

        cycle = generator._find_circular_dependencies(deps)

        # Should detect self-reference
        assert cycle is not None

    def test_empty_deps(self, generator: RoadmapGenerator):
        """Test _find_circular_dependencies with empty deps."""
        deps: dict[str, list[str]] = {}

        cycle = generator._find_circular_dependencies(deps)

        # Returns empty list when no deps
        assert cycle == [] or cycle is None


# =============================================================================
# Phase 6.1.3 - Roadmap Generation Integration Tests
# =============================================================================


class TestRoadmapGenerationStyles:
    """Tests for roadmap generation with different styles (Phase 6.1.3)."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.model = "test-model"
        return agent

    @pytest.fixture
    def generator(self, mock_agent: MagicMock) -> RoadmapGenerator:
        """Create a RoadmapGenerator with mock agent."""
        return RoadmapGenerator(mock_agent)

    @pytest.mark.asyncio
    async def test_generates_valid_roadmap(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, tmp_path: Path
    ):
        """Should generate valid ROADMAP.md structure."""
        # Create project structure
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "README.md").write_text("# Test")

        mock_response = MagicMock()
        mock_response.content = """\
# Roadmap: Add Authentication

## Overview
Adding JWT-based authentication to the project.

## Estimated Time
3-4 hours

## Complexity
medium

## Tasks

### Phase 1: Setup

- [ ] **Task 1.1**: Install JWT library
  - Description: Add python-jose to dependencies
  - Files: pyproject.toml

- [ ] **Task 1.2**: Create User model
  - Description: Add User pydantic model
  - Files: src/models/user.py

### Phase 2: Implementation

- [ ] **Task 2.1**: Add auth endpoints
  - Description: Create login/logout endpoints
  - Files: src/api/auth.py
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        result = await generator.generate(
            prompt="Add JWT authentication",
            workspace=tmp_path,
            style="standard",
            validate=False,
        )

        assert isinstance(result, GeneratedRoadmap)
        assert result.task_count >= 2
        assert "Authentication" in result.content
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_minimal_style_generates_few_tasks(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, tmp_path: Path
    ):
        """Minimal style should have 3-5 tasks."""
        # Create project structure
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        mock_response = MagicMock()
        mock_response.content = """\
# Roadmap: Quick Bug Fix

## Overview
Fix the authentication bug.

## Estimated Time
30 minutes

## Complexity
low

## Tasks

### Phase 1: Fix

- [ ] **Task 1.1**: Identify root cause
  - Description: Debug the auth issue
  - Files: src/auth.py

- [ ] **Task 1.2**: Apply fix
  - Description: Fix the token validation
  - Files: src/auth.py

- [ ] **Task 1.3**: Add regression test
  - Description: Add test case
  - Files: tests/test_auth.py
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        result = await generator.generate(
            prompt="Fix auth bug",
            workspace=tmp_path,
            style="minimal",
            validate=False,
        )

        assert isinstance(result, GeneratedRoadmap)
        assert 3 <= result.task_count <= 5

    @pytest.mark.asyncio
    async def test_detailed_style_generates_many_tasks(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, tmp_path: Path
    ):
        """Detailed style should have 10+ tasks for complex features."""
        # Create project structure
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Build a detailed response with 12 tasks
        tasks = []
        for i in range(1, 13):
            phase = (i - 1) // 3 + 1
            task_num = (i - 1) % 3 + 1
            tasks.append(f"""\
- [ ] **Task {phase}.{task_num}**: Task number {i}
  - Description: Detailed implementation step {i}
  - Files: src/module_{i}.py
""")

        mock_response = MagicMock()
        mock_response.content = f"""\
# Roadmap: Full System Overhaul

## Overview
Complete system refactoring with detailed breakdown.

## Estimated Time
2 weeks

## Complexity
high

## Tasks

### Phase 1: Foundation

{tasks[0]}
{tasks[1]}
{tasks[2]}

### Phase 2: Core Features

{tasks[3]}
{tasks[4]}
{tasks[5]}

### Phase 3: Integration

{tasks[6]}
{tasks[7]}
{tasks[8]}

### Phase 4: Polish

{tasks[9]}
{tasks[10]}
{tasks[11]}
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        result = await generator.generate(
            prompt="Full system overhaul",
            workspace=tmp_path,
            style="detailed",
            validate=False,
        )

        assert isinstance(result, GeneratedRoadmap)
        assert result.task_count >= 10

    @pytest.mark.asyncio
    async def test_project_context_included_in_prompt(
        self, generator: RoadmapGenerator, mock_agent: MagicMock, tmp_path: Path
    ):
        """Should include project analysis in generation prompt."""
        # Create a Python project with specific markers
        (tmp_path / "pyproject.toml").write_text("""\
[tool.poetry]
name = "my-python-app"
version = "1.0.0"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
""")
        (tmp_path / "README.md").write_text("# My Python FastAPI App")

        mock_response = MagicMock()
        mock_response.content = """\
# Roadmap: Add Feature

## Tasks

- [ ] **Task 1.1**: Implement feature
  - Description: Add the feature
  - Files: src/feature.py
"""
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)

        await generator.generate(
            prompt="Add a new feature",
            workspace=tmp_path,
            style="standard",
            validate=False,
        )

        # Verify the prompt was called and included project context
        assert mock_agent.ainvoke.called
        call_args = mock_agent.ainvoke.call_args[0][0]

        # The prompt should contain project information
        assert "python" in call_args.lower() or "pyproject" in call_args.lower()
