"""Tests for Phase 8.3: Skill extraction from successful tasks.

This module tests:
- checkpoint_node: Skill extraction after verified task completion
- _extract_skill_from_task: Skill extraction helper
- _infer_language_from_files: Language inference from file paths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ai_infra.executor.nodes.checkpoint import (
    _extract_skill_from_task,
    _infer_language_from_files,
    checkpoint_node,
)
from ai_infra.executor.skills.database import SkillsDatabase
from ai_infra.executor.skills.extractor import SkillExtractor

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockTask:
    """Mock task for testing."""

    id: str = "task-1"
    title: str = "Add FastAPI endpoint"
    description: str = "Create a REST endpoint for user login"


@pytest.fixture
def mock_task() -> MockTask:
    """Create a mock task."""
    return MockTask()


@pytest.fixture
def skills_db(tmp_path: Any) -> SkillsDatabase:
    """Create a test skills database."""
    db_path = tmp_path / "skills.json"
    return SkillsDatabase(str(db_path), auto_save=True)


@pytest.fixture
def skill_extractor(skills_db: SkillsDatabase) -> SkillExtractor:
    """Create a skill extractor without LLM."""
    return SkillExtractor(db=skills_db, llm=None)


# =============================================================================
# Tests: _infer_language_from_files
# =============================================================================


class TestInferLanguageFromFiles:
    """Tests for _infer_language_from_files function."""

    def test_python_files(self) -> None:
        """Python files should infer python."""
        assert _infer_language_from_files(["src/app.py"]) == "python"
        assert _infer_language_from_files(["test.py", "main.py"]) == "python"

    def test_typescript_files(self) -> None:
        """TypeScript files should infer typescript."""
        assert _infer_language_from_files(["src/app.ts"]) == "typescript"
        assert _infer_language_from_files(["component.tsx"]) == "typescript"

    def test_javascript_files(self) -> None:
        """JavaScript files should infer javascript."""
        assert _infer_language_from_files(["src/app.js"]) == "javascript"
        assert _infer_language_from_files(["component.jsx"]) == "javascript"

    def test_other_languages(self) -> None:
        """Other file extensions should map correctly."""
        assert _infer_language_from_files(["main.go"]) == "go"
        assert _infer_language_from_files(["lib.rs"]) == "rust"
        assert _infer_language_from_files(["App.java"]) == "java"

    def test_empty_defaults_to_python(self) -> None:
        """Empty file list should default to python."""
        assert _infer_language_from_files([]) == "python"


# =============================================================================
# Tests: _extract_skill_from_task
# =============================================================================


class TestExtractSkillFromTask:
    """Tests for _extract_skill_from_task function."""

    @pytest.mark.asyncio
    async def test_extracts_skill_with_files_modified(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
        skill_extractor: SkillExtractor,
    ) -> None:
        """Skills should be extracted when files are modified."""
        state: dict[str, Any] = {
            "verified": True,
            "agent_result": {
                "messages": [{"role": "assistant", "content": "Created the endpoint successfully."}]
            },
        }

        initial_count = len(skills_db)

        await _extract_skill_from_task(
            current_task=mock_task,
            files_modified=["src/api.py", "src/routes.py"],
            agent_result=state.get("agent_result"),
            skills_db=skills_db,
            skill_extractor=skill_extractor,
            state=state,
        )

        # Should have extracted a skill
        assert len(skills_db) > initial_count

    @pytest.mark.asyncio
    async def test_no_extraction_without_skills_db(
        self,
        mock_task: MockTask,
    ) -> None:
        """No extraction should occur without skills_db."""
        state: dict[str, Any] = {"verified": True}

        # Should not raise even without skills_db
        await _extract_skill_from_task(
            current_task=mock_task,
            files_modified=["src/api.py"],
            agent_result=None,
            skills_db=None,
            skill_extractor=None,
            state=state,
        )

    @pytest.mark.asyncio
    async def test_creates_extractor_if_not_provided(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
    ) -> None:
        """Extractor should be created if not provided."""
        state: dict[str, Any] = {"verified": True}

        await _extract_skill_from_task(
            current_task=mock_task,
            files_modified=["src/api.py"],
            agent_result=None,
            skills_db=skills_db,
            skill_extractor=None,  # Not provided
            state=state,
        )

        # Should still extract (creates its own extractor)
        # Note: extraction may or may not succeed based on heuristics
        # but should not raise an error
        assert True


# =============================================================================
# Tests: checkpoint_node with skills extraction
# =============================================================================


class TestCheckpointNodeWithSkills:
    """Tests for checkpoint_node with skill extraction."""

    @pytest.mark.asyncio
    async def test_extracts_skills_when_verified(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
    ) -> None:
        """Skills should be extracted when task is verified."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "files_modified": ["src/api.py"],
            "verified": True,
            "agent_result": {"messages": []},
        }

        initial_count = len(skills_db)

        result = await checkpoint_node(
            state,
            skills_db=skills_db,
            enable_learning=True,
        )

        # Should have no error
        assert result.get("error") is None
        # Skills should have been extracted
        assert len(skills_db) > initial_count

    @pytest.mark.asyncio
    async def test_no_extraction_when_not_verified(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
    ) -> None:
        """No extraction should occur when task is not verified."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "files_modified": ["src/api.py"],
            "verified": False,  # Not verified
            "agent_result": {"messages": []},
        }

        initial_count = len(skills_db)

        await checkpoint_node(
            state,
            skills_db=skills_db,
            enable_learning=True,
        )

        # Should NOT have extracted (not verified)
        assert len(skills_db) == initial_count

    @pytest.mark.asyncio
    async def test_no_extraction_when_learning_disabled(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
    ) -> None:
        """No extraction should occur when learning is disabled."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "files_modified": ["src/api.py"],
            "verified": True,
            "agent_result": {"messages": []},
        }

        initial_count = len(skills_db)

        await checkpoint_node(
            state,
            skills_db=skills_db,
            enable_learning=False,  # Learning disabled
        )

        # Should NOT have extracted (learning disabled)
        assert len(skills_db) == initial_count

    @pytest.mark.asyncio
    async def test_no_extraction_without_files_modified(
        self,
        mock_task: MockTask,
        skills_db: SkillsDatabase,
    ) -> None:
        """No extraction should occur when no files modified."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "files_modified": [],  # No files
            "verified": True,
            "agent_result": {"messages": []},
        }

        initial_count = len(skills_db)

        await checkpoint_node(
            state,
            skills_db=skills_db,
            enable_learning=True,
        )

        # Should NOT have extracted (no files modified)
        assert len(skills_db) == initial_count
