"""Tests for Phase 8.2: Skills injection into build_context_node.

This module tests:
- _extract_keywords: Keyword extraction from task title/description
- _infer_language: Language inference from file hints
- _build_skills_section: Skills section formatting for prompts
- build_context_node: Skills injection into state and prompt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ai_infra.executor.nodes.context import (
    _build_skills_section,
    _extract_keywords,
    _infer_language,
    build_context_node,
)
from ai_infra.executor.skills.database import SkillsDatabase
from ai_infra.executor.skills.models import Skill, SkillType

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockTask:
    """Mock task for testing."""

    id: str = "1"
    title: str = "Add FastAPI endpoint"
    description: str = "Create a REST endpoint for user login"
    file_hints: list[str] | None = None

    def __post_init__(self) -> None:
        if self.file_hints is None:
            self.file_hints = ["src/api.py"]


@pytest.fixture
def mock_task() -> MockTask:
    """Create a mock task."""
    return MockTask()


@pytest.fixture
def skills_db(tmp_path: Any) -> SkillsDatabase:
    """Create a test skills database with sample skills."""
    db_path = tmp_path / "skills.json"
    db = SkillsDatabase(str(db_path), auto_save=False)

    # Add a matching skill
    skill = Skill.create(
        type=SkillType.PATTERN,
        title="FastAPI error handling",
        description="Use HTTPException for errors",
        languages=["python"],
        frameworks=["fastapi"],
        task_keywords=["endpoint", "api", "fastapi"],
        pattern='raise HTTPException(status_code=404, detail="Not found")',
        rationale="Consistent error responses",
    )
    db.add(skill)

    # Add another skill that matches less
    skill2 = Skill.create(
        type=SkillType.APPROACH,
        title="REST API design",
        description="Follow REST conventions",
        languages=["python"],
        task_keywords=["api", "rest", "endpoint"],
        pattern="Use proper HTTP methods: GET, POST, PUT, DELETE",
        rationale="Industry standard approach",
    )
    db.add(skill2)

    return db


# =============================================================================
# Tests: _extract_keywords
# =============================================================================


class TestExtractKeywords:
    """Tests for _extract_keywords function."""

    def test_extracts_meaningful_words(self) -> None:
        """Keywords should extract meaningful words from title and description."""
        keywords = _extract_keywords(
            "Add user authentication endpoint",
            "Create a REST endpoint for user login with JWT tokens",
        )

        assert "user" in keywords
        assert "authentication" in keywords
        assert "endpoint" in keywords
        assert "jwt" in keywords
        assert "tokens" in keywords

    def test_filters_stop_words(self) -> None:
        """Common stop words should be filtered out."""
        keywords = _extract_keywords(
            "The user authentication",
            "Create a new endpoint for the login",
        )

        # Stop words should not be in keywords
        assert "the" not in keywords
        assert "a" not in keywords
        assert "for" not in keywords

    def test_limits_keywords(self) -> None:
        """Should limit to maximum 20 keywords."""
        long_text = " ".join([f"keyword{i}" for i in range(50)])
        keywords = _extract_keywords(long_text, long_text)

        assert len(keywords) <= 20

    def test_handles_empty_input(self) -> None:
        """Should handle empty input gracefully."""
        keywords = _extract_keywords("", "")
        assert keywords == []


# =============================================================================
# Tests: _infer_language
# =============================================================================


class TestInferLanguage:
    """Tests for _infer_language function."""

    def test_python_files(self) -> None:
        """Python files should infer python."""
        assert _infer_language(["src/app.py"]) == "python"
        assert _infer_language(["test.py", "main.py"]) == "python"

    def test_typescript_files(self) -> None:
        """TypeScript files should infer typescript."""
        assert _infer_language(["src/app.ts"]) == "typescript"
        assert _infer_language(["component.tsx"]) == "typescript"

    def test_javascript_files(self) -> None:
        """JavaScript files should infer javascript."""
        assert _infer_language(["src/app.js"]) == "javascript"
        assert _infer_language(["component.jsx"]) == "javascript"

    def test_other_languages(self) -> None:
        """Other file extensions should map correctly."""
        assert _infer_language(["main.go"]) == "go"
        assert _infer_language(["lib.rs"]) == "rust"
        assert _infer_language(["App.java"]) == "java"

    def test_empty_defaults_to_python(self) -> None:
        """Empty file hints should default to python."""
        assert _infer_language([]) == "python"

    def test_first_match_wins(self) -> None:
        """First file with recognized extension wins."""
        assert _infer_language(["README.md", "app.py"]) == "markdown"


# =============================================================================
# Tests: _build_skills_section
# =============================================================================


class TestBuildSkillsSection:
    """Tests for _build_skills_section function."""

    def test_formats_skill_correctly(self) -> None:
        """Skill should be formatted with title, pattern, and rationale."""
        skills = [
            {
                "title": "Error handling",
                "pattern": "try:\n    ...\nexcept:",
                "rationale": "Prevents crashes",
                "confidence": 0.8,
            }
        ]

        section = _build_skills_section(skills)

        assert "Error handling" in section
        assert "confidence: 80%" in section
        assert "Prevents crashes" in section
        assert "try:" in section

    def test_multiple_skills(self) -> None:
        """Multiple skills should all appear in section."""
        skills = [
            {"title": "Skill 1", "pattern": "code1", "rationale": "r1", "confidence": 0.9},
            {"title": "Skill 2", "pattern": "code2", "rationale": "r2", "confidence": 0.7},
        ]

        section = _build_skills_section(skills)

        assert "Skill 1" in section
        assert "Skill 2" in section
        assert "confidence: 90%" in section
        assert "confidence: 70%" in section

    def test_empty_skills_returns_empty(self) -> None:
        """Empty skills list should return empty string."""
        assert _build_skills_section([]) == ""
        assert _build_skills_section(None) == ""  # type: ignore[arg-type]

    def test_anti_pattern_note(self) -> None:
        """Anti-patterns should include a warning note."""
        skills = [
            {
                "title": "Bad practice",
                "pattern": "eval(user_input)",
                "rationale": "Security risk",
                "confidence": 0.9,
                "type": "anti_pattern",
            }
        ]

        section = _build_skills_section(skills)

        assert "anti-pattern" in section.lower()
        assert "avoid" in section.lower()


# =============================================================================
# Tests: build_context_node with skills
# =============================================================================


class TestBuildContextNodeWithSkills:
    """Tests for skills injection in build_context_node."""

    @pytest.mark.asyncio
    async def test_injects_skills_into_state(
        self, mock_task: MockTask, skills_db: SkillsDatabase
    ) -> None:
        """skills_context should be populated in state."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "roadmap_path": "/tmp/ROADMAP.md",
        }

        result = await build_context_node(state, skills_db=skills_db, max_skills=3)

        assert "skills_context" in result
        assert len(result["skills_context"]) > 0

    @pytest.mark.asyncio
    async def test_skills_appear_in_prompt(
        self, mock_task: MockTask, skills_db: SkillsDatabase
    ) -> None:
        """Matched skills should appear in the prompt."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "roadmap_path": "/tmp/ROADMAP.md",
        }

        result = await build_context_node(state, skills_db=skills_db)

        prompt = result.get("prompt", "")
        assert "Relevant Patterns from Past Experience" in prompt
        assert "FastAPI error handling" in prompt

    @pytest.mark.asyncio
    async def test_no_skills_without_db(self, mock_task: MockTask) -> None:
        """Without skills_db, no skills should be injected."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "roadmap_path": "/tmp/ROADMAP.md",
        }

        result = await build_context_node(state, skills_db=None)

        assert result.get("skills_context", []) == []
        assert "Relevant Patterns" not in result.get("prompt", "")

    @pytest.mark.asyncio
    async def test_respects_max_skills(
        self, mock_task: MockTask, skills_db: SkillsDatabase
    ) -> None:
        """max_skills should limit number of injected skills."""
        state: dict[str, Any] = {
            "current_task": mock_task,
            "roadmap_path": "/tmp/ROADMAP.md",
        }

        result = await build_context_node(state, skills_db=skills_db, max_skills=1)

        assert len(result.get("skills_context", [])) <= 1
