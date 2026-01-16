"""Tests for skills CLI commands (Phase 8.4).

These tests verify the CLI commands work correctly by testing through
the Typer CliRunner with proper database path injection via DEFAULT_PATH patching.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ai_infra.cli.cmds.executor_cmds import app
from ai_infra.executor.skills.database import SkillsDatabase

runner = CliRunner()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_skills_data() -> dict:
    """Create sample skills data for JSON file (in database format)."""
    return {
        "version": "1.0",
        "updated_at": "2024-01-20T00:00:00+00:00",
        "skills": [
            {
                "id": "skill-001",
                "title": "Python Exception Handling",
                "description": "Best practices for handling exceptions in Python",
                "type": "pattern",
                "pattern": "try:\\n    # code\\nexcept SpecificError",
                "rationale": "Always catch specific exceptions and log before re-raising",
                "languages": ["python"],
                "frameworks": [],
                "task_keywords": ["exception", "error", "try", "catch"],
                "success_count": 5,
                "failure_count": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-15T00:00:00+00:00",
            },
            {
                "id": "skill-002",
                "title": "TypeScript Type Guards",
                "description": "Using type guards for runtime type checking",
                "type": "approach",
                "pattern": "function isUser(obj: unknown): obj is User { ... }",
                "rationale": "Type guards provide runtime safety with compile-time benefits",
                "languages": ["typescript"],
                "frameworks": ["react"],
                "task_keywords": ["type", "guard", "typescript", "runtime"],
                "success_count": 8,
                "failure_count": 0,
                "created_at": "2024-01-05T00:00:00+00:00",
                "updated_at": "2024-01-20T00:00:00+00:00",
            },
            {
                "id": "skill-003",
                "title": "Python Async Context Manager",
                "description": "Creating async context managers for resource management",
                "type": "pattern",
                "pattern": "async with resource: ...",
                "rationale": "Ensures proper cleanup of async resources",
                "languages": ["python"],
                "frameworks": ["asyncio"],
                "task_keywords": ["async", "context", "manager", "resource"],
                "success_count": 3,
                "failure_count": 2,
                "created_at": "2024-01-10T00:00:00+00:00",
                "updated_at": "2024-01-10T00:00:00+00:00",
            },
        ],
    }


@pytest.fixture
def skills_json_path(tmp_path: Path, sample_skills_data: dict) -> Path:
    """Create a JSON file with sample skills data and return its path."""
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(json.dumps(sample_skills_data))
    return skills_path


@pytest.fixture
def empty_skills_json_path(tmp_path: Path) -> Path:
    """Create an empty skills path (no file exists)."""
    return tmp_path / "skills.json"


# =============================================================================
# skills-list Command Tests
# =============================================================================


class TestSkillsListCommand:
    """Tests for the skills-list command."""

    def test_list_all_skills(self, skills_json_path: Path):
        """Should list all skills when no filters applied."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-list"])

        assert result.exit_code == 0
        assert "Learned Skills" in result.output
        assert "Python Exception" in result.output
        assert "TypeScript Type" in result.output
        assert "Python Async" in result.output

    def test_list_skills_empty_database(self, empty_skills_json_path: Path):
        """Should show message when no skills exist."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", empty_skills_json_path):
            result = runner.invoke(app, ["skills-list"])

        assert result.exit_code == 0
        assert "No skills learned yet" in result.output

    def test_list_skills_filter_by_language(self, skills_json_path: Path):
        """Should filter skills by language."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-list", "--language", "python"])

        assert result.exit_code == 0
        assert "Python Exception" in result.output
        assert "Python Async" in result.output
        # TypeScript skill should not appear
        assert "TypeScript" not in result.output

    def test_list_skills_filter_by_language_no_match(self, skills_json_path: Path):
        """Should show message when no skills match language filter."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-list", "--language", "rust"])

        assert result.exit_code == 0
        assert "No skills found for language: rust" in result.output

    def test_list_skills_with_limit(self, skills_json_path: Path):
        """Should respect the limit option."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-list", "--limit", "1"])

        assert result.exit_code == 0
        # Should show table title with total but only 1 row
        assert "3 total" in result.output

    def test_list_skills_shows_statistics(self, skills_json_path: Path):
        """Should display summary statistics."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-list"])

        assert result.exit_code == 0
        assert "Average confidence" in result.output
        assert "Total uses" in result.output


# =============================================================================
# skills-show Command Tests
# =============================================================================


class TestSkillsShowCommand:
    """Tests for the skills-show command."""

    def test_show_skill_by_exact_id(self, skills_json_path: Path):
        """Should display full skill details by exact ID."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-show", "skill-001"])

        assert result.exit_code == 0
        assert "Python Exception Handling" in result.output
        assert "skill-001" in result.output
        assert "pattern" in result.output.lower()
        assert "Description" in result.output
        assert "Pattern" in result.output

    def test_show_skill_by_partial_id(self, skills_json_path: Path):
        """Should find skill by partial ID match."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-show", "skill-002"])

        assert result.exit_code == 0
        assert "TypeScript Type Guards" in result.output

    def test_show_skill_not_found(self, skills_json_path: Path):
        """Should show error when skill not found."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-show", "nonexistent"])

        assert result.exit_code == 1
        assert "Skill not found" in result.output
        assert "skills-list" in result.output

    def test_show_skill_displays_all_fields(self, skills_json_path: Path):
        """Should display all relevant skill fields."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-show", "skill-001"])

        assert result.exit_code == 0
        # Check all major fields are displayed
        assert "ID:" in result.output
        assert "Type:" in result.output
        assert "Languages:" in result.output
        assert "Keywords:" in result.output
        assert "Confidence:" in result.output
        assert "Success/Failure:" in result.output
        assert "Created:" in result.output
        assert "Updated:" in result.output

    def test_show_skill_with_rationale(self, skills_json_path: Path):
        """Should display skill rationale when present."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-show", "skill-001"])

        assert result.exit_code == 0
        assert "Rationale" in result.output
        assert "Always catch specific exceptions" in result.output


# =============================================================================
# skills-clear Command Tests
# =============================================================================


class TestSkillsClearCommand:
    """Tests for the skills-clear command."""

    def test_clear_requires_confirm_flag(self, skills_json_path: Path):
        """Should require --confirm flag to clear skills."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-clear"])

        assert result.exit_code == 1
        assert "permanently delete" in result.output
        assert "--confirm" in result.output

    def test_clear_with_confirm_flag(self, skills_json_path: Path):
        """Should clear all skills when --confirm is provided."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            result = runner.invoke(app, ["skills-clear", "--confirm"])

        assert result.exit_code == 0
        assert "Cleared 3 skill(s)" in result.output

    def test_clear_empty_database(self, empty_skills_json_path: Path):
        """Should handle clearing an empty database gracefully."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", empty_skills_json_path):
            result = runner.invoke(app, ["skills-clear", "--confirm"])

        assert result.exit_code == 0
        assert "No skills to clear" in result.output


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkillsCliIntegration:
    """Integration tests for skills CLI workflows."""

    def test_workflow_list_show_clear(self, tmp_path: Path, sample_skills_data: dict):
        """Test full workflow: list, show, clear."""
        # Create a single skill for clarity using the proper database format
        single_skill_data = {
            "version": "1.0",
            "updated_at": "2024-01-20T00:00:00+00:00",
            "skills": [
                {
                    "id": "test-skill-123",
                    "title": "Test Skill",
                    "description": "A test skill for integration testing",
                    "type": "pattern",
                    "pattern": "test pattern code",
                    "rationale": "Test rationale",
                    "languages": ["python"],
                    "frameworks": [],
                    "task_keywords": ["test"],
                    "success_count": 5,
                    "failure_count": 1,
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-15T00:00:00+00:00",
                }
            ],
        }

        skills_path = tmp_path / "skills.json"
        skills_path.write_text(json.dumps(single_skill_data))

        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_path):
            # List skills
            result = runner.invoke(app, ["skills-list"])
            assert result.exit_code == 0
            assert "Test Skill" in result.output

            # Show skill
            result = runner.invoke(app, ["skills-show", "test-skill-123"])
            assert result.exit_code == 0
            assert "Test Skill" in result.output

            # Clear skills
            result = runner.invoke(app, ["skills-clear", "--confirm"])
            assert result.exit_code == 0
            assert "Cleared 1 skill(s)" in result.output

    def test_language_filter_separates_correctly(self, skills_json_path: Path):
        """Test that language filter correctly separates skills."""
        with patch.object(SkillsDatabase, "DEFAULT_PATH", skills_json_path):
            # Python skills
            result = runner.invoke(app, ["skills-list", "--language", "python"])
            assert result.exit_code == 0
            assert "python" in result.output.lower()
            assert "typescript" not in result.output.lower()

            # TypeScript skills
            result = runner.invoke(app, ["skills-list", "--language", "typescript"])
            assert result.exit_code == 0
            assert "typescript" in result.output.lower()
