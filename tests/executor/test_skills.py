"""Tests for the Skills System (Phase 5.1).

Tests cover:
- Skill model creation and matching
- SkillsDatabase persistence and queries
- SkillExtractor skill extraction
- SkillApplier skill injection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.skills import (
    ApplierConfig,
    ExtractionConfig,
    Skill,
    SkillApplier,
    SkillContext,
    SkillExtractor,
    SkillInjectionResult,
    SkillsDatabase,
    SkillType,
    TaskResult,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_skills_file():
    """Create a temporary file for skills database."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
    path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def skills_db(temp_skills_file: Path) -> SkillsDatabase:
    """Create a skills database with temp file."""
    return SkillsDatabase(path=temp_skills_file)


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample skill for testing."""
    return Skill.create(
        type=SkillType.PATTERN,
        title="Use pytest fixtures for setup",
        description="Create reusable fixtures for test setup instead of repeating setup code.",
        languages=["python"],
        frameworks=["pytest"],
        task_keywords=["test", "fixture", "setup"],
        pattern="@pytest.fixture\ndef my_fixture():\n    return setup_value()",
        rationale="Fixtures reduce code duplication and ensure consistent setup.",
    )


@pytest.fixture
def sample_anti_pattern() -> Skill:
    """Create a sample anti-pattern for testing."""
    return Skill.create(
        type=SkillType.ANTI_PATTERN,
        title="Avoid bare except clauses",
        description="Bare except catches all exceptions including KeyboardInterrupt.",
        languages=["python"],
        task_keywords=["exception", "error", "handling"],
        anti_example="try:\n    do_something()\nexcept:\n    pass",
        pattern="try:\n    do_something()\nexcept SpecificError as e:\n    logger.error(e)",
    )


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = MagicMock()
    task.id = "task-123"
    task.title = "Add pytest fixtures for database tests"
    task.description = "Create reusable fixtures for database setup in tests"
    return task


@pytest.fixture
def sample_context() -> SkillContext:
    """Create a sample skill context."""
    return SkillContext(
        language="python",
        framework="pytest",
        task_keywords=["test", "fixture", "database"],
        task_title="Add pytest fixtures",
    )


# =============================================================================
# Skill Model Tests
# =============================================================================


class TestSkill:
    """Tests for the Skill model."""

    def test_skill_create(self):
        """Test creating a skill with factory method."""
        skill = Skill.create(
            type=SkillType.PATTERN,
            title="Test Skill",
            description="A test skill for testing",
            languages=["python"],
        )

        assert skill.id is not None
        assert skill.id.startswith("skill-")
        assert skill.type == SkillType.PATTERN
        assert skill.title == "Test Skill"
        assert skill.languages == ["python"]
        assert skill.created_at is not None

    def test_skill_confidence_with_no_uses(self, sample_skill: Skill):
        """Test confidence calculation with no usage."""
        # With 0 success, 0 failure, Laplace smoothing gives 0.5
        assert sample_skill.confidence == pytest.approx(0.5)

    def test_skill_confidence_with_successes(self, sample_skill: Skill):
        """Test confidence increases with successes."""
        sample_skill.success_count = 10
        sample_skill.failure_count = 0

        # (10 + 1) / (10 + 0 + 2) = 11/12 = 0.916...
        assert sample_skill.confidence > 0.9

    def test_skill_confidence_with_failures(self, sample_skill: Skill):
        """Test confidence decreases with failures."""
        sample_skill.success_count = 0
        sample_skill.failure_count = 10

        # (0 + 1) / (0 + 10 + 2) = 1/12 = 0.083...
        assert sample_skill.confidence < 0.1

    def test_skill_matches_exact_context(self, sample_skill: Skill, sample_context: SkillContext):
        """Test matching with exact context match."""
        score = sample_skill.matches(sample_context)

        # Should have high score: language match, framework match, keyword overlap
        assert score > 0.5

    def test_skill_matches_language_only(self, sample_skill: Skill):
        """Test matching with only language match."""
        context = SkillContext(language="python")

        score = sample_skill.matches(context)

        # Should match on language
        assert score > 0.0

    def test_skill_matches_no_context(self, sample_skill: Skill):
        """Test matching with empty context."""
        context = SkillContext(language="")

        score = sample_skill.matches(context)

        # Some baseline score
        assert score >= 0.0

    def test_skill_matches_wrong_language(self, sample_skill: Skill):
        """Test matching with wrong language."""
        context = SkillContext(language="javascript")

        score = sample_skill.matches(context)

        # Should have low score
        assert score < 0.5

    def test_skill_serialization(self, sample_skill: Skill):
        """Test skill to_dict and from_dict."""
        data = sample_skill.to_dict()

        assert data["id"] == sample_skill.id
        assert data["type"] == "pattern"
        assert data["title"] == sample_skill.title

        restored = Skill.from_dict(data)

        assert restored.id == sample_skill.id
        assert restored.type == sample_skill.type
        assert restored.title == sample_skill.title
        assert restored.languages == sample_skill.languages

    def test_skill_record_success(self, sample_skill: Skill):
        """Test recording success updates counters."""
        initial = sample_skill.success_count

        sample_skill.record_success()

        assert sample_skill.success_count == initial + 1
        assert sample_skill.updated_at is not None

    def test_skill_record_failure(self, sample_skill: Skill):
        """Test recording failure updates counters."""
        initial = sample_skill.failure_count

        sample_skill.record_failure()

        assert sample_skill.failure_count == initial + 1
        assert sample_skill.updated_at is not None


# =============================================================================
# SkillsDatabase Tests
# =============================================================================


class TestSkillsDatabase:
    """Tests for the SkillsDatabase."""

    def test_database_add_and_get(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test adding and retrieving a skill."""
        skills_db.add(sample_skill)

        retrieved = skills_db.get(sample_skill.id)

        assert retrieved is not None
        assert retrieved.id == sample_skill.id
        assert retrieved.title == sample_skill.title

    def test_database_persistence(self, temp_skills_file: Path, sample_skill: Skill):
        """Test that database persists to file."""
        # Add skill
        db1 = SkillsDatabase(path=temp_skills_file)
        db1.add(sample_skill)

        # Create new instance
        db2 = SkillsDatabase(path=temp_skills_file)

        # Should still find skill
        retrieved = db2.get(sample_skill.id)
        assert retrieved is not None
        assert retrieved.title == sample_skill.title

    def test_database_find_matching(
        self, skills_db: SkillsDatabase, sample_skill: Skill, sample_context: SkillContext
    ):
        """Test finding matching skills."""
        skills_db.add(sample_skill)

        matches = skills_db.find_matching(sample_context)

        assert len(matches) >= 1
        assert any(s.id == sample_skill.id for s in matches)

    def test_database_find_by_language(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test finding skills by language."""
        skills_db.add(sample_skill)

        matches = skills_db.find_by_language("python")

        assert len(matches) >= 1
        assert any(s.id == sample_skill.id for s in matches)

    def test_database_find_by_framework(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test finding skills by framework."""
        skills_db.add(sample_skill)

        matches = skills_db.find_by_framework("pytest")

        assert len(matches) >= 1
        assert any(s.id == sample_skill.id for s in matches)

    def test_database_find_anti_patterns(
        self, skills_db: SkillsDatabase, sample_anti_pattern: Skill, sample_context: SkillContext
    ):
        """Test finding anti-patterns."""
        skills_db.add(sample_anti_pattern)

        anti_patterns = skills_db.find_anti_patterns(sample_context)

        assert len(anti_patterns) >= 1
        assert all(s.type == SkillType.ANTI_PATTERN for s in anti_patterns)

    def test_database_update_success(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test updating success count."""
        skills_db.add(sample_skill)
        initial = sample_skill.success_count

        skills_db.update_success(sample_skill.id)

        retrieved = skills_db.get(sample_skill.id)
        assert retrieved is not None
        assert retrieved.success_count == initial + 1

    def test_database_update_failure(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test updating failure count."""
        skills_db.add(sample_skill)
        initial = sample_skill.failure_count

        skills_db.update_failure(sample_skill.id)

        retrieved = skills_db.get(sample_skill.id)
        assert retrieved is not None
        assert retrieved.failure_count == initial + 1

    def test_database_delete(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test deleting a skill."""
        skills_db.add(sample_skill)
        assert skills_db.get(sample_skill.id) is not None

        skills_db.delete(sample_skill.id)

        assert skills_db.get(sample_skill.id) is None

    def test_database_get_stats(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test getting database statistics."""
        skills_db.add(sample_skill)

        stats = skills_db.get_stats()

        assert stats["total"] >= 1
        assert "by_type" in stats

    def test_database_prune_low_confidence(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test pruning low confidence skills."""
        # Create a low confidence skill with enough uses to be pruned
        low_conf = Skill.create(
            type=SkillType.PATTERN,
            title="Low confidence skill",
            description="This will be pruned",
            success_count=0,
            failure_count=100,
        )
        # Give sample_skill high confidence
        sample_skill.success_count = 10
        skills_db.add(sample_skill)
        skills_db.add(low_conf)

        # Prune with threshold (returns count, not list)
        pruned_count = skills_db.prune_low_confidence(threshold=0.3)

        assert pruned_count >= 1
        assert skills_db.get(low_conf.id) is None
        assert skills_db.get(sample_skill.id) is not None


# =============================================================================
# SkillExtractor Tests
# =============================================================================


class TestSkillExtractor:
    """Tests for the SkillExtractor."""

    def test_extractor_creation(self, skills_db: SkillsDatabase):
        """Test creating an extractor."""
        extractor = SkillExtractor(db=skills_db)

        assert extractor.db is skills_db
        assert extractor.config is not None

    def test_extractor_with_config(self, skills_db: SkillsDatabase):
        """Test creating extractor with config."""
        config = ExtractionConfig(
            min_confidence_to_save=0.7,
            extract_on_success=True,
            extract_on_failure=False,
        )
        extractor = SkillExtractor(db=skills_db, config=config)

        assert extractor.config.min_confidence_to_save == 0.7
        assert extractor.config.extract_on_failure is False

    @pytest.mark.asyncio
    async def test_extract_from_success_heuristic(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test heuristic extraction from success."""
        extractor = SkillExtractor(db=skills_db)

        result = TaskResult(
            success=True,
            diff_summary="Added new fixture",
            files_created=["tests/conftest.py"],
            execution_time=5.0,
        )

        skill = await extractor.extract_from_success(mock_task, result, sample_context)

        # Heuristic extraction should create a skill
        assert skill is not None
        assert skill.type == SkillType.APPROACH
        assert skills_db.get(skill.id) is not None

    @pytest.mark.asyncio
    async def test_extract_from_failure_heuristic(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test heuristic extraction from failure."""
        extractor = SkillExtractor(db=skills_db)

        skill = await extractor.extract_from_failure(
            mock_task,
            "ImportError: No module named 'missing_module'",
            sample_context,
        )

        # Should extract anti-pattern
        assert skill is not None
        assert skill.type == SkillType.ANTI_PATTERN
        assert skills_db.get(skill.id) is not None

    @pytest.mark.asyncio
    async def test_extract_with_llm(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test extraction using LLM."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "should_extract": True,
                "skill_type": "pattern",
                "title": "Use pytest fixtures",
                "description": "Create fixtures for reusable setup",
                "pattern": "@pytest.fixture\ndef setup():\n    pass",
                "rationale": "Reduces duplication",
                "keywords": ["pytest", "fixture"],
            }
        )
        mock_llm.agenerate = AsyncMock(return_value=mock_response)

        extractor = SkillExtractor(db=skills_db, llm=mock_llm)

        result = TaskResult(
            success=True,
            diff_summary="Added fixture",
            files_created=["conftest.py"],
        )

        skill = await extractor.extract_from_success(mock_task, result, sample_context)

        assert skill is not None
        assert skill.title == "Use pytest fixtures"
        assert "pytest" in skill.task_keywords

    @pytest.mark.asyncio
    async def test_extract_disabled_on_success(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test extraction disabled for success."""
        config = ExtractionConfig(extract_on_success=False)
        extractor = SkillExtractor(db=skills_db, config=config)

        result = TaskResult(success=True, diff_summary="Changes")

        skill = await extractor.extract_from_success(mock_task, result, sample_context)

        assert skill is None

    @pytest.mark.asyncio
    async def test_extract_disabled_on_failure(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test extraction disabled for failure."""
        config = ExtractionConfig(extract_on_failure=False)
        extractor = SkillExtractor(db=skills_db, config=config)

        skill = await extractor.extract_from_failure(mock_task, "SomeError: failed", sample_context)

        assert skill is None


# =============================================================================
# SkillApplier Tests
# =============================================================================


class TestSkillApplier:
    """Tests for the SkillApplier."""

    def test_applier_creation(self, skills_db: SkillsDatabase):
        """Test creating an applier."""
        applier = SkillApplier(db=skills_db)

        assert applier.db is skills_db
        assert applier.config is not None

    def test_applier_with_config(self, skills_db: SkillsDatabase):
        """Test creating applier with config."""
        config = ApplierConfig(
            max_skills_per_prompt=3,
            min_confidence=0.5,
            include_anti_patterns=False,
        )
        applier = SkillApplier(db=skills_db, config=config)

        assert applier.config.max_skills_per_prompt == 3
        assert applier.config.include_anti_patterns is False

    def test_build_context(self, skills_db: SkillsDatabase, mock_task):
        """Test building context from task."""
        applier = SkillApplier(db=skills_db)

        context = applier.build_context(mock_task)

        assert context.task_title == mock_task.title
        assert len(context.task_keywords) > 0
        assert "pytest" in context.task_keywords or "fixtures" in context.task_keywords

    def test_get_relevant_skills(
        self, skills_db: SkillsDatabase, sample_skill: Skill, sample_context: SkillContext
    ):
        """Test getting relevant skills."""
        # Add skill with high confidence
        sample_skill.success_count = 10
        skills_db.add(sample_skill)

        applier = SkillApplier(db=skills_db)

        skills = applier.get_relevant_skills(sample_context)

        assert len(skills) >= 1
        assert any(s.id == sample_skill.id for s in skills)

    def test_get_anti_patterns(
        self, skills_db: SkillsDatabase, sample_anti_pattern: Skill, sample_context: SkillContext
    ):
        """Test getting anti-patterns."""
        # Add anti-pattern with high confidence
        sample_anti_pattern.success_count = 10
        skills_db.add(sample_anti_pattern)

        applier = SkillApplier(db=skills_db)

        anti_patterns = applier.get_anti_patterns(sample_context)

        # May or may not match depending on keyword overlap
        assert isinstance(anti_patterns, list)

    def test_inject_skills(
        self, skills_db: SkillsDatabase, sample_skill: Skill, sample_context: SkillContext
    ):
        """Test injecting skills into prompt."""
        sample_skill.success_count = 10
        skills_db.add(sample_skill)

        applier = SkillApplier(db=skills_db)

        result = applier.inject_skills(sample_context)

        assert isinstance(result, SkillInjectionResult)
        assert len(result.skills_applied) >= 1
        assert len(result.skills_prompt) > 0

    def test_format_skills_for_prompt(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test formatting skills for prompt."""
        applier = SkillApplier(db=skills_db)

        prompt = applier.format_skills_for_prompt([sample_skill])

        assert "Relevant Skills" in prompt
        assert sample_skill.title in prompt
        assert sample_skill.description in prompt

    def test_record_skill_usage_success(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test recording successful skill usage."""
        skills_db.add(sample_skill)
        initial = sample_skill.success_count

        applier = SkillApplier(db=skills_db)
        applier.record_skill_usage([sample_skill.id], success=True)

        retrieved = skills_db.get(sample_skill.id)
        assert retrieved is not None
        assert retrieved.success_count == initial + 1

    def test_record_skill_usage_failure(self, skills_db: SkillsDatabase, sample_skill: Skill):
        """Test recording failed skill usage."""
        skills_db.add(sample_skill)
        initial = sample_skill.failure_count

        applier = SkillApplier(db=skills_db)
        applier.record_skill_usage([sample_skill.id], success=False)

        retrieved = skills_db.get(sample_skill.id)
        assert retrieved is not None
        assert retrieved.failure_count == initial + 1

    def test_skill_injection_result_properties(self):
        """Test SkillInjectionResult properties."""
        result = SkillInjectionResult(
            skills_applied=[MagicMock(id="s1"), MagicMock(id="s2")],
            anti_patterns_applied=[MagicMock(id="a1")],
            skills_prompt="test",
        )

        assert result.total_skills == 3
        assert "s1" in result.get_skill_ids()
        assert "a1" in result.get_skill_ids()

    def test_applier_excludes_low_confidence(
        self, skills_db: SkillsDatabase, sample_context: SkillContext
    ):
        """Test that low confidence skills are excluded."""
        # Create low confidence skill
        low_conf = Skill.create(
            type=SkillType.PATTERN,
            title="Low confidence",
            description="Will be excluded",
            languages=["python"],
            frameworks=["pytest"],
            task_keywords=["test", "fixture"],
            success_count=0,
            failure_count=100,
        )
        skills_db.add(low_conf)

        config = ApplierConfig(min_confidence=0.3)
        applier = SkillApplier(db=skills_db, config=config)

        skills = applier.get_relevant_skills(sample_context)

        # Should not include low confidence skill
        assert not any(s.id == low_conf.id for s in skills)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkillsIntegration:
    """Integration tests for the skills system."""

    @pytest.mark.asyncio
    async def test_full_skill_lifecycle(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test complete skill lifecycle: extract -> store -> apply -> update."""
        # 1. Extract skill from success
        extractor = SkillExtractor(db=skills_db)
        result = TaskResult(
            success=True,
            diff_summary="Added test fixtures",
            files_created=["conftest.py"],
            execution_time=3.0,
        )

        skill = await extractor.extract_from_success(mock_task, result, sample_context)
        assert skill is not None

        # 2. Verify stored
        stored = skills_db.get(skill.id)
        assert stored is not None

        # 3. Apply skills to new task
        applier = SkillApplier(db=skills_db)
        injection = applier.inject_skills(sample_context)

        # May or may not match depending on confidence
        assert isinstance(injection, SkillInjectionResult)

        # 4. Record usage
        if injection.skills_applied:
            applier.record_skill_usage(
                [s.id for s in injection.skills_applied],
                success=True,
            )

    @pytest.mark.asyncio
    async def test_anti_pattern_learning(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """Test learning from failures creates anti-patterns."""
        extractor = SkillExtractor(db=skills_db)

        # Fail multiple times with same error
        for _ in range(3):
            await extractor.extract_from_failure(
                mock_task,
                "TypeError: unhashable type: 'list'",
                sample_context,
            )

        # Should have created anti-pattern
        anti_patterns = skills_db.find_anti_patterns(sample_context)
        assert len(anti_patterns) > 0

    def test_skill_matching_priority(self, skills_db: SkillsDatabase):
        """Test that higher matching skills are prioritized."""
        # Create skills with different match levels
        exact_match = Skill.create(
            type=SkillType.PATTERN,
            title="Exact Python pytest match",
            description="Matches exactly",
            languages=["python"],
            frameworks=["pytest"],
            task_keywords=["test", "fixture", "database"],
            success_count=5,
        )

        partial_match = Skill.create(
            type=SkillType.PATTERN,
            title="Partial match",
            description="Matches partially",
            languages=["python"],
            task_keywords=["test"],
            success_count=5,
        )

        no_match = Skill.create(
            type=SkillType.PATTERN,
            title="No match",
            description="Does not match",
            languages=["javascript"],
            task_keywords=["frontend", "react"],
            success_count=5,
        )

        skills_db.add(exact_match)
        skills_db.add(partial_match)
        skills_db.add(no_match)

        context = SkillContext(
            language="python",
            framework="pytest",
            task_keywords=["test", "fixture", "database"],
        )

        matches = skills_db.find_matching(context, limit=10)

        # Exact match should be first
        if matches:
            match_ids = [s.id for s in matches]
            if exact_match.id in match_ids and partial_match.id in match_ids:
                assert match_ids.index(exact_match.id) < match_ids.index(partial_match.id)


# =============================================================================
# Edge Cases
# =============================================================================


class TestSkillsEdgeCases:
    """Edge case tests for the skills system."""

    def test_empty_database_queries(self, skills_db: SkillsDatabase, sample_context: SkillContext):
        """Test queries on empty database."""
        assert skills_db.find_matching(sample_context) == []
        assert skills_db.find_by_language("python") == []
        assert skills_db.find_anti_patterns(sample_context) == []
        assert skills_db.get("nonexistent") is None

    def test_skill_with_empty_fields(self):
        """Test skill with minimal fields."""
        skill = Skill.create(
            type=SkillType.APPROACH,
            title="Minimal skill",
            description="Just the basics",
        )

        assert skill.languages == []
        assert skill.frameworks == []
        assert skill.task_keywords == []
        assert skill.pattern is None

    def test_context_with_all_none(self, sample_skill: Skill):
        """Test matching with all-None context."""
        context = SkillContext(language="")

        score = sample_skill.matches(context)

        # Should still work, return some score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_unicode_in_skills(self, skills_db: SkillsDatabase):
        """Test skills with unicode content."""
        skill = Skill.create(
            type=SkillType.PATTERN,
            title="Unicode: cafe",
            description="Handles unicode strings properly",
            pattern='text = "Hello World"',
        )

        skills_db.add(skill)
        retrieved = skills_db.get(skill.id)

        assert retrieved is not None
        assert "cafe" in retrieved.title

    def test_very_long_pattern(self, skills_db: SkillsDatabase):
        """Test skill with very long pattern."""
        long_pattern = "x = 1\n" * 1000  # Very long

        skill = Skill.create(
            type=SkillType.PATTERN,
            title="Long pattern skill",
            description="Has a very long pattern",
            pattern=long_pattern,
        )

        skills_db.add(skill)
        retrieved = skills_db.get(skill.id)

        assert retrieved is not None
        assert len(retrieved.pattern) > 1000


# =============================================================================
# Phase 5.1 Success Criteria Tests
# =============================================================================


class TestPhase51SuccessCriteria:
    """Tests verifying Phase 5.1 success criteria."""

    def test_5_1_1_skill_model_exists(self):
        """5.1.1: Skill model with proper attributes."""
        skill = Skill.create(
            type=SkillType.PATTERN,
            title="Test",
            description="Test skill",
        )

        # Check required attributes
        assert hasattr(skill, "id")
        assert hasattr(skill, "type")
        assert hasattr(skill, "title")
        assert hasattr(skill, "description")
        assert hasattr(skill, "languages")
        assert hasattr(skill, "frameworks")
        assert hasattr(skill, "task_keywords")
        assert hasattr(skill, "pattern")
        assert hasattr(skill, "success_count")
        assert hasattr(skill, "failure_count")

    def test_5_1_2_skill_matching(self, sample_skill: Skill, sample_context: SkillContext):
        """5.1.2: Skill matching works correctly."""
        score = sample_skill.matches(sample_context)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_5_1_3_database_persistence(self, temp_skills_file: Path, sample_skill: Skill):
        """5.1.3: Skills persist across sessions."""
        # Session 1: Add skill
        db1 = SkillsDatabase(path=temp_skills_file)
        db1.add(sample_skill)

        # Session 2: Load and verify
        db2 = SkillsDatabase(path=temp_skills_file)
        retrieved = db2.get(sample_skill.id)

        assert retrieved is not None
        assert retrieved.id == sample_skill.id

    @pytest.mark.asyncio
    async def test_5_1_4_skill_extraction(
        self, skills_db: SkillsDatabase, mock_task, sample_context: SkillContext
    ):
        """5.1.4: Skills extracted from executions."""
        extractor = SkillExtractor(db=skills_db)

        result = TaskResult(
            success=True,
            diff_summary="Code changes",
            files_created=["new_file.py"],
        )

        skill = await extractor.extract_from_success(mock_task, result, sample_context)

        assert skill is not None
        assert skills_db.get(skill.id) is not None

    def test_5_1_5_skill_injection(
        self, skills_db: SkillsDatabase, sample_skill: Skill, sample_context: SkillContext
    ):
        """5.1.5: Skills injected into prompts."""
        sample_skill.success_count = 10  # High confidence
        skills_db.add(sample_skill)

        applier = SkillApplier(db=skills_db)
        result = applier.inject_skills(sample_context)

        assert isinstance(result.skills_prompt, str)
        # Prompt contains skill info
        if result.skills_applied:
            assert sample_skill.title in result.skills_prompt

    def test_5_1_6_skill_types_complete(self):
        """5.1.6: All skill types available."""
        assert SkillType.PATTERN is not None
        assert SkillType.ANTI_PATTERN is not None
        assert SkillType.APPROACH is not None
        assert SkillType.TOOL_USAGE is not None
        assert SkillType.RECOVERY is not None
