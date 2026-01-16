"""Tests for skill matching (Phase 8.5.1).

This module tests the SkillsDatabase.find_matching functionality
to ensure that skills are correctly matched to task contexts.

Tests cover:
- Exact matches on language/framework/keywords
- Partial matches with scoring
- Match score boosting by confidence
- No matches when context doesn't align
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_infra.executor.skills.database import SkillsDatabase
from ai_infra.executor.skills.models import Skill, SkillContext, SkillType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def skills_db(tmp_path: Any) -> SkillsDatabase:
    """Create an empty skills database."""
    db_path = tmp_path / "skills.json"
    return SkillsDatabase(str(db_path), auto_save=False)


@pytest.fixture
def fastapi_auth_skill() -> Skill:
    """Create a FastAPI authentication skill."""
    return Skill.create(
        type=SkillType.PATTERN,
        title="FastAPI auth pattern",
        description="JWT-based authentication for FastAPI endpoints",
        languages=["python"],
        frameworks=["fastapi"],
        task_keywords=["auth", "login", "jwt", "authentication"],
        pattern="""@app.post('/login')
async def login(credentials: LoginRequest):
    user = await authenticate(credentials)
    if not user:
        raise HTTPException(status_code=401)
    token = create_jwt_token(user)
    return {"access_token": token}""",
        rationale="Consistent auth pattern with proper error handling",
    )


@pytest.fixture
def pytest_fixture_skill() -> Skill:
    """Create a pytest fixture skill."""
    return Skill.create(
        type=SkillType.PATTERN,
        title="Pytest database fixtures",
        description="Reusable fixtures for database setup",
        languages=["python"],
        frameworks=["pytest", "sqlalchemy"],
        task_keywords=["test", "fixture", "database", "setup"],
        pattern="""@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    yield Session()
    engine.dispose()""",
        rationale="Isolated database per test prevents flaky tests",
    )


@pytest.fixture
def react_hooks_skill() -> Skill:
    """Create a React hooks skill."""
    return Skill.create(
        type=SkillType.PATTERN,
        title="React custom hooks pattern",
        description="Creating reusable custom hooks",
        languages=["typescript", "javascript"],
        frameworks=["react"],
        task_keywords=["hook", "custom", "state", "effect"],
        pattern="""function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState(value);
    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);
    return debouncedValue;
}""",
        rationale="Custom hooks encapsulate reusable stateful logic",
    )


@pytest.fixture
def anti_pattern_skill() -> Skill:
    """Create an anti-pattern skill for bare except."""
    return Skill.create(
        type=SkillType.ANTI_PATTERN,
        title="Avoid bare except clauses",
        description="Bare except catches all exceptions including system exits",
        languages=["python"],
        frameworks=[],
        task_keywords=["exception", "error", "handling", "except"],
        pattern="""try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise""",
        anti_example="""try:
    risky_operation()
except:
    pass""",
        rationale="Bare except catches KeyboardInterrupt and SystemExit",
    )


@pytest.fixture
def populated_db(
    skills_db: SkillsDatabase,
    fastapi_auth_skill: Skill,
    pytest_fixture_skill: Skill,
    react_hooks_skill: Skill,
    anti_pattern_skill: Skill,
) -> SkillsDatabase:
    """Create a database populated with various skills."""
    skills_db.add(fastapi_auth_skill)
    skills_db.add(pytest_fixture_skill)
    skills_db.add(react_hooks_skill)
    skills_db.add(anti_pattern_skill)
    return skills_db


# =============================================================================
# Tests: Basic Matching
# =============================================================================


class TestSkillMatching:
    """Tests for SkillsDatabase.find_matching."""

    def test_finds_matching_skill_by_language_framework_keywords(
        self, populated_db: SkillsDatabase
    ) -> None:
        """Should find skill that matches language, framework, and keywords."""
        ctx = SkillContext(
            language="python",
            framework="fastapi",
            task_keywords=["auth", "user"],
            task_title="Add user authentication",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        assert len(matches) >= 1
        # FastAPI auth skill should be the top match
        titles = [m.title for m in matches]
        assert "FastAPI auth pattern" in titles

    def test_finds_multiple_relevant_skills(self, populated_db: SkillsDatabase) -> None:
        """Should find multiple skills when context matches several."""
        ctx = SkillContext(
            language="python",
            framework="pytest",
            task_keywords=["test", "database", "fixture"],
            task_title="Add database tests",
            task_description="Create unit tests for database models",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        assert len(matches) >= 1
        titles = [m.title for m in matches]
        assert "Pytest database fixtures" in titles

    def test_no_matches_for_unrelated_context(self, populated_db: SkillsDatabase) -> None:
        """Should return empty list when no skills match context."""
        ctx = SkillContext(
            language="rust",
            framework="actix",
            task_keywords=["websocket", "streaming"],
            task_title="Add websocket support",
            task_description="Implement real-time communication",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        # No skills match Rust/Actix
        assert len(matches) == 0

    def test_respects_limit_parameter(self, populated_db: SkillsDatabase) -> None:
        """Should respect the limit parameter."""
        ctx = SkillContext(
            language="python",
            task_keywords=["error", "exception"],
            task_title="Handle errors",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=1)

        assert len(matches) <= 1

    def test_respects_min_score_parameter(self, populated_db: SkillsDatabase) -> None:
        """Should filter out skills below min_score."""
        ctx = SkillContext(
            language="python",
            task_keywords=["random", "unrelated"],
            task_title="Random task",
            task_description="",
        )

        # With high min_score, should get no matches
        matches = populated_db.find_matching(ctx, limit=10, min_score=0.9)

        assert len(matches) == 0


# =============================================================================
# Tests: Score Boosting by Confidence
# =============================================================================


class TestConfidenceBoost:
    """Tests for confidence-based score boosting."""

    def test_higher_confidence_ranks_higher(self, skills_db: SkillsDatabase) -> None:
        """Skills with higher confidence should rank higher."""
        # Create two similar skills with different confidence
        high_conf = Skill.create(
            type=SkillType.PATTERN,
            title="High confidence auth",
            description="Auth pattern with high success rate",
            languages=["python"],
            frameworks=["fastapi"],
            task_keywords=["auth", "login"],
            pattern="...",
        )
        high_conf.success_count = 10
        high_conf.failure_count = 0

        low_conf = Skill.create(
            type=SkillType.PATTERN,
            title="Low confidence auth",
            description="Auth pattern with low success rate",
            languages=["python"],
            frameworks=["fastapi"],
            task_keywords=["auth", "login"],
            pattern="...",
        )
        low_conf.success_count = 1
        low_conf.failure_count = 5

        skills_db.add(high_conf)
        skills_db.add(low_conf)

        ctx = SkillContext(
            language="python",
            framework="fastapi",
            task_keywords=["auth"],
            task_title="Add authentication",
            task_description="",
        )

        matches = skills_db.find_matching(ctx, limit=2)

        assert len(matches) == 2
        # High confidence should be first
        assert matches[0].title == "High confidence auth"
        assert matches[1].title == "Low confidence auth"


# =============================================================================
# Tests: Language and Framework Matching
# =============================================================================


class TestLanguageFrameworkMatching:
    """Tests for language and framework specific matching."""

    def test_matches_typescript_skill_for_typescript_context(
        self, populated_db: SkillsDatabase
    ) -> None:
        """TypeScript context should match TypeScript skills."""
        ctx = SkillContext(
            language="typescript",
            framework="react",
            task_keywords=["hook", "state"],
            task_title="Create custom hook",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        assert len(matches) >= 1
        titles = [m.title for m in matches]
        assert "React custom hooks pattern" in titles

    def test_language_mismatch_reduces_score(self, populated_db: SkillsDatabase) -> None:
        """Different language should significantly reduce match score."""
        ctx = SkillContext(
            language="go",
            framework="fastapi",  # Wrong framework for Go
            task_keywords=["auth", "login"],
            task_title="Add authentication",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=5, min_score=0.5)

        # Should not find Python FastAPI auth with Go language and high min_score
        fastapi_found = any("FastAPI" in m.title for m in matches)
        assert not fastapi_found


# =============================================================================
# Tests: Keyword Matching
# =============================================================================


class TestKeywordMatching:
    """Tests for keyword-based skill matching."""

    def test_exact_keyword_match(self, populated_db: SkillsDatabase) -> None:
        """Exact keyword matches should find relevant skills."""
        ctx = SkillContext(
            language="python",
            task_keywords=["exception", "error", "handling"],
            task_title="Improve error handling",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        assert len(matches) >= 1
        titles = [m.title for m in matches]
        assert "Avoid bare except clauses" in titles

    def test_partial_keyword_overlap(self, populated_db: SkillsDatabase) -> None:
        """Partial keyword overlap should still find matches."""
        ctx = SkillContext(
            language="python",
            framework="pytest",
            task_keywords=["test", "unit"],  # Only 'test' overlaps
            task_title="Add unit tests",
            task_description="",
        )

        matches = populated_db.find_matching(ctx, limit=5)

        # Should find pytest fixture skill due to 'test' keyword
        assert len(matches) >= 1


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestMatchingEdgeCases:
    """Tests for edge cases in skill matching."""

    def test_empty_database_returns_empty(self, skills_db: SkillsDatabase) -> None:
        """Empty database should return empty list."""
        ctx = SkillContext(
            language="python",
            task_keywords=["anything"],
            task_title="Any task",
            task_description="",
        )

        matches = skills_db.find_matching(ctx, limit=5)

        assert matches == []

    def test_empty_context_still_works(self, populated_db: SkillsDatabase) -> None:
        """Empty context should not crash."""
        ctx = SkillContext(
            language="",
            framework=None,
            task_keywords=[],
            task_title="",
            task_description="",
        )

        # Should not raise
        matches = populated_db.find_matching(ctx, limit=5)
        assert isinstance(matches, list)

    def test_special_characters_in_keywords(self, populated_db: SkillsDatabase) -> None:
        """Special characters in keywords should not crash."""
        ctx = SkillContext(
            language="python",
            task_keywords=["@app.post", "async/await", "try:except"],
            task_title="Handle special syntax",
            task_description="",
        )

        # Should not raise
        matches = populated_db.find_matching(ctx, limit=5)
        assert isinstance(matches, list)
