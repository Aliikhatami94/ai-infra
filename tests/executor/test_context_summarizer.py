"""Tests for ContextSummarizer (Phase 9.2).

This module tests:
- Relevance filtering based on keyword overlap
- Token budget enforcement
- Summary formatting
- Edge cases (empty inputs, no relevant tasks)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_infra.executor.context_carryover import (
    ContextSummarizer,
    _count_tokens_estimate,
    _extract_keywords_from_text,
)

# =============================================================================
# Mock Task for Testing
# =============================================================================


@dataclass
class MockTask:
    """Mock TodoItem for testing."""

    id: int
    title: str
    description: str = ""
    file_hints: list[str] = field(default_factory=list)


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestCountTokensEstimate:
    """Tests for _count_tokens_estimate."""

    def test_empty_string(self):
        """Empty string returns 0 tokens."""
        assert _count_tokens_estimate("") == 0

    def test_short_string(self):
        """Short strings are estimated correctly."""
        # 4 chars = 1 token (with ceiling)
        assert _count_tokens_estimate("test") == 1
        assert _count_tokens_estimate("ab") == 1
        assert _count_tokens_estimate("abcdef") == 2

    def test_longer_string(self):
        """Longer strings are estimated proportionally."""
        text = "a" * 100
        assert _count_tokens_estimate(text) == 25  # 100 / 4 = 25

    def test_conservative_estimation(self):
        """Estimation rounds up to be conservative."""
        # 5 chars should round up to 2 tokens
        assert _count_tokens_estimate("abcde") == 2


class TestExtractKeywordsFromText:
    """Tests for _extract_keywords_from_text."""

    def test_empty_string(self):
        """Empty string returns empty set."""
        assert _extract_keywords_from_text("") == set()

    def test_filters_stop_words(self):
        """Stop words are filtered out."""
        text = "the quick and brown fox"
        keywords = _extract_keywords_from_text(text)
        assert "the" not in keywords
        assert "and" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_filters_short_words(self):
        """Words with 2 or fewer characters are filtered."""
        text = "a to is on it"
        keywords = _extract_keywords_from_text(text)
        assert len(keywords) == 0

    def test_extracts_code_identifiers(self):
        """Extracts code-style identifiers."""
        text = "user_service UserModel get_user_by_id"
        keywords = _extract_keywords_from_text(text)
        assert "user_service" in keywords
        assert "usermodel" in keywords  # lowercase
        assert "get_user_by_id" in keywords

    def test_lowercase_normalization(self):
        """Keywords are normalized to lowercase."""
        text = "UserService CREATE DELETE"
        keywords = _extract_keywords_from_text(text)
        assert "userservice" in keywords
        assert "create" in keywords
        assert "delete" in keywords


# =============================================================================
# Test ContextSummarizer
# =============================================================================


class TestContextSummarizerInit:
    """Tests for ContextSummarizer initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        summarizer = ContextSummarizer()
        assert summarizer.max_tokens == 1000
        assert summarizer.max_tasks == 3
        assert summarizer.relevance_threshold == 0.1

    def test_custom_values(self):
        """Test custom initialization values."""
        summarizer = ContextSummarizer(
            max_tokens=500,
            max_tasks=5,
            relevance_threshold=0.2,
        )
        assert summarizer.max_tokens == 500
        assert summarizer.max_tasks == 5
        assert summarizer.relevance_threshold == 0.2


class TestContextSummarizerSummarize:
    """Tests for summarize_for_task method."""

    def test_empty_previous_tasks(self):
        """Empty previous tasks returns empty string."""
        summarizer = ContextSummarizer()
        task = MockTask(id=1, title="Create user service")
        result = summarizer.summarize_for_task([], task)
        assert result == ""

    def test_no_relevant_tasks(self):
        """Returns empty when no tasks are relevant."""
        summarizer = ContextSummarizer(relevance_threshold=0.9)  # Very high threshold
        previous = [
            {"title": "Configure database", "files": ["db.py"], "summary": "Set up DB"},
        ]
        task = MockTask(id=1, title="Create unrelated component")
        result = summarizer.summarize_for_task(previous, task)
        # With such high threshold and no keyword overlap, should be empty
        assert result == ""

    def test_relevant_task_included(self):
        """Relevant tasks are included in summary."""
        summarizer = ContextSummarizer(relevance_threshold=0.05)
        previous = [
            {
                "title": "Create User model",
                "files": ["models.py"],
                "summary": "Added User class with name and email fields",
            },
        ]
        task = MockTask(
            id=2,
            title="Create UserService that uses User model",
            description="Implement service layer for user operations",
        )
        result = summarizer.summarize_for_task(previous, task)

        assert "## Context from Previous Tasks" in result
        assert "Create User model" in result
        assert "models.py" in result

    def test_file_overlap_boosts_relevance(self):
        """Tasks with file overlap get boosted relevance."""
        summarizer = ContextSummarizer(relevance_threshold=0.1)
        previous = [
            {
                "title": "Initialize project",
                "files": ["models.py"],  # Same file as current task hints
                "summary": "Created initial structure",
            },
            {
                "title": "Add documentation",
                "files": ["README.md"],
                "summary": "Added readme",
            },
        ]
        task = MockTask(
            id=3,
            title="Add fields to model",
            file_hints=["models.py"],  # Overlap with first task
        )
        result = summarizer.summarize_for_task(previous, task)

        # models.py overlap should boost first task
        assert "Initialize project" in result or "models.py" in result

    def test_respects_max_tokens(self):
        """Summary respects token limit."""
        summarizer = ContextSummarizer(max_tokens=100)
        previous = [
            {
                "title": "Task 1 with a very long title that takes many tokens",
                "files": ["file1.py", "file2.py", "file3.py"],
                "summary": "A very long summary " * 50,  # ~200 words
            },
            {
                "title": "Task 2",
                "files": ["file4.py"],
                "summary": "Another long summary " * 50,
            },
        ]
        task = MockTask(id=3, title="Next task", file_hints=["file1.py"])
        result = summarizer.summarize_for_task(previous, task, max_tokens=100)

        # Should be truncated to fit budget
        tokens = _count_tokens_estimate(result)
        assert tokens <= 100

    def test_respects_max_tasks(self):
        """Summary includes at most max_tasks."""
        summarizer = ContextSummarizer(max_tasks=2, relevance_threshold=0.01)
        previous = [
            {"title": f"Task {i}", "files": ["common.py"], "summary": f"Did task {i}"}
            for i in range(5)
        ]
        task = MockTask(id=6, title="Next task using common", file_hints=["common.py"])
        result = summarizer.summarize_for_task(previous, task)

        # Count how many task sections appear (count "### " headers)
        task_sections = result.count("###")
        assert task_sections <= 2


class TestContextSummarizerRelevanceFiltering:
    """Tests for relevance filtering logic."""

    def test_keyword_overlap_relevance(self):
        """Tasks with keyword overlap are considered relevant."""
        summarizer = ContextSummarizer(relevance_threshold=0.1)
        previous = [
            {"title": "Create user authentication", "files": [], "summary": "Auth done"},
        ]
        task = MockTask(id=2, title="Add authentication tests")

        relevant = summarizer._filter_relevant_tasks(previous, task)
        # "authentication" keyword overlap
        assert len(relevant) >= 1

    def test_no_keyword_overlap(self):
        """Tasks without keyword overlap may not be relevant."""
        summarizer = ContextSummarizer(relevance_threshold=0.5)  # High threshold
        previous = [
            {"title": "Configure database", "files": ["db.py"], "summary": "DB setup"},
        ]
        task = MockTask(id=2, title="Create user interface")

        relevant = summarizer._filter_relevant_tasks(previous, task)
        # Different domains, high threshold
        assert len(relevant) == 0

    def test_sorts_by_relevance(self):
        """Tasks are sorted by relevance score."""
        summarizer = ContextSummarizer(max_tasks=2, relevance_threshold=0.01)
        previous = [
            {"title": "Low relevance task", "files": [], "summary": "Other"},
            {"title": "Create user model", "files": ["user.py"], "summary": "User model"},
            {"title": "User authentication setup", "files": ["auth.py"], "summary": "Auth"},
        ]
        task = MockTask(id=4, title="Add user service")

        relevant = summarizer._filter_relevant_tasks(previous, task)
        # "user" keyword should boost last two tasks
        assert len(relevant) <= 2
        # Most relevant should have "user" in title
        if relevant:
            assert any("user" in t.get("title", "").lower() for t in relevant)


class TestContextSummarizerFormatting:
    """Tests for summary formatting."""

    def test_includes_header(self):
        """Summary includes proper header."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [{"title": "Task", "files": ["f.py"], "summary": "Done"}]
        task = MockTask(id=1, title="Next task", file_hints=["f.py"])
        result = summarizer.summarize_for_task(previous, task)

        assert "## Context from Previous Tasks" in result

    def test_task_section_format(self):
        """Task sections are properly formatted."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [
            {
                "title": "Create utils module",
                "files": ["utils.py"],
                "summary": "Added helper functions",
            },
        ]
        task = MockTask(id=2, title="Use utils in main", file_hints=["utils.py"])
        result = summarizer.summarize_for_task(previous, task)

        assert "### Create utils module (completed)" in result
        assert "Files modified:" in result
        assert "utils.py" in result
        assert "Key outcome:" in result

    def test_truncates_many_files(self):
        """Many files are truncated in display."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [
            {
                "title": "Refactor",
                "files": [f"file{i}.py" for i in range(10)],
                "summary": "Done",
            },
        ]
        task = MockTask(id=2, title="Continue refactor", file_hints=["file1.py"])
        result = summarizer.summarize_for_task(previous, task)

        # Should show truncation indicator
        assert "+" in result and "more" in result

    def test_includes_key_decisions(self):
        """Key decisions are included when available."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [
            {
                "title": "Design API",
                "files": ["api.py"],
                "summary": "Created endpoints",
                "key_decisions": ["Use REST over GraphQL", "JSON responses"],
            },
        ]
        task = MockTask(id=2, title="Extend API", file_hints=["api.py"])
        result = summarizer.summarize_for_task(previous, task)

        assert "Decisions:" in result
        assert "REST" in result


class TestContextSummarizerIsRelevant:
    """Tests for is_relevant convenience method."""

    def test_relevant_task_returns_true(self):
        """is_relevant returns True for relevant tasks."""
        summarizer = ContextSummarizer(relevance_threshold=0.05)
        previous_task = {
            "title": "Create user model",
            "files": ["models.py"],
            "summary": "Done",
        }
        current_task = MockTask(id=2, title="User service using model")

        assert summarizer.is_relevant(previous_task, current_task)

    def test_irrelevant_task_returns_false(self):
        """is_relevant returns False for irrelevant tasks."""
        summarizer = ContextSummarizer(relevance_threshold=0.5)
        previous_task = {
            "title": "Configure logging",
            "files": ["logging.py"],
            "summary": "Done",
        }
        current_task = MockTask(id=2, title="Create payment processor")

        assert not summarizer.is_relevant(previous_task, current_task)


class TestContextSummarizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_task_with_no_keywords(self):
        """Handles tasks with no extractable keywords."""
        summarizer = ContextSummarizer()
        previous = [{"title": "a", "files": [], "summary": ""}]
        task = MockTask(id=1, title="")

        # Should not raise
        result = summarizer.summarize_for_task(previous, task)
        assert isinstance(result, str)

    def test_previous_task_missing_fields(self):
        """Handles previous tasks with missing fields."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [
            {"title": "Task only"},  # Missing files and summary
        ]
        task = MockTask(id=1, title="Related task")

        # Should not raise
        result = summarizer.summarize_for_task(previous, task)
        assert isinstance(result, str)

    def test_none_values_in_task(self):
        """Handles None values gracefully."""
        summarizer = ContextSummarizer(relevance_threshold=0.01)
        previous = [
            {
                "title": "Task",
                "files": ["file.py"],
                "summary": None,  # None summary
            },
        ]
        task = MockTask(id=1, title="Next", file_hints=["file.py"])

        result = summarizer.summarize_for_task(previous, task)
        assert isinstance(result, str)

    def test_very_small_token_budget(self):
        """Handles very small token budgets."""
        summarizer = ContextSummarizer(max_tokens=10)
        previous = [
            {"title": "A very long task title", "files": ["f.py"], "summary": "Done"},
        ]
        task = MockTask(id=1, title="Next", file_hints=["f.py"])

        result = summarizer.summarize_for_task(previous, task, max_tokens=10)
        # Should return empty or very short
        assert len(result) < 100


class TestContextSummarizerIntegration:
    """Integration tests for ContextSummarizer with real-world scenarios."""

    def test_multi_task_workflow(self):
        """Test summarizing a multi-task workflow."""
        summarizer = ContextSummarizer(
            max_tokens=2000,
            max_tasks=3,
            relevance_threshold=0.05,
        )

        previous = [
            {
                "title": "Create User model in models.py",
                "files": ["src/models.py"],
                "summary": "Added User dataclass with id, name, email fields",
                "key_decisions": ["Used dataclass over Pydantic", "Added email validation"],
            },
            {
                "title": "Create database layer",
                "files": ["src/database.py"],
                "summary": "Added SQLite connection and CRUD operations",
                "key_decisions": ["Used SQLite for simplicity"],
            },
            {
                "title": "Add configuration",
                "files": ["config.yaml"],
                "summary": "Added YAML config for database path",
                "key_decisions": [],
            },
        ]

        task = MockTask(
            id=4,
            title="Create UserService that uses User model",
            description="Implement service layer using the User model and database",
            file_hints=["src/services.py"],
        )

        result = summarizer.summarize_for_task(previous, task)

        # Should include header
        assert "## Context from Previous Tasks" in result

        # Should include User model task (most relevant)
        assert "User" in result

        # Should include database task (relevant to service)
        assert "database" in result.lower()

    def test_preserves_important_context(self):
        """Test that important context is preserved."""
        summarizer = ContextSummarizer(
            max_tokens=500,
            max_tasks=2,
            relevance_threshold=0.05,
        )

        previous = [
            {
                "title": "Set up authentication",
                "files": ["auth.py"],
                "summary": "JWT-based auth with refresh tokens",
                "key_decisions": ["Used HS256 algorithm", "24h token expiry"],
            },
        ]

        task = MockTask(
            id=2,
            title="Add protected endpoints",
            description="Create endpoints requiring authentication",
            file_hints=["api.py"],
        )

        result = summarizer.summarize_for_task(previous, task)

        # Authentication context should be included
        assert "authentication" in result.lower() or "auth" in result.lower()
        # Key decisions should be visible
        assert "HS256" in result or "token" in result.lower()
