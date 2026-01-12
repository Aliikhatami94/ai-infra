"""Unit tests for ProjectContext.

These tests validate that ProjectContext correctly:
- Builds project structure via project_scan
- Indexes files for semantic search
- Searches for relevant code
- Builds task-specific context
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_infra.executor.context import (
    DEFAULT_CODE_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
    ProjectContext,
)
from ai_infra.executor.models import Task, TaskStatus


class TestProjectContextInit:
    """Test ProjectContext initialization."""

    def test_default_init(self, tmp_path: Path):
        """Can create ProjectContext with defaults."""
        context = ProjectContext(tmp_path)
        assert context.root == tmp_path
        assert context.code_extensions == DEFAULT_CODE_EXTENSIONS
        assert context.exclude_dirs == DEFAULT_EXCLUDE_DIRS
        assert not context.is_indexed()

    def test_custom_extensions(self, tmp_path: Path):
        """Can specify custom code extensions."""
        custom = {".py", ".rs"}
        context = ProjectContext(tmp_path, code_extensions=custom)
        assert context.code_extensions == custom

    def test_custom_exclude_dirs(self, tmp_path: Path):
        """Can specify custom exclude directories."""
        custom = {"vendor", "third_party"}
        context = ProjectContext(tmp_path, exclude_dirs=custom)
        assert context.exclude_dirs == custom

    def test_string_path(self, tmp_path: Path):
        """Accepts string path."""
        context = ProjectContext(str(tmp_path))
        assert context.root == tmp_path


class TestProjectContextStructure:
    """Test build_structure functionality."""

    @pytest.mark.asyncio
    async def test_build_structure_returns_json(self, tmp_path: Path):
        """build_structure returns valid JSON."""
        # Create a minimal project structure
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")

        context = ProjectContext(tmp_path)
        result = await context.build_structure()

        # Should be valid JSON
        data = json.loads(result)
        assert "tree" in data
        assert "repo_root" in data

    @pytest.mark.asyncio
    async def test_build_structure_caching(self, tmp_path: Path):
        """build_structure caches results."""
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")

        context = ProjectContext(tmp_path)

        # First call
        result1 = await context.build_structure()
        # Second call should use cache
        result2 = await context.build_structure(use_cache=True)

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_build_structure_no_cache(self, tmp_path: Path):
        """Can bypass cache with use_cache=False."""
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")

        context = ProjectContext(tmp_path)

        await context.build_structure()
        # Modify project
        (tmp_path / "new_file.py").write_text("# new")
        # Should get fresh result
        result2 = await context.build_structure(use_cache=False)

        # Results should be different (new_file.py in tree)
        assert "new_file.py" in result2


# Check if langchain_huggingface is available
try:
    import langchain_huggingface  # noqa: F401

    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

requires_huggingface = pytest.mark.skipif(
    not HAS_HUGGINGFACE,
    reason="langchain_huggingface not installed",
)


class TestProjectContextSemanticIndex:
    """Test semantic indexing functionality."""

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_build_semantic_index_indexes_files(self, tmp_path: Path):
        """build_semantic_index indexes Python files."""
        # Create Python files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text(
            '''"""A module that processes data."""

def process_data(items: list) -> list:
    """Process a list of items."""
    return [item.upper() for item in items]
'''
        )
        (tmp_path / "src" / "utils.py").write_text(
            '''"""Utility functions."""

def format_output(text: str) -> str:
    """Format text for display."""
    return f">> {text}"
'''
        )

        context = ProjectContext(tmp_path)
        count = await context.build_semantic_index()

        assert count == 2
        assert context.is_indexed()
        assert context.count() > 0

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_excludes_cache_directories(self, tmp_path: Path):
        """Does not index files in excluded directories."""
        # Create normal file
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main code")

        # Create file in __pycache__ (should be excluded)
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("# cached")

        context = ProjectContext(tmp_path)
        count = await context.build_semantic_index()

        # Should only index src/main.py, not __pycache__/cached.py
        assert count == 1

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_respects_custom_extensions(self, tmp_path: Path):
        """Only indexes files with specified extensions."""
        (tmp_path / "code.py").write_text("# python")
        (tmp_path / "code.rs").write_text("// rust")
        (tmp_path / "code.go").write_text("// go")

        # Only index .py files
        context = ProjectContext(tmp_path, code_extensions={".py"})
        count = await context.build_semantic_index()

        assert count == 1


@requires_huggingface
class TestProjectContextSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_finds_relevant_code(self, tmp_path: Path):
        """search returns relevant code snippets."""
        (tmp_path / "auth.py").write_text(
            '''"""Authentication module."""

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    # Check credentials against database
    return True

def logout_user(user_id: int) -> None:
    """Log out a user."""
    pass
'''
        )
        (tmp_path / "data.py").write_text(
            '''"""Data processing module."""

def process_data(data: list) -> list:
    """Process raw data."""
    return data
'''
        )

        context = ProjectContext(tmp_path)
        await context.build_semantic_index()

        results = await context.search("user authentication login")

        # Should find auth-related code
        assert len(results) > 0
        # At least one result should mention authentication
        combined = " ".join(results)
        assert "authenticate" in combined.lower() or "user" in combined.lower()

    @pytest.mark.asyncio
    async def test_search_lazy_indexes(self, tmp_path: Path):
        """search automatically indexes if not already done."""
        (tmp_path / "code.py").write_text("def hello(): pass")

        context = ProjectContext(tmp_path)
        assert not context.is_indexed()

        # Search should trigger indexing
        await context.search("hello function")

        assert context.is_indexed()

    @pytest.mark.asyncio
    async def test_search_empty_index(self, tmp_path: Path):
        """search returns empty list for empty project."""
        context = ProjectContext(tmp_path)
        results = await context.search("anything")
        assert results == []


class TestProjectContextTaskContext:
    """Test get_task_context functionality."""

    @pytest.mark.asyncio
    async def test_get_task_context_includes_structure(self, tmp_path: Path):
        """get_task_context includes project structure."""
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")

        context = ProjectContext(tmp_path)
        task = Task(id="1.1", title="Add feature")

        result = await context.get_task_context(task, include_semantic=False)

        assert "## Project Structure" in result
        assert "tree" in result

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_get_task_context_includes_semantic(self, tmp_path: Path):
        """get_task_context includes semantic search results."""
        (tmp_path / "cache.py").write_text(
            '''"""Cache module."""

class Cache:
    def get(self, key: str):
        """Get value from cache."""
        pass

    def set(self, key: str, value):
        """Set value in cache."""
        pass
'''
        )

        context = ProjectContext(tmp_path)
        task = Task(id="1.1", title="Improve cache performance")

        result = await context.get_task_context(task, include_structure=False)

        assert "## Relevant Code" in result
        # Should find cache-related code
        assert "cache" in result.lower() or "Cache" in result

    @pytest.mark.asyncio
    async def test_get_task_context_includes_file_hints(self, tmp_path: Path):
        """get_task_context includes contents of hinted files."""
        (tmp_path / "target.py").write_text(
            '''"""Target module to modify."""

def existing_function():
    return 42
'''
        )

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Modify target",
            file_hints=["target.py"],
        )

        result = await context.get_task_context(
            task, include_structure=False, include_semantic=False
        )

        assert "## Referenced Files" in result
        assert "target.py" in result
        assert "existing_function" in result

    @pytest.mark.asyncio
    async def test_get_task_context_includes_description(self, tmp_path: Path):
        """get_task_context includes task description."""
        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Add feature",
            description="This is a detailed description of the task.",
        )

        result = await context.get_task_context(
            task, include_structure=False, include_semantic=False
        )

        assert "## Task Description" in result
        assert "detailed description" in result

    @pytest.mark.asyncio
    async def test_get_task_context_handles_missing_files(self, tmp_path: Path):
        """get_task_context gracefully handles missing hinted files."""
        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Add feature",
            file_hints=["nonexistent.py", "also_missing.py"],
        )

        # Should not raise an error
        result = await context.get_task_context(
            task, include_structure=False, include_semantic=False
        )

        # Should not include references to missing files
        assert "## Referenced Files" not in result


class TestProjectContextUtilities:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_clear_cache(self, tmp_path: Path):
        """clear_cache clears the structure cache."""
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")

        context = ProjectContext(tmp_path)
        await context.build_structure()
        assert context._structure_cache is not None

        context.clear_cache()
        assert context._structure_cache is None

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_refresh(self, tmp_path: Path):
        """refresh rebuilds both caches."""
        (tmp_path / "code.py").write_text("# original")

        context = ProjectContext(tmp_path)
        await context.build_structure()
        await context.build_semantic_index()

        # Modify project
        (tmp_path / "code.py").write_text("# modified content")

        # Refresh
        count = await context.refresh()

        assert context._structure_cache is None
        assert count == 1  # Re-indexed 1 file

    def test_count_before_index(self, tmp_path: Path):
        """count returns 0 before indexing."""
        context = ProjectContext(tmp_path)
        assert context.count() == 0

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_count_after_index(self, tmp_path: Path):
        """count returns chunk count after indexing."""
        (tmp_path / "code.py").write_text("# code\n" * 100)

        context = ProjectContext(tmp_path)
        await context.build_semantic_index()

        assert context.count() > 0


# =============================================================================
# Phase 5.1: Token Budget Management Tests
# =============================================================================


class TestContextBudget:
    """Test ContextBudget configuration."""

    def test_default_budget(self):
        """ContextBudget has reasonable defaults."""
        from ai_infra.executor.context import ContextBudget

        budget = ContextBudget()
        assert budget.max_tokens == 50_000
        assert budget.structure_budget == 2_000
        assert budget.semantic_budget == 20_000
        assert budget.file_hints_budget == 20_000
        assert budget.description_budget == 5_000

    def test_custom_budget(self):
        """Can configure custom budget allocations."""
        from ai_infra.executor.context import ContextBudget

        budget = ContextBudget(
            max_tokens=100_000,
            structure_budget=5_000,
            semantic_budget=40_000,
        )
        assert budget.max_tokens == 100_000
        assert budget.structure_budget == 5_000
        assert budget.semantic_budget == 40_000

    def test_total_allocated(self):
        """total_allocated calculates sum of section budgets."""
        from ai_infra.executor.context import ContextBudget

        budget = ContextBudget(
            structure_budget=1000,
            semantic_budget=2000,
            file_hints_budget=3000,
            description_budget=4000,
        )
        assert budget.total_allocated() == 10000


class TestContextCache:
    """Test ContextCache for semantic search results."""

    def test_cache_set_and_get(self):
        """Can store and retrieve cached values."""
        from ai_infra.executor.context import ContextCache

        cache = ContextCache()
        cache.set("key1", ["result1", "result2"])

        result = cache.get("key1")
        assert result == ["result1", "result2"]

    def test_cache_miss(self):
        """Returns None for cache miss."""
        from ai_infra.executor.context import ContextCache

        cache = ContextCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_ttl_expiry(self):
        """Entries expire after TTL."""
        import time

        from ai_infra.executor.context import ContextCache

        cache = ContextCache(ttl=0.1)  # 100ms TTL
        cache.set("key1", ["result"])

        # Should exist immediately
        assert cache.get("key1") is not None

        # Wait for expiry
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_max_entries_eviction(self):
        """Evicts oldest entry when max_entries reached."""
        from ai_infra.executor.context import ContextCache

        cache = ContextCache(max_entries=3)

        cache.set("key1", ["result1"])
        cache.set("key2", ["result2"])
        cache.set("key3", ["result3"])
        cache.set("key4", ["result4"])  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key4") is not None

    def test_cache_stats(self):
        """Returns accurate cache statistics."""
        from ai_infra.executor.context import ContextCache

        cache = ContextCache()

        # Generate some stats
        cache.set("key1", ["result"])
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_make_key(self):
        """make_key generates deterministic keys."""
        from ai_infra.executor.context import ContextCache

        key1 = ContextCache.make_key("query", 10, None)
        key2 = ContextCache.make_key("query", 10, None)
        key3 = ContextCache.make_key("query", 5, None)

        assert key1 == key2  # Same params = same key
        assert key1 != key3  # Different k = different key

    def test_cache_clear(self):
        """Can clear all entries."""
        from ai_infra.executor.context import ContextCache

        cache = ContextCache()
        cache.set("key1", ["result1"])
        cache.set("key2", ["result2"])

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats()["size"] == 0


class TestContextResult:
    """Test ContextResult dataclass."""

    def test_context_result_basic(self):
        """ContextResult stores content and metrics."""
        from ai_infra.executor.context import ContextResult

        result = ContextResult(
            content="## Context\nSome content",
            tokens_used=100,
            sections_included=["description", "structure"],
        )

        assert "Some content" in result.content
        assert result.tokens_used == 100
        assert "description" in result.sections_included
        assert result.within_budget is True

    def test_context_result_truncated(self):
        """ContextResult tracks truncated sections."""
        from ai_infra.executor.context import ContextResult

        result = ContextResult(
            content="truncated content",
            tokens_used=5000,
            sections_included=["semantic"],
            sections_truncated=["semantic", "file_hints"],
        )

        assert "semantic" in result.sections_truncated
        assert "file_hints" in result.sections_truncated


class TestProjectContextTokenManagement:
    """Test ProjectContext token budget integration."""

    def test_init_with_max_tokens(self, tmp_path: Path):
        """Can initialize with custom max_tokens."""
        context = ProjectContext(tmp_path, max_tokens=100_000)
        assert context._max_tokens == 100_000
        assert context._budget.max_tokens == 100_000

    def test_set_max_tokens(self, tmp_path: Path):
        """Can update max_tokens after init."""
        context = ProjectContext(tmp_path)
        context.set_max_tokens(75_000)

        assert context._max_tokens == 75_000
        assert context._budget.max_tokens == 75_000

    def test_set_custom_budget(self, tmp_path: Path):
        """Can set custom budget allocation."""
        from ai_infra.executor.context import ContextBudget

        context = ProjectContext(tmp_path)
        custom_budget = ContextBudget(
            max_tokens=80_000,
            structure_budget=1000,
            semantic_budget=50_000,
        )
        context.set_budget(custom_budget)

        assert context._budget.structure_budget == 1000
        assert context._budget.semantic_budget == 50_000

    @pytest.mark.asyncio
    async def test_get_task_context_with_budget_basic(self, tmp_path: Path):
        """get_task_context_with_budget returns ContextResult."""
        (tmp_path / "code.py").write_text("def hello(): pass")

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Add logging",
            description="Add logging to the application",
            status=TaskStatus.PENDING,
        )

        result = await context.get_task_context_with_budget(task)

        assert isinstance(result.content, str)
        assert result.tokens_used > 0
        assert len(result.sections_included) > 0
        assert result.build_time_ms > 0

    @pytest.mark.asyncio
    async def test_get_task_context_respects_budget(self, tmp_path: Path):
        """Context fits within specified token budget."""
        # Create a large file
        (tmp_path / "large.py").write_text("# code\n" * 5000)

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Modify code",
            status=TaskStatus.PENDING,
            file_hints=["large.py"],
        )

        # Set a small budget
        result = await context.get_task_context_with_budget(
            task,
            max_tokens=1000,
        )

        # Should fit within budget
        assert result.tokens_used <= 1100  # Allow small overhead

    @pytest.mark.asyncio
    async def test_get_task_context_tracks_truncation(self, tmp_path: Path):
        """Truncated sections are tracked in result."""
        # Create content that will require truncation
        (tmp_path / "huge.py").write_text("# " + "x" * 100000)

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Process data",
            status=TaskStatus.PENDING,
            file_hints=["huge.py"],
        )

        result = await context.get_task_context_with_budget(
            task,
            max_tokens=5000,
        )

        # File hints should be truncated
        assert len(result.sections_truncated) > 0 or result.tokens_used <= 5000


class TestProjectContextCaching:
    """Test ProjectContext search caching."""

    def test_init_with_cache_config(self, tmp_path: Path):
        """Can configure cache TTL and max entries."""
        context = ProjectContext(
            tmp_path,
            cache_ttl=7200.0,  # 2 hours
            cache_max_entries=500,
        )

        assert context._search_cache.ttl == 7200.0
        assert context._search_cache.max_entries == 500

    def test_cache_stats(self, tmp_path: Path):
        """cache_stats returns cache statistics."""
        context = ProjectContext(tmp_path)
        stats = context.cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_search_caches_results(self, tmp_path: Path):
        """search_detailed caches results."""
        (tmp_path / "code.py").write_text("def process(): pass\ndef handle(): pass")

        context = ProjectContext(tmp_path)
        await context.build_semantic_index()

        # First search - cache miss
        await context.search_detailed("process data", k=5)
        stats1 = context.cache_stats()
        assert stats1["misses"] == 1

        # Second search - cache hit
        await context.search_detailed("process data", k=5)
        stats2 = context.cache_stats()
        assert stats2["hits"] == 1

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_search_use_cache_false(self, tmp_path: Path):
        """Can bypass cache with use_cache=False."""
        (tmp_path / "code.py").write_text("def test(): pass")

        context = ProjectContext(tmp_path)
        await context.build_semantic_index()

        # First search - cache miss, result gets cached
        await context.search_detailed("test", k=5)
        stats1 = context.cache_stats()
        assert stats1["misses"] == 1

        # Second search bypassing cache - does not check cache
        # (so no additional hit or miss recorded from get())
        await context.search_detailed("test", k=5, use_cache=False)
        stats2 = context.cache_stats()

        # With use_cache=False, we skip both get and set
        # So stats should remain same as after first call
        assert stats2["misses"] == 1
        assert stats2["hits"] == 0

    def test_clear_search_cache(self, tmp_path: Path):
        """clear_search_cache clears only search cache."""
        context = ProjectContext(tmp_path)
        context._search_cache.set("key", ["result"])
        context._structure_cache = "cached structure"

        context.clear_search_cache()

        assert context._search_cache.get("key") is None
        assert context._structure_cache == "cached structure"

    def test_clear_cache_clears_all(self, tmp_path: Path):
        """clear_cache clears both structure and search cache."""
        context = ProjectContext(tmp_path)
        context._search_cache.set("key", ["result"])
        context._structure_cache = "cached structure"

        context.clear_cache()

        assert context._search_cache.get("key") is None
        assert context._structure_cache is None

    @requires_huggingface
    @pytest.mark.asyncio
    async def test_get_task_context_with_budget_tracks_cache_hits(self, tmp_path: Path):
        """get_task_context_with_budget reports cache hits."""
        (tmp_path / "code.py").write_text("def test(): pass")

        context = ProjectContext(tmp_path)
        await context.build_semantic_index()

        task = Task(id="1.1", title="test query", status=TaskStatus.PENDING)

        # First call - cache miss
        result1 = await context.get_task_context_with_budget(task)
        assert result1.cache_hits == 0

        # Second call - cache hit
        result2 = await context.get_task_context_with_budget(task)
        assert result2.cache_hits == 1


class TestProjectContextPriorityOrdering:
    """Test section priority and ordering."""

    @pytest.mark.asyncio
    async def test_description_has_highest_priority(self, tmp_path: Path):
        """Task description is always included first."""
        (tmp_path / "code.py").write_text("def test(): pass")

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Test task",
            description="Important task description",
            status=TaskStatus.PENDING,
        )

        result = await context.get_task_context_with_budget(task)

        # Description should be first in content
        assert result.content.startswith("## Task Description")
        assert "description" in result.sections_included

    @pytest.mark.asyncio
    async def test_sections_ordered_correctly(self, tmp_path: Path):
        """Sections appear in correct order."""
        (tmp_path / "code.py").write_text("def helper(): pass")

        context = ProjectContext(tmp_path)
        task = Task(
            id="1.1",
            title="Test",
            description="Description here",
            status=TaskStatus.PENDING,
            file_hints=["code.py"],
        )

        result = await context.get_task_context_with_budget(task)

        # Check order: description, file_hints, semantic, structure
        expected_order = ["description", "file_hints", "semantic", "structure"]
        actual_order = result.sections_included

        # Each section that appears should be in correct relative order
        for i, section in enumerate(actual_order):
            expected_idx = expected_order.index(section)
            for j in range(i + 1, len(actual_order)):
                other_idx = expected_order.index(actual_order[j])
                assert expected_idx < other_idx


class TestTruncation:
    """Test text truncation utilities."""

    def test_truncate_to_tokens_no_truncation(self, tmp_path: Path):
        """Short text is not truncated."""
        context = ProjectContext(tmp_path)
        text = "Short text"

        result = context._truncate_to_tokens(text, max_tokens=1000)
        assert result == text
        assert "truncated" not in result.lower()

    def test_truncate_to_tokens_truncates_long_text(self, tmp_path: Path):
        """Long text is truncated with ellipsis."""
        context = ProjectContext(tmp_path)
        text = "x" * 10000  # Very long

        result = context._truncate_to_tokens(text, max_tokens=100)

        assert len(result) < len(text)
        assert "truncated" in result.lower()

    def test_truncate_at_word_boundary(self, tmp_path: Path):
        """Truncation happens at word boundary when possible."""
        context = ProjectContext(tmp_path)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"

        result = context._truncate_to_tokens(text, max_tokens=5)

        # Should not end mid-word
        assert not result.rstrip().endswith("wor")
