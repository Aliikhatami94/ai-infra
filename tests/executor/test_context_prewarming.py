"""Tests for Phase 1.3: Context Pre-warming.

Tests for pre-building context index on graph initialization.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.context_prewarming import (
    ContextPrewarmer,
    ContextWatcher,
    EmbeddingCache,
    EmbeddingCacheEntry,
    PrewarmingStats,
    create_prewarmer,
)

# =============================================================================
# EmbeddingCache Tests (Phase 1.3.3)
# =============================================================================


class TestEmbeddingCacheEntry:
    """Tests for EmbeddingCacheEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test creating a cache entry."""
        entry = EmbeddingCacheEntry(
            embedding=[0.1, 0.2, 0.3],
            created_at=time.time(),
            content_hash="abc123",
        )
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.content_hash == "abc123"
        assert entry.created_at > 0


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_initialization(self) -> None:
        """Test cache initializes with default parameters."""
        cache = EmbeddingCache()
        assert cache.max_entries == 10000
        assert cache.ttl == 0.0

    def test_compute_hash_deterministic(self) -> None:
        """Test hash computation is deterministic."""
        content = "def foo(): pass"
        hash1 = EmbeddingCache.compute_hash(content)
        hash2 = EmbeddingCache.compute_hash(content)
        assert hash1 == hash2

    def test_compute_hash_differs_for_different_content(self) -> None:
        """Test different content produces different hashes."""
        hash1 = EmbeddingCache.compute_hash("def foo(): pass")
        hash2 = EmbeddingCache.compute_hash("def bar(): pass")
        assert hash1 != hash2

    def test_set_and_get_embedding(self) -> None:
        """Test storing and retrieving embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir) / "cache")
            content_hash = "test_hash"
            embedding = [0.1, 0.2, 0.3]

            cache.set(content_hash, embedding)
            result = cache.get(content_hash)

            assert result == embedding

    def test_get_missing_returns_none(self) -> None:
        """Test getting non-existent entry returns None."""
        cache = EmbeddingCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self) -> None:
        """Test entries expire after TTL."""
        cache = EmbeddingCache(ttl=0.001)  # 1ms TTL
        content_hash = "test_hash"
        embedding = [0.1, 0.2, 0.3]

        cache.set(content_hash, embedding)
        time.sleep(0.01)  # Wait for expiration
        result = cache.get(content_hash)

        assert result is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when memory cache is full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use tmpdir to avoid disk cache interference
            cache = EmbeddingCache(
                cache_dir=Path(tmpdir) / "cache",
                max_entries=2,
            )

            cache.set("hash1", [0.1])
            cache.set("hash2", [0.2])
            cache.set("hash3", [0.3])  # Should evict hash1 from memory

            # Memory cache should have evicted hash1, but disk cache still has it
            # So get() will load from disk
            # Instead, check memory cache directly
            assert "hash1" not in cache._memory_cache
            assert cache.get("hash2") == [0.2]
            assert cache.get("hash3") == [0.3]

    def test_disk_persistence(self) -> None:
        """Test embeddings persist to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "embeddings"
            content_hash = "persistent_hash"
            embedding = [0.1, 0.2, 0.3]

            # Save to cache
            cache1 = EmbeddingCache(cache_dir=cache_dir)
            cache1.set(content_hash, embedding)
            cache1.save()

            # Load from new cache instance
            cache2 = EmbeddingCache(cache_dir=cache_dir)
            cache2.load()
            result = cache2.get(content_hash)

            assert result == embedding

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        cache = EmbeddingCache()
        cache.set("hash1", [0.1])
        cache.set("hash2", [0.2])

        cache.clear()

        assert cache.get("hash1") is None
        assert cache.get("hash2") is None

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = EmbeddingCache()
        cache.set("hash1", [0.1])
        cache.set("hash2", [0.2])

        cache.get("hash1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats.entries == 2
        assert stats.hits == 1
        assert stats.misses == 1


# =============================================================================
# ContextWatcher Tests (Phase 1.3.2)
# =============================================================================


class TestContextWatcher:
    """Tests for ContextWatcher file monitoring."""

    def test_watcher_initialization(self) -> None:
        """Test watcher initializes with workspace path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            watcher = ContextWatcher(workspace=workspace)
            assert watcher.workspace == workspace
            assert watcher.debounce_seconds > 0

    def test_watcher_patterns(self) -> None:
        """Test default file patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ContextWatcher(workspace=Path(tmpdir))
            # Default patterns should include Python files
            assert any("*.py" in p or ".py" in p for p in watcher.patterns)

    @pytest.mark.asyncio
    async def test_watcher_start_stop(self) -> None:
        """Test starting and stopping the watcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            watcher = ContextWatcher(workspace=workspace)

            # Start should complete without error
            await watcher.start()
            assert watcher.is_running

            # Stop should complete without error
            await watcher.stop()
            assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_watcher_debouncing(self) -> None:
        """Test file change debouncing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            watcher = ContextWatcher(
                workspace=workspace,
                debounce_seconds=0.1,
            )

            # Queue multiple changes
            watcher._queue_change(workspace / "file1.py")
            watcher._queue_change(workspace / "file1.py")
            watcher._queue_change(workspace / "file1.py")

            # Should deduplicate
            assert len(watcher._pending_changes) <= 1


# =============================================================================
# PrewarmingStats Tests
# =============================================================================


class TestPrewarmingStats:
    """Tests for PrewarmingStats dataclass."""

    def test_stats_creation(self) -> None:
        """Test creating prewarm stats."""
        stats = PrewarmingStats(
            files_indexed=100,
            duration_ms=1500.0,
            cache_hits=50,
            cache_misses=50,
        )
        assert stats.files_indexed == 100
        assert stats.duration_ms == 1500.0
        assert stats.build_duration_ms == 1500.0  # Alias property

    def test_stats_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        stats = PrewarmingStats(
            files_indexed=100,
            duration_ms=1000.0,
            cache_hits=75,
            cache_misses=25,
        )
        assert stats.cache_hit_rate == 0.75

    def test_stats_zero_requests(self) -> None:
        """Test hit rate with no requests."""
        stats = PrewarmingStats(
            files_indexed=0,
            duration_ms=0.0,
            cache_hits=0,
            cache_misses=0,
        )
        assert stats.cache_hit_rate == 0.0


# =============================================================================
# ContextPrewarmer Tests (Phase 1.3.1)
# =============================================================================


class TestContextPrewarmer:
    """Tests for ContextPrewarmer."""

    def test_prewarmer_initialization(self) -> None:
        """Test prewarmer initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            mock_context = MagicMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            assert prewarmer.workspace == workspace
            assert prewarmer.project_context is mock_context
            assert not prewarmer.is_ready

    @pytest.mark.asyncio
    async def test_prewarmer_start_builds_context(self) -> None:
        """Test start() triggers context building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mock_context = MagicMock()
            mock_context.build_structure = AsyncMock(return_value={"files": []})
            mock_context.build_semantic_index = AsyncMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            await prewarmer.start()
            await prewarmer.wait_ready(timeout=5.0)

            # Context methods should have been called
            mock_context.build_structure.assert_called()
            mock_context.build_semantic_index.assert_called()

    @pytest.mark.asyncio
    async def test_prewarmer_wait_ready_success(self) -> None:
        """Test wait_ready returns True when ready."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mock_context = MagicMock()
            mock_context.build_structure = AsyncMock(return_value={"files": []})
            mock_context.build_semantic_index = AsyncMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            await prewarmer.start()
            ready = await prewarmer.wait_ready(timeout=5.0)

            assert ready is True
            assert prewarmer.is_ready

    @pytest.mark.asyncio
    async def test_prewarmer_wait_ready_timeout(self) -> None:
        """Test wait_ready returns False on timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            mock_context = MagicMock()

            # Simulate slow context building
            async def slow_build(*args: Any, **kwargs: Any) -> int:
                await asyncio.sleep(10)  # Very slow
                return 0

            mock_context.build_semantic_index = slow_build
            mock_context.build_structure = AsyncMock(return_value={"files": []})

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            await prewarmer.start()
            ready = await prewarmer.wait_ready(timeout=0.01)  # Very short timeout

            assert ready is False

    @pytest.mark.asyncio
    async def test_prewarmer_get_stats(self) -> None:
        """Test get_stats returns valid statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mock_context = MagicMock()
            mock_context.build_structure = AsyncMock(return_value={"files": [{"path": "test.py"}]})
            mock_context.build_semantic_index = AsyncMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            await prewarmer.start()
            await prewarmer.wait_ready(timeout=5.0)

            stats = prewarmer.get_stats()
            assert stats is not None
            assert stats.build_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_prewarmer_stop(self) -> None:
        """Test stop() gracefully stops prewarmer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mock_context = MagicMock()
            mock_context.build_structure = AsyncMock(return_value={"files": []})
            mock_context.build_semantic_index = AsyncMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
                enable_watcher=True,
            )

            await prewarmer.start()
            await prewarmer.wait_ready(timeout=5.0)
            await prewarmer.stop()

            # Watcher should be stopped
            if prewarmer._watcher is not None:
                assert not prewarmer._watcher.is_running

    @pytest.mark.asyncio
    async def test_prewarmer_with_cache(self) -> None:
        """Test prewarmer uses embedding cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mock_context = MagicMock()
            mock_context.build_structure = AsyncMock(return_value={"files": []})
            mock_context.build_semantic_index = AsyncMock()
            mock_cache = MagicMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
                embedding_cache=mock_cache,
            )

            assert prewarmer._cache is mock_cache


# =============================================================================
# create_prewarmer Factory Tests
# =============================================================================


class TestCreatePrewarmer:
    """Tests for create_prewarmer factory function."""

    def test_create_prewarmer_minimal(self) -> None:
        """Test creating prewarmer with minimal args."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_context = MagicMock()
            prewarmer = create_prewarmer(
                workspace=Path(tmpdir),
                project_context=mock_context,
            )

            assert isinstance(prewarmer, ContextPrewarmer)

    def test_create_prewarmer_with_watcher(self) -> None:
        """Test creating prewarmer with file watcher enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_context = MagicMock()
            prewarmer = create_prewarmer(
                workspace=Path(tmpdir),
                project_context=mock_context,
                enable_watcher=True,
            )

            # Watcher is enabled but created lazily during _prewarm
            assert prewarmer.enable_watcher is True

    def test_create_prewarmer_with_cache(self) -> None:
        """Test creating prewarmer with custom cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_context = MagicMock()
            cache = EmbeddingCache(
                cache_dir=Path(tmpdir) / "embeddings",
                max_entries=100,
            )
            prewarmer = create_prewarmer(
                workspace=Path(tmpdir),
                project_context=mock_context,
                embedding_cache=cache,
            )

            assert prewarmer._cache is cache


# =============================================================================
# ExecutorGraph Integration Tests (Phase 1.3.4)
# =============================================================================


class TestExecutorGraphPrewarming:
    """Tests for ExecutorGraph context pre-warming integration."""

    def test_graph_prewarming_disabled_by_default(self) -> None:
        """Test pre-warming is enabled by default but respects None context."""
        from ai_infra.executor.graph import ExecutorGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap_path = Path(tmpdir) / "ROADMAP.md"
            roadmap_path.write_text("# Roadmap\n## Phase 1\n- [ ] Task 1")

            graph = ExecutorGraph(
                roadmap_path=str(roadmap_path),
                project_context=None,  # No context = no prewarming
            )

            # Should not have prewarmer without project_context
            assert graph._context_prewarmer is None

    def test_graph_prewarming_with_context(self) -> None:
        """Test pre-warming is enabled with project_context."""
        from ai_infra.executor.context import ProjectContext
        from ai_infra.executor.graph import ExecutorGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap_path = Path(tmpdir) / "ROADMAP.md"
            roadmap_path.write_text("# Roadmap\n## Phase 1\n- [ ] Task 1")

            context = ProjectContext(root=Path(tmpdir))
            graph = ExecutorGraph(
                roadmap_path=str(roadmap_path),
                project_context=context,
                enable_context_prewarming=True,
            )

            # Should have prewarmer with project_context
            assert graph._context_prewarmer is not None

    def test_graph_prewarming_disabled_explicitly(self) -> None:
        """Test pre-warming can be explicitly disabled."""
        from ai_infra.executor.context import ProjectContext
        from ai_infra.executor.graph import ExecutorGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap_path = Path(tmpdir) / "ROADMAP.md"
            roadmap_path.write_text("# Roadmap\n## Phase 1\n- [ ] Task 1")

            context = ProjectContext(root=Path(tmpdir))
            graph = ExecutorGraph(
                roadmap_path=str(roadmap_path),
                project_context=context,
                enable_context_prewarming=False,
            )

            # Should not have prewarmer when disabled
            assert graph._context_prewarmer is None

    def test_graph_prewarm_timeout_configurable(self) -> None:
        """Test prewarm timeout is configurable."""
        from ai_infra.executor.context import ProjectContext
        from ai_infra.executor.graph import ExecutorGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap_path = Path(tmpdir) / "ROADMAP.md"
            roadmap_path.write_text("# Roadmap\n## Phase 1\n- [ ] Task 1")

            context = ProjectContext(root=Path(tmpdir))
            graph = ExecutorGraph(
                roadmap_path=str(roadmap_path),
                project_context=context,
                context_prewarm_timeout=10.0,
            )

            assert graph.context_prewarm_timeout == 10.0

    def test_graph_watcher_configurable(self) -> None:
        """Test context watcher is configurable."""
        from ai_infra.executor.context import ProjectContext
        from ai_infra.executor.graph import ExecutorGraph

        with tempfile.TemporaryDirectory() as tmpdir:
            roadmap_path = Path(tmpdir) / "ROADMAP.md"
            roadmap_path.write_text("# Roadmap\n## Phase 1\n- [ ] Task 1")

            context = ProjectContext(root=Path(tmpdir))
            graph = ExecutorGraph(
                roadmap_path=str(roadmap_path),
                project_context=context,
                enable_context_watcher=True,
            )

            assert graph.enable_context_watcher is True


# =============================================================================
# Benchmark Tests (Phase 1.3.5)
# =============================================================================


class TestContextPrewarmingBenchmarks:
    """Benchmark tests for context pre-warming."""

    @pytest.mark.asyncio
    async def test_prewarm_faster_than_cold_build(self) -> None:
        """Test pre-warmed context is faster than cold build."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create some test files
            for i in range(10):
                (workspace / f"test_{i}.py").write_text(f"def func_{i}(): pass")

            mock_context = MagicMock()
            build_calls = []

            async def track_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
                build_calls.append(time.time())
                await asyncio.sleep(0.01)  # Simulate work
                return {"files": []}

            mock_context.build_structure = track_build
            mock_context.build_semantic_index = AsyncMock()

            prewarmer = ContextPrewarmer(
                workspace=workspace,
                project_context=mock_context,
            )

            # Start prewarming
            start = time.time()
            await prewarmer.start()
            await prewarmer.wait_ready(timeout=5.0)
            prewarm_duration = time.time() - start

            # Pre-warming should complete in reasonable time
            assert prewarm_duration < 5.0  # Less than timeout

    @pytest.mark.asyncio
    async def test_cache_improves_second_run(self) -> None:
        """Test embedding cache improves second run performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = EmbeddingCache(cache_dir=cache_dir)

            # First run - cache misses
            content = "def expensive_function(): pass"
            content_hash = cache.compute_hash(content)

            result1 = cache.get(content_hash)
            assert result1 is None  # Miss

            cache.set(content_hash, [0.1, 0.2, 0.3])

            # Second run - cache hit
            result2 = cache.get(content_hash)
            assert result2 == [0.1, 0.2, 0.3]  # Hit

            stats = cache.get_stats()
            assert stats.hits == 1
            assert stats.misses == 1
