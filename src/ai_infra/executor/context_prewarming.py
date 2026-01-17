"""Context pre-warming for the Executor.

Phase 1.3: Context pre-warming to eliminate build latency during task execution.

This module provides:
- ContextPrewarmer: Async pre-builds context index on graph initialization
- EmbeddingCache: Caches embeddings by content hash to avoid recomputation
- ContextWatcher: Monitors file changes for incremental index updates

The problem: Context building happens after task starts, adding 1-2s latency.
The solution: Pre-build context index immediately on graph initialization,
so context is ready before the first task starts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.context import ProjectContext

logger = get_logger("executor.context_prewarming")


# =============================================================================
# Embedding Cache (Phase 1.3.3)
# =============================================================================


@dataclass
class EmbeddingCacheEntry:
    """A cached embedding entry.

    Attributes:
        embedding: The embedding vector.
        created_at: Unix timestamp when cached.
        content_hash: Hash of the original content.
    """

    embedding: list[float]
    created_at: float
    content_hash: str


@dataclass
class EmbeddingCacheStats:
    """Statistics for embedding cache.

    Attributes:
        entries: Number of entries in memory cache.
        disk_entries: Number of entries on disk.
        max_entries: Maximum entries allowed.
        ttl: Time-to-live in seconds.
        hits: Number of cache hits.
        misses: Number of cache misses.
    """

    entries: int
    disk_entries: int
    max_entries: int
    ttl: float
    hits: int
    misses: int


class EmbeddingCache:
    """Cache embeddings to avoid recomputation.

    Phase 1.3.3: Caches embeddings by content hash for fast lookup.

    The cache persists to disk in the .executor/embeddings directory,
    allowing embeddings to be reused across runs.

    Attributes:
        cache_dir: Directory for cache files.
        max_entries: Maximum cache entries (LRU eviction).
        ttl: Time-to-live in seconds (0 = no expiry).

    Example:
        >>> cache = EmbeddingCache(cache_dir=Path(".executor/embeddings"))
        >>> content = "def foo(): pass"
        >>> content_hash = cache.compute_hash(content)
        >>> embedding = cache.get(content_hash)
        >>> if embedding is None:
        ...     embedding = model.embed(content)
        ...     cache.set(content_hash, embedding)
    """

    def __init__(
        self,
        cache_dir: Path | str = Path(".executor/embeddings"),
        max_entries: int = 10000,
        ttl: float = 0.0,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory for cache files.
            max_entries: Maximum cache entries (default: 10000).
            ttl: Time-to-live in seconds (default: 0 = no expiry).
        """
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.ttl = ttl
        self._memory_cache: dict[str, EmbeddingCacheEntry] = {}
        self._access_order: list[str] = []  # LRU tracking
        self._hits = 0
        self._misses = 0

    def _ensure_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Text content to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, content_hash: str) -> list[float] | None:
        """Get cached embedding by content hash.

        Args:
            content_hash: SHA-256 hash of content.

        Returns:
            Embedding vector, or None if not cached or expired.
        """
        # Check memory cache first
        if content_hash in self._memory_cache:
            entry = self._memory_cache[content_hash]
            # Check TTL
            if self.ttl > 0 and (time.time() - entry.created_at) > self.ttl:
                del self._memory_cache[content_hash]
                self._access_order.remove(content_hash)
                self._misses += 1
                return None
            # Update LRU order
            if content_hash in self._access_order:
                self._access_order.remove(content_hash)
            self._access_order.append(content_hash)
            self._hits += 1
            return entry.embedding

        # Check disk cache
        cache_file = self.cache_dir / f"{content_hash}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                created_at = data.get("created_at", 0)
                # Check TTL
                if self.ttl > 0 and (time.time() - created_at) > self.ttl:
                    cache_file.unlink()
                    self._misses += 1
                    return None
                embedding = data.get("embedding", [])
                # Load into memory cache
                self._memory_cache[content_hash] = EmbeddingCacheEntry(
                    embedding=embedding,
                    created_at=created_at,
                    content_hash=content_hash,
                )
                self._access_order.append(content_hash)
                self._evict_if_needed()
                self._hits += 1
                return embedding
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read embedding cache: {e}")
                self._misses += 1
                return None

        self._misses += 1
        return None

    def set(self, content_hash: str, embedding: list[float]) -> None:
        """Cache an embedding.

        Args:
            content_hash: SHA-256 hash of content.
            embedding: Embedding vector to cache.
        """
        now = time.time()

        # Add to memory cache
        self._memory_cache[content_hash] = EmbeddingCacheEntry(
            embedding=embedding,
            created_at=now,
            content_hash=content_hash,
        )
        if content_hash in self._access_order:
            self._access_order.remove(content_hash)
        self._access_order.append(content_hash)
        self._evict_if_needed()

        # Write to disk cache
        self._ensure_dir()
        cache_file = self.cache_dir / f"{content_hash}.json"
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "embedding": embedding,
                        "created_at": now,
                        "content_hash": content_hash,
                    }
                )
            )
        except OSError as e:
            logger.warning(f"Failed to write embedding cache: {e}")

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over max_entries."""
        while len(self._memory_cache) > self.max_entries:
            if self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._memory_cache:
                    del self._memory_cache[oldest]

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def save(self) -> None:
        """Save memory cache to disk.

        Persists all in-memory entries to disk for future sessions.
        """
        self._ensure_dir()
        for content_hash, entry in self._memory_cache.items():
            cache_file = self.cache_dir / f"{content_hash}.json"
            if not cache_file.exists():
                try:
                    cache_file.write_text(
                        json.dumps(
                            {
                                "embedding": entry.embedding,
                                "created_at": entry.created_at,
                                "content_hash": entry.content_hash,
                            }
                        )
                    )
                except OSError as e:
                    logger.warning(f"Failed to save embedding cache: {e}")

    def load(self) -> None:
        """Load cached embeddings from disk.

        Populates memory cache from disk cache entries.
        """
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                content_hash = data.get("content_hash", cache_file.stem)
                created_at = data.get("created_at", 0)

                # Check TTL
                if self.ttl > 0 and (time.time() - created_at) > self.ttl:
                    cache_file.unlink()
                    continue

                self._memory_cache[content_hash] = EmbeddingCacheEntry(
                    embedding=data.get("embedding", []),
                    created_at=created_at,
                    content_hash=content_hash,
                )
                self._access_order.append(content_hash)

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load embedding cache: {e}")

        self._evict_if_needed()

    def get_stats(self) -> EmbeddingCacheStats:
        """Get cache statistics.

        Returns:
            EmbeddingCacheStats with cache metrics.
        """
        disk_count = 0
        if self.cache_dir.exists():
            disk_count = len(list(self.cache_dir.glob("*.json")))
        return EmbeddingCacheStats(
            entries=len(self._memory_cache),
            disk_entries=disk_count,
            max_entries=self.max_entries,
            ttl=self.ttl,
            hits=self._hits,
            misses=self._misses,
        )

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        disk_count = 0
        if self.cache_dir.exists():
            disk_count = len(list(self.cache_dir.glob("*.json")))
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_count,
            "max_entries": self.max_entries,
            "ttl": self.ttl,
        }


# =============================================================================
# Context Watcher (Phase 1.3.2)
# =============================================================================


class ContextWatcher:
    """Watch for file changes and update context incrementally.

    Phase 1.3.2: Uses asyncio-based file watching for incremental updates.

    This class monitors the workspace for file changes and queues them
    for incremental context updates, avoiding full re-indexing.

    Attributes:
        workspace: Workspace directory to watch.
        context: ProjectContext to update.
        pending_updates: Queue of file paths to update.
        debounce_seconds: Debounce time for rapid changes.

    Example:
        >>> watcher = ContextWatcher(workspace=Path("./project"))
        >>> await watcher.start()
        >>> # ... file changes detected and queued ...
        >>> await watcher.stop()
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        *,
        root: Path | str | None = None,  # Deprecated alias
        context: ProjectContext | None = None,
        debounce_ms: float = 100.0,
        debounce_seconds: float | None = None,
        extensions: set[str] | None = None,
    ):
        """Initialize context watcher.

        Args:
            workspace: Workspace directory to watch.
            root: Deprecated alias for workspace.
            context: ProjectContext to update on changes.
            debounce_ms: Debounce time in milliseconds (default: 100ms).
            debounce_seconds: Debounce time in seconds (overrides debounce_ms).
            extensions: File extensions to watch (default: code extensions).
        """
        # Support both workspace and root for backwards compatibility
        if workspace is not None:
            self._workspace = Path(workspace).resolve()
        elif root is not None:
            self._workspace = Path(root).resolve()
        else:
            raise ValueError("workspace or root must be provided")

        self.root = self._workspace  # Backwards compatibility
        self.context = context

        # Handle debounce time
        if debounce_seconds is not None:
            self._debounce_seconds = debounce_seconds
        else:
            self._debounce_seconds = debounce_ms / 1000.0

        self.extensions = extensions or {".py", ".ts", ".js", ".md", ".yaml", ".yml"}
        self.pending_updates: asyncio.Queue[Path] = asyncio.Queue()
        self._observer: Any | None = None
        self._process_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_events: dict[str, float] = {}  # path -> timestamp for debounce
        self._pending_changes: set[Path] = set()  # For debouncing

    @property
    def workspace(self) -> Path:
        """Get the workspace directory."""
        return self._workspace

    @property
    def debounce_seconds(self) -> float:
        """Get the debounce time in seconds."""
        return self._debounce_seconds

    @property
    def debounce_ms(self) -> float:
        """Get the debounce time in milliseconds."""
        return self._debounce_seconds * 1000.0

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._running

    @property
    def patterns(self) -> list[str]:
        """Get the file patterns being watched."""
        return [f"*{ext}" for ext in self.extensions]

    def _queue_change(self, path: Path) -> None:
        """Queue a file change for processing.

        Args:
            path: Path to the changed file.
        """
        self._pending_changes.add(path)

    def _should_watch(self, path: Path) -> bool:
        """Check if a file should be watched.

        Args:
            path: File path to check.

        Returns:
            True if file should be watched.
        """
        # Check extension
        if path.suffix not in self.extensions:
            return False
        # Skip hidden files and common excludes
        parts = path.parts
        if any(p.startswith(".") for p in parts):
            return False
        if any(
            p in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build") for p in parts
        ):
            return False
        return True

    def on_file_changed(self, path: Path) -> None:
        """Handle file change event.

        Args:
            path: Path to the changed file.
        """
        if not self._should_watch(path):
            return

        # Debounce rapid changes
        path_str = str(path)
        now = time.time() * 1000  # ms
        last_event = self._last_events.get(path_str, 0)
        if (now - last_event) < self.debounce_ms:
            return
        self._last_events[path_str] = now

        # Queue update
        try:
            self.pending_updates.put_nowait(path)
        except asyncio.QueueFull:
            logger.warning(f"Update queue full, dropping change for {path}")

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        self._running = True

        # Start watchdog observer if available
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class _Handler(FileSystemEventHandler):
                def __init__(self, watcher: ContextWatcher):
                    self.watcher = watcher

                def on_modified(self, event: Any) -> None:
                    if not event.is_directory:
                        self.watcher.on_file_changed(Path(event.src_path))

                def on_created(self, event: Any) -> None:
                    if not event.is_directory:
                        self.watcher.on_file_changed(Path(event.src_path))

            self._observer = Observer()
            self._observer.schedule(_Handler(self), str(self.root), recursive=True)
            self._observer.start()
            logger.info(f"Started file watcher for {self.root}")

        except ImportError:
            logger.warning("watchdog not installed, file watching disabled")
            self._observer = None

        # Start background processor
        self._process_task = asyncio.create_task(self._process_updates())

    async def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None

        if self._process_task is not None:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        logger.info("Stopped file watcher")

    async def _process_updates(self) -> None:
        """Process queued file updates."""
        while self._running:
            try:
                # Wait for update with timeout
                path = await asyncio.wait_for(
                    self.pending_updates.get(),
                    timeout=1.0,
                )
                await self._update_file(path)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error processing file update: {e}")

    async def _update_file(self, path: Path) -> None:
        """Update context for a single file.

        Args:
            path: Path to the file to update.
        """
        if self.context is None:
            return

        try:
            # Check if file still exists
            if not path.exists():
                logger.debug(f"File deleted: {path}")
                # TODO: Remove from index
                return

            # Read content and update index
            rel_path = path.relative_to(self.root)
            content = path.read_text(errors="ignore")

            if content.strip():
                # Update retriever if available
                retriever = getattr(self.context, "_retriever", None)
                if retriever is not None:
                    # Remove old entry and add new
                    # Note: This is a simple approach; production would use upsert
                    retriever.add_text(
                        content,
                        metadata={
                            "source": str(rel_path),
                            "extension": path.suffix,
                            "updated": True,
                        },
                    )
                    logger.debug(f"Updated index for {rel_path}")

        except Exception as e:
            logger.warning(f"Failed to update file {path}: {e}")

    def get_pending_count(self) -> int:
        """Get number of pending updates.

        Returns:
            Number of files waiting to be processed.
        """
        return self.pending_updates.qsize()


# =============================================================================
# Context Prewarmer (Phase 1.3.1)
# =============================================================================


@dataclass
class PrewarmingStats:
    """Statistics from context pre-warming.

    Attributes:
        started_at: When pre-warming started.
        completed_at: When pre-warming completed.
        files_indexed: Number of files indexed.
        duration_ms: Total duration in milliseconds.
        cache_hits: Number of embedding cache hits.
        cache_misses: Number of embedding cache misses.
        build_duration_ms: Alias for duration_ms.
    """

    started_at: float = 0.0
    completed_at: float = 0.0
    files_indexed: int = 0
    duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def build_duration_ms(self) -> float:
        """Get build duration in milliseconds (alias for duration_ms)."""
        return self.duration_ms

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "files_indexed": self.files_indexed,
            "duration_ms": self.duration_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
        }


class ContextPrewarmer:
    """Pre-warms context index for zero-latency task execution.

    Phase 1.3.1: Starts building context index immediately on initialization,
    so context is ready before the first task starts.

    This class:
    1. Starts context building as a background task
    2. Tracks pre-warming progress
    3. Provides a wait mechanism with timeout
    4. Optionally starts a file watcher for incremental updates

    Attributes:
        workspace: Workspace root directory.
        context: ProjectContext being pre-warmed.
        embedding_cache: Optional embedding cache.
        watcher: Optional file watcher for incremental updates.
        ready: Event signaling context is ready.
        stats: Pre-warming statistics.

    Example:
        >>> prewarmer = ContextPrewarmer(workspace=Path("./project"))
        >>> await prewarmer.start()
        >>> # ... context building in background ...
        >>> await prewarmer.wait_ready(timeout=5.0)
        >>> context = prewarmer.context
    """

    def __init__(
        self,
        workspace: Path | str,
        *,
        context: ProjectContext | None = None,
        project_context: ProjectContext | None = None,  # Alias for context
        embedding_cache: EmbeddingCache | None = None,
        enable_watcher: bool = False,
        max_tokens: int = 50000,
    ):
        """Initialize context prewarmer.

        Args:
            workspace: Workspace root directory.
            context: Optional existing ProjectContext to use.
            project_context: Alias for context parameter.
            embedding_cache: Optional embedding cache.
            enable_watcher: Enable file watcher for incremental updates.
            max_tokens: Maximum tokens for context building.
        """
        self.workspace = Path(workspace).resolve()
        self._context = context or project_context
        self._cache = embedding_cache
        self.enable_watcher = enable_watcher
        self.max_tokens = max_tokens

        self._ready = asyncio.Event()
        self._prewarm_task: asyncio.Task[None] | None = None
        self._watcher: ContextWatcher | None = None
        self._stats = PrewarmingStats()
        self._started = False
        self._error: Exception | None = None

    @property
    def context(self) -> ProjectContext | None:
        """Get the pre-warmed ProjectContext."""
        return self._context

    @property
    def project_context(self) -> ProjectContext | None:
        """Get the pre-warmed ProjectContext (alias for context)."""
        return self._context

    # Alias embedding_cache to _cache for external access
    @property
    def embedding_cache(self) -> EmbeddingCache | None:
        """Get the embedding cache."""
        return self._cache

    @property
    def ready(self) -> asyncio.Event:
        """Get the ready event."""
        return self._ready

    @property
    def stats(self) -> PrewarmingStats:
        """Get pre-warming statistics."""
        return self._stats

    def get_stats(self) -> PrewarmingStats:
        """Get pre-warming statistics.

        Returns:
            PrewarmingStats with current metrics.
        """
        return self._stats

    @property
    def is_ready(self) -> bool:
        """Check if context is ready."""
        return self._ready.is_set()

    @property
    def error(self) -> Exception | None:
        """Get any error that occurred during pre-warming."""
        return self._error

    async def start(self) -> None:
        """Start pre-warming context in the background.

        This method returns immediately. Use wait_ready() to wait
        for context to be ready.
        """
        if self._started:
            return

        self._started = True
        self._stats.started_at = time.time()

        # Start pre-warming as background task
        self._prewarm_task = asyncio.create_task(self._prewarm())

    async def wait_ready(self, timeout: float = 10.0) -> bool:
        """Wait for context to be ready.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if context is ready, False if timeout.

        Raises:
            Exception: If pre-warming failed with an error.
        """
        if not self._started:
            await self.start()

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            if self._error is not None:
                raise self._error
            return True
        except TimeoutError:
            logger.warning(f"Context pre-warming timed out after {timeout}s")
            return False

    async def stop(self) -> None:
        """Stop pre-warming and cleanup resources."""
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
            try:
                await self._prewarm_task
            except asyncio.CancelledError:
                pass
            self._prewarm_task = None

        if self._watcher is not None:
            await self._watcher.stop()
            self._watcher = None

    async def _prewarm(self) -> None:
        """Pre-warm the context index."""
        try:
            logger.info(f"Starting context pre-warming for {self.workspace}")

            # Create ProjectContext if not provided
            if self._context is None:
                from ai_infra.executor.context import ProjectContext

                self._context = ProjectContext(
                    root=self.workspace,
                    lazy_index=True,  # We'll index manually
                    max_tokens=self.max_tokens,
                )

            # Build semantic index
            files_indexed = await self._context.build_semantic_index()
            self._stats.files_indexed = files_indexed

            # Build structure cache
            await self._context.build_structure()

            # Start file watcher if enabled
            if self.enable_watcher:
                self._watcher = ContextWatcher(
                    root=self.workspace,
                    context=self._context,
                )
                await self._watcher.start()

            # Mark as ready
            self._stats.completed_at = time.time()
            self._stats.duration_ms = (self._stats.completed_at - self._stats.started_at) * 1000

            logger.info(
                f"Context pre-warming complete: "
                f"{files_indexed} files indexed in {self._stats.duration_ms:.1f}ms"
            )

            self._ready.set()

        except asyncio.CancelledError:
            logger.info("Context pre-warming cancelled")
            raise

        except Exception as e:
            logger.exception(f"Context pre-warming failed: {e}")
            self._error = e
            self._ready.set()  # Set ready so waiters don't hang


# =============================================================================
# Factory Functions
# =============================================================================


def create_prewarmer(
    workspace: Path | str,
    *,
    project_context: ProjectContext | None = None,
    cache_dir: Path | str | None = None,
    enable_watcher: bool = False,
    enable_cache: bool = True,
    embedding_cache: EmbeddingCache | None = None,
) -> ContextPrewarmer:
    """Create a context prewarmer with optional cache and watcher.

    Args:
        workspace: Workspace root directory.
        project_context: Optional existing ProjectContext to use.
        cache_dir: Directory for embedding cache (default: .executor/embeddings).
        enable_watcher: Enable file watcher for incremental updates.
        enable_cache: Enable embedding cache.
        embedding_cache: Optional pre-configured embedding cache.

    Returns:
        Configured ContextPrewarmer.
    """
    workspace = Path(workspace)

    # Use provided cache or create new one
    cache = embedding_cache
    if cache is None and enable_cache:
        if cache_dir is None:
            cache_dir = workspace / ".executor" / "embeddings"
        cache = EmbeddingCache(cache_dir=cache_dir)

    return ContextPrewarmer(
        workspace=workspace,
        project_context=project_context,
        embedding_cache=cache,
        enable_watcher=enable_watcher,
    )
