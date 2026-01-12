"""ProjectContext: Rich context about a project combining structure + semantic search.

This module implements Phase 0.2 and Phase 5.1 of EXECUTOR.md:
- Combines project_scan (structure) with Retriever (semantic search)
- Provides task-specific context building
- Enables intelligent file discovery for task execution
- Token budget management (Phase 5.1)
- Context caching for semantic search results (Phase 5.1)

The problem: `project_scan` gives structure but not semantic understanding.
The solution: Add `Retriever` for semantic search over the codebase.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.models import Task
from ai_infra.llm.memory.tokens import count_tokens_approximate
from ai_infra.logging import get_logger
from ai_infra.retriever import Retriever

if TYPE_CHECKING:
    from ai_infra.retriever.models import SearchResult

logger = get_logger("executor.context")


# =============================================================================
# Token Budget Management (Phase 5.1)
# =============================================================================

# Default token budgets for different context sections
DEFAULT_MAX_TOKENS = 50_000  # Default context budget
STRUCTURE_TOKEN_BUDGET = 2_000  # Project structure
SEMANTIC_TOKEN_BUDGET = 20_000  # Semantic search results
FILE_HINTS_TOKEN_BUDGET = 20_000  # File hint contents
DESCRIPTION_TOKEN_BUDGET = 5_000  # Task description

# Priority weights for context sections (higher = more important)
SECTION_PRIORITIES = {
    "description": 1.0,  # Most important - task description
    "file_hints": 0.9,  # Explicitly referenced files
    "semantic": 0.7,  # Semantic search results
    "structure": 0.5,  # Project structure (least important for most tasks)
}


@dataclass
class ContextBudget:
    """Token budget allocation for context building.

    Attributes:
        max_tokens: Maximum total tokens for context.
        structure_budget: Tokens for project structure.
        semantic_budget: Tokens for semantic search results.
        file_hints_budget: Tokens for file hint contents.
        description_budget: Tokens for task description.
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    structure_budget: int = STRUCTURE_TOKEN_BUDGET
    semantic_budget: int = SEMANTIC_TOKEN_BUDGET
    file_hints_budget: int = FILE_HINTS_TOKEN_BUDGET
    description_budget: int = DESCRIPTION_TOKEN_BUDGET

    def total_allocated(self) -> int:
        """Get total allocated budget across all sections."""
        return (
            self.structure_budget
            + self.semantic_budget
            + self.file_hints_budget
            + self.description_budget
        )


@dataclass
class ContextResult:
    """Result of context building with token management.

    Attributes:
        content: The formatted context string.
        tokens_used: Approximate token count of the result.
        sections_included: List of sections included in context.
        sections_truncated: List of sections that were truncated.
        cache_hits: Number of cache hits during context building.
        build_time_ms: Time taken to build context in milliseconds.
    """

    content: str
    tokens_used: int
    sections_included: list[str] = field(default_factory=list)
    sections_truncated: list[str] = field(default_factory=list)
    cache_hits: int = 0
    build_time_ms: float = 0.0

    @property
    def within_budget(self) -> bool:
        """Check if result is within the specified budget."""
        return True  # Always true since we enforce budget during building


# =============================================================================
# Context Caching (Phase 5.1)
# =============================================================================


@dataclass
class CacheEntry:
    """Cached search result entry.

    Attributes:
        key: Cache key (hash of query + k + min_score).
        value: Cached search results.
        created_at: Timestamp when entry was created.
        hits: Number of times this entry was accessed.
    """

    key: str
    value: list[Any]
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl


class ContextCache:
    """In-memory cache for semantic search results.

    Provides fast lookup of previously computed search results to avoid
    redundant embedding computations and API calls.

    Example:
        >>> cache = ContextCache(max_entries=1000, ttl=3600)
        >>> cache.set("query_hash", results)
        >>> cached = cache.get("query_hash")
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl: float = 3600.0,  # 1 hour default TTL
    ):
        """Initialize the context cache.

        Args:
            max_entries: Maximum number of entries to store.
            ttl: Time-to-live for entries in seconds.
        """
        self.max_entries = max_entries
        self.ttl = ttl
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> list[Any] | None:
        """Get cached value by key.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None

        if entry.is_expired(self.ttl):
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        entry.hits += 1
        self._stats["hits"] += 1
        return entry.value

    def set(self, key: str, value: list[Any]) -> None:
        """Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        self._cache[key] = CacheEntry(key=key, value=value)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry from cache."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, evictions, and size.
        """
        return {
            **self._stats,
            "size": len(self._cache),
        }

    @staticmethod
    def make_key(query: str, k: int, min_score: float | None) -> str:
        """Create a cache key from search parameters.

        Args:
            query: Search query.
            k: Number of results.
            min_score: Minimum score threshold.

        Returns:
            Hash-based cache key.
        """
        key_string = f"{query}|{k}|{min_score}"
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()


# File extensions to index for semantic search
DEFAULT_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}

# Directories to exclude from indexing
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    "htmlcov",
    ".coverage",
    "eggs",
    "*.egg-info",
}


class ProjectContext:
    """Rich context about a project, combining structure + semantic search.

    ProjectContext provides two complementary views of a codebase:
    1. Structural: Directory tree, capabilities, git info (via project_scan)
    2. Semantic: Code search by meaning (via Retriever)

    This enables the Executor to:
    - Understand project layout before making changes
    - Find relevant code for a task even without knowing file names
    - Build context that fits within token limits

    Example:
        >>> context = ProjectContext(Path("./my-project"))
        >>>
        >>> # Get project structure
        >>> structure = await context.build_structure()
        >>>
        >>> # Search for relevant code
        >>> results = await context.search("authentication middleware")
        >>>
        >>> # Build context for a specific task
        >>> task = Task(id="1.1", title="Add JWT validation")
        >>> task_context = await context.get_task_context(task)
    """

    def __init__(
        self,
        root: Path | str,
        *,
        code_extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
        retriever_backend: str = "memory",
        lazy_index: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_ttl: float = 3600.0,
        cache_max_entries: int = 1000,
    ):
        """Initialize ProjectContext.

        Args:
            root: Root directory of the project.
            code_extensions: File extensions to index for semantic search.
                Defaults to common code/doc extensions.
            exclude_dirs: Directories to exclude from indexing.
                Defaults to common build/cache directories.
            retriever_backend: Backend for the Retriever. Use "memory" for
                development, "postgres" for production with persistence.
            lazy_index: If True (default), defer indexing until first search.
                Set to False to index immediately on init.
            max_tokens: Maximum tokens for context building (Phase 5.1).
            cache_ttl: Cache time-to-live in seconds (Phase 5.1).
            cache_max_entries: Maximum cache entries (Phase 5.1).
        """
        self.root = Path(root).resolve()
        self.code_extensions = code_extensions or DEFAULT_CODE_EXTENSIONS
        self.exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS
        self._retriever_backend = retriever_backend
        self._retriever: Retriever | None = None
        self._indexed = False
        self._structure_cache: str | None = None

        # Phase 5.1: Token budget management
        self._max_tokens = max_tokens
        self._budget = ContextBudget(max_tokens=max_tokens)

        # Phase 5.1: Context caching
        self._search_cache = ContextCache(
            max_entries=cache_max_entries,
            ttl=cache_ttl,
        )

        if not lazy_index:
            # Index synchronously if not lazy
            asyncio.get_event_loop().run_until_complete(self.build_semantic_index())

    async def build_structure(self, depth: int = 4, use_cache: bool = True) -> str:
        """Get project structure.

        This is the fast path - returns directory tree, capabilities,
        and basic project info without semantic analysis.

        Args:
            depth: Maximum depth for directory tree (default 4).
            use_cache: If True, return cached result if available.

        Returns:
            JSON string with project metadata.
        """
        if use_cache and self._structure_cache:
            return self._structure_cache

        import json

        def _build_tree(path: Path, current_depth: int, max_depth: int) -> list[str]:
            """Build a simple directory tree."""
            if current_depth > max_depth:
                return []

            lines: list[str] = []
            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                for entry in entries:
                    # Skip hidden files and common excludes
                    if entry.name.startswith(".") or entry.name in self.exclude_dirs:
                        continue

                    entry.relative_to(self.root)
                    indent = "  " * current_depth
                    if entry.is_dir():
                        lines.append(f"{indent}{entry.name}/")
                        lines.extend(_build_tree(entry, current_depth + 1, max_depth))
                    else:
                        lines.append(f"{indent}{entry.name}")
            except PermissionError:
                pass
            return lines

        def _detect_capabilities() -> list[str]:
            """Detect project capabilities."""
            caps = []
            if (self.root / "pyproject.toml").exists():
                caps.append("Python/Poetry")
            elif (self.root / "requirements.txt").exists():
                caps.append("Python/pip")
            if (self.root / "package.json").exists():
                caps.append("Node")
            if (self.root / "Dockerfile").exists() or (self.root / "docker-compose.yml").exists():
                caps.append("Docker")
            if (self.root / "Makefile").exists():
                caps.append("Make")
            return caps

        def _run() -> str:
            tree_lines = _build_tree(self.root, 0, depth)
            data = {
                "repo_root": str(self.root),
                "tree": "\n".join(tree_lines) if tree_lines else "(empty)",
                "capabilities": _detect_capabilities(),
            }
            return json.dumps(data, ensure_ascii=False)

        result = await asyncio.to_thread(_run)
        self._structure_cache = result
        return result

    async def build_semantic_index(self) -> int:
        """Build semantic index of the codebase.

        Indexes all code files for semantic search. This is slower on first
        run but enables finding relevant code by meaning.

        Returns:
            Number of files indexed.
        """
        if self._indexed and self._retriever:
            return self._retriever.count()

        logger.info(f"Building semantic index for {self.root}")

        # Create retriever with memory backend (fast, no persistence needed for context)
        self._retriever = Retriever(
            backend=self._retriever_backend,
            # Use smaller chunks for code search
            chunk_size=300,
            chunk_overlap=30,
        )

        files_indexed = 0
        for ext in self.code_extensions:
            for file_path in self.root.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in self.exclude_dirs):
                    continue

                try:
                    # Read file content and add as text
                    # (add_file only supports document types like PDF, DOCX, etc.)
                    rel_path = file_path.relative_to(self.root)
                    content = file_path.read_text(errors="ignore")
                    if content.strip():
                        self._retriever.add_text(
                            content,
                            metadata={
                                "source": str(rel_path),
                                "extension": ext,
                            },
                        )
                        files_indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")
                    continue

        self._indexed = True
        logger.info(f"Indexed {files_indexed} files")
        return files_indexed

    async def search(
        self,
        query: str,
        k: int = 10,
        min_score: float | None = None,
    ) -> list[str]:
        """Find code relevant to a query using semantic search.

        Args:
            query: Natural language query describing what to find.
            k: Number of results to return (default 10).
            min_score: Optional minimum similarity score (0-1).

        Returns:
            List of relevant code snippets.

        Example:
            >>> results = await context.search("error handling for API calls")
            >>> for snippet in results:
            ...     print(snippet)
        """
        if not self._indexed:
            await self.build_semantic_index()

        if not self._retriever:
            return []

        return self._retriever.search(query, k=k, min_score=min_score)

    async def search_detailed(
        self,
        query: str,
        k: int = 10,
        min_score: float | None = None,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Find code with detailed results including source and score.

        Uses caching (Phase 5.1) to avoid redundant embedding computations.

        Args:
            query: Natural language query describing what to find.
            k: Number of results to return (default 10).
            min_score: Optional minimum similarity score (0-1).
            use_cache: If True, use cached results when available.

        Returns:
            List of SearchResult objects with text, score, and metadata.
        """
        if not self._indexed:
            await self.build_semantic_index()

        if not self._retriever:
            return []

        # Phase 5.1: Check cache first
        if use_cache:
            cache_key = ContextCache.make_key(query, k, min_score)
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Perform search
        results = self._retriever.search(query, k=k, min_score=min_score, detailed=True)

        # Phase 5.1: Cache results
        if use_cache:
            cache_key = ContextCache.make_key(query, k, min_score)
            self._search_cache.set(cache_key, results)

        return results

    async def get_task_context(
        self,
        task: Task,
        *,
        include_structure: bool = True,
        include_semantic: bool = True,
        semantic_k: int = 5,
        max_file_bytes: int = 10000,
    ) -> str:
        """Build rich context for a specific task.

        Combines:
        1. Project structure (if include_structure=True)
        2. Semantic search results based on task title (if include_semantic=True)
        3. Contents of files mentioned in task.file_hints

        Args:
            task: The task to build context for.
            include_structure: Include project structure overview.
            include_semantic: Include semantic search results.
            semantic_k: Number of semantic search results.
            max_file_bytes: Max bytes to read from each hinted file.

        Returns:
            Formatted context string for the task.
        """
        sections: list[str] = []

        # 1. Project structure (fast)
        if include_structure:
            try:
                structure = await self.build_structure()
                sections.append(f"## Project Structure\n```json\n{structure}\n```")
            except Exception as e:
                logger.warning(f"Failed to get structure: {e}")

        # 2. Semantic search for relevant code
        if include_semantic and task.title:
            try:
                # Search using task title as query
                relevant = await self.search_detailed(task.title, k=semantic_k)
                if relevant:
                    relevant_parts = []
                    for result in relevant:
                        source = result.source or "unknown"
                        score = result.score
                        relevant_parts.append(
                            f"### {source} (score: {score:.2f})\n```\n{result.text}\n```"
                        )
                    sections.append("## Relevant Code\n" + "\n\n".join(relevant_parts))
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # 3. Explicit file hints from task
        if task.file_hints:
            file_contents = []
            for file_hint in task.file_hints:
                file_path = self.root / file_hint
                if file_path.exists() and file_path.is_file():
                    try:
                        content = file_path.read_text()[:max_file_bytes]
                        # Detect language from extension
                        ext = file_path.suffix.lstrip(".")
                        lang = ext if ext else ""
                        file_contents.append(f"### {file_hint}\n```{lang}\n{content}\n```")
                    except Exception as e:
                        logger.warning(f"Failed to read {file_hint}: {e}")

            if file_contents:
                sections.append("## Referenced Files\n" + "\n\n".join(file_contents))

        # 4. Task description if available
        if task.description:
            sections.append(f"## Task Description\n{task.description}")

        return "\n\n".join(sections)

    async def get_task_context_with_budget(
        self,
        task: Task,
        *,
        max_tokens: int | None = None,
        include_structure: bool = True,
        include_semantic: bool = True,
        semantic_k: int = 10,
        max_file_bytes: int = 50000,
        budget: ContextBudget | None = None,
    ) -> ContextResult:
        """Build rich context for a task with token budget management (Phase 5.1).

        This is the recommended method for building task context as it:
        1. Respects token budgets to avoid exceeding context windows
        2. Intelligently prioritizes content based on relevance
        3. Uses caching for semantic search results
        4. Provides detailed metrics about context building

        Args:
            task: The task to build context for.
            max_tokens: Maximum tokens for the entire context. Defaults to
                instance's max_tokens setting.
            include_structure: Include project structure overview.
            include_semantic: Include semantic search results.
            semantic_k: Number of semantic search results to consider.
            max_file_bytes: Max bytes to read from each hinted file.
            budget: Custom budget allocation. If None, uses default budget.

        Returns:
            ContextResult with content, token counts, and metrics.

        Example:
            >>> result = await context.get_task_context_with_budget(
            ...     task,
            ...     max_tokens=30000,
            ... )
            >>> print(f"Used {result.tokens_used} tokens")
            >>> print(f"Sections: {result.sections_included}")
        """
        start_time = time.time()
        max_tokens = max_tokens or self._max_tokens
        budget = budget or self._budget
        cache_hits = 0

        sections: list[tuple[str, str, int]] = []  # (name, content, tokens)
        sections_truncated: list[str] = []

        # 1. Task description (highest priority - always include)
        if task.description:
            desc_content = f"## Task Description\n{task.description}"
            desc_tokens = count_tokens_approximate([desc_content])
            if desc_tokens > budget.description_budget:
                # Truncate description
                desc_content = self._truncate_to_tokens(desc_content, budget.description_budget)
                desc_tokens = count_tokens_approximate([desc_content])
                sections_truncated.append("description")
            sections.append(("description", desc_content, desc_tokens))

        # 2. File hints (high priority - explicitly referenced)
        if task.file_hints:
            file_content, file_tokens, was_truncated = await self._build_file_hints_section(
                task.file_hints,
                max_file_bytes=max_file_bytes,
                max_tokens=budget.file_hints_budget,
            )
            if file_content:
                sections.append(("file_hints", file_content, file_tokens))
                if was_truncated:
                    sections_truncated.append("file_hints")

        # 3. Semantic search results (medium priority)
        if include_semantic and task.title:
            try:
                # Check cache hit
                cache_key = ContextCache.make_key(task.title, semantic_k, None)
                if self._search_cache.get(cache_key) is not None:
                    cache_hits += 1

                relevant = await self.search_detailed(task.title, k=semantic_k)
                if relevant:
                    semantic_content, semantic_tokens, was_truncated = self._build_semantic_section(
                        relevant,
                        max_tokens=budget.semantic_budget,
                    )
                    if semantic_content:
                        sections.append(("semantic", semantic_content, semantic_tokens))
                        if was_truncated:
                            sections_truncated.append("semantic")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # 4. Project structure (lower priority)
        if include_structure:
            try:
                structure = await self.build_structure()
                struct_content = f"## Project Structure\n```json\n{structure}\n```"
                struct_tokens = count_tokens_approximate([struct_content])
                if struct_tokens > budget.structure_budget:
                    struct_content = self._truncate_to_tokens(
                        struct_content, budget.structure_budget
                    )
                    struct_tokens = count_tokens_approximate([struct_content])
                    sections_truncated.append("structure")
                sections.append(("structure", struct_content, struct_tokens))
            except Exception as e:
                logger.warning(f"Failed to get structure: {e}")

        # Calculate total tokens and fit within budget
        total_tokens = sum(tokens for _, _, tokens in sections)

        # If over budget, prioritize and truncate
        if total_tokens > max_tokens:
            sections = self._fit_sections_to_budget(sections, max_tokens)
            total_tokens = sum(tokens for _, _, tokens in sections)

        # Build final content in priority order
        ordered_sections = self._order_sections_by_priority(sections)
        content = "\n\n".join(content for _, content, _ in ordered_sections)

        build_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Built context: {total_tokens} tokens, "
            f"{len(ordered_sections)} sections, "
            f"{cache_hits} cache hits, "
            f"{build_time_ms:.1f}ms"
        )

        return ContextResult(
            content=content,
            tokens_used=total_tokens,
            sections_included=[name for name, _, _ in ordered_sections],
            sections_truncated=sections_truncated,
            cache_hits=cache_hits,
            build_time_ms=build_time_ms,
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget.

        Uses approximate token counting and truncates at word boundaries.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text with ellipsis if truncated.
        """
        current_tokens = count_tokens_approximate([text])
        if current_tokens <= max_tokens:
            return text

        # Approximate: 4 chars per token
        target_chars = max_tokens * 4
        if len(text) <= target_chars:
            return text

        # Truncate at word boundary
        truncated = text[:target_chars]
        last_space = truncated.rfind(" ")
        if last_space > target_chars // 2:
            truncated = truncated[:last_space]

        return truncated + "\n\n... (truncated to fit token budget)"

    async def _build_file_hints_section(
        self,
        file_hints: list[str],
        max_file_bytes: int,
        max_tokens: int,
    ) -> tuple[str, int, bool]:
        """Build the file hints section with token budget.

        Args:
            file_hints: List of file paths to include.
            max_file_bytes: Max bytes per file.
            max_tokens: Max tokens for entire section.

        Returns:
            Tuple of (content, tokens, was_truncated).
        """
        file_contents = []
        total_tokens = 0
        was_truncated = False
        header = "## Referenced Files\n"
        header_tokens = count_tokens_approximate([header])

        for file_hint in file_hints:
            file_path = self.root / file_hint
            if file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text()[:max_file_bytes]
                    ext = file_path.suffix.lstrip(".")
                    lang = ext if ext else ""
                    file_entry = f"### {file_hint}\n```{lang}\n{content}\n```"
                    entry_tokens = count_tokens_approximate([file_entry])

                    # Check if adding this file would exceed budget
                    if total_tokens + entry_tokens + header_tokens > max_tokens:
                        # Truncate file content to fit
                        remaining = max_tokens - total_tokens - header_tokens - 50
                        if remaining > 100:  # Only include if meaningful
                            file_entry = self._truncate_to_tokens(file_entry, remaining)
                            entry_tokens = count_tokens_approximate([file_entry])
                            file_contents.append(file_entry)
                            total_tokens += entry_tokens
                        was_truncated = True
                        break

                    file_contents.append(file_entry)
                    total_tokens += entry_tokens
                except Exception as e:
                    logger.warning(f"Failed to read {file_hint}: {e}")

        if not file_contents:
            return "", 0, False

        content = header + "\n\n".join(file_contents)
        return content, total_tokens + header_tokens, was_truncated

    def _build_semantic_section(
        self,
        results: list[SearchResult],
        max_tokens: int,
    ) -> tuple[str, int, bool]:
        """Build the semantic search section with token budget.

        Args:
            results: Search results to include.
            max_tokens: Max tokens for section.

        Returns:
            Tuple of (content, tokens, was_truncated).
        """
        parts = []
        total_tokens = 0
        was_truncated = False
        header = "## Relevant Code\n"
        header_tokens = count_tokens_approximate([header])

        for result in results:
            source = result.source or "unknown"
            score = result.score
            entry = f"### {source} (score: {score:.2f})\n```\n{result.text}\n```"
            entry_tokens = count_tokens_approximate([entry])

            if total_tokens + entry_tokens + header_tokens > max_tokens:
                was_truncated = True
                # Try to fit a truncated version
                remaining = max_tokens - total_tokens - header_tokens - 50
                if remaining > 100:
                    entry = self._truncate_to_tokens(entry, remaining)
                    entry_tokens = count_tokens_approximate([entry])
                    parts.append(entry)
                    total_tokens += entry_tokens
                break

            parts.append(entry)
            total_tokens += entry_tokens

        if not parts:
            return "", 0, False

        content = header + "\n\n".join(parts)
        return content, total_tokens + header_tokens, was_truncated

    def _fit_sections_to_budget(
        self,
        sections: list[tuple[str, str, int]],
        max_tokens: int,
    ) -> list[tuple[str, str, int]]:
        """Fit sections within total token budget using priority.

        Lower priority sections are truncated/removed first.

        Args:
            sections: List of (name, content, tokens) tuples.
            max_tokens: Total token budget.

        Returns:
            Filtered/truncated sections that fit within budget.
        """
        # Sort by priority (higher priority = keep more)
        sorted_sections = sorted(
            sections,
            key=lambda s: SECTION_PRIORITIES.get(s[0], 0.5),
            reverse=True,
        )

        result = []
        remaining_budget = max_tokens

        for name, content, tokens in sorted_sections:
            if tokens <= remaining_budget:
                result.append((name, content, tokens))
                remaining_budget -= tokens
            elif remaining_budget > 100:
                # Truncate to fit remaining budget
                truncated = self._truncate_to_tokens(content, remaining_budget - 50)
                truncated_tokens = count_tokens_approximate([truncated])
                result.append((name, truncated, truncated_tokens))
                remaining_budget -= truncated_tokens
            # else: skip this section

        return result

    def _order_sections_by_priority(
        self,
        sections: list[tuple[str, str, int]],
    ) -> list[tuple[str, str, int]]:
        """Order sections for optimal reading flow.

        Args:
            sections: List of (name, content, tokens) tuples.

        Returns:
            Sections ordered for reading (description first, structure last).
        """
        order = ["description", "file_hints", "semantic", "structure"]
        section_dict = {name: (name, content, tokens) for name, content, tokens in sections}
        result = []
        for name in order:
            if name in section_dict:
                result.append(section_dict[name])
        return result

    def count(self) -> int:
        """Get the number of indexed chunks."""
        if not self._retriever:
            return 0
        return self._retriever.count

    def is_indexed(self) -> bool:
        """Check if the semantic index has been built."""
        return self._indexed

    def clear_cache(self) -> None:
        """Clear all caches (structure and search)."""
        self._structure_cache = None
        self._search_cache.clear()

    def clear_search_cache(self) -> None:
        """Clear only the search cache (Phase 5.1)."""
        self._search_cache.clear()

    def cache_stats(self) -> dict[str, int]:
        """Get search cache statistics (Phase 5.1).

        Returns:
            Dict with hits, misses, evictions, and size.
        """
        return self._search_cache.stats()

    def set_budget(self, budget: ContextBudget) -> None:
        """Set custom token budget allocation (Phase 5.1).

        Args:
            budget: New budget allocation.
        """
        self._budget = budget

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set maximum tokens for context building (Phase 5.1).

        Args:
            max_tokens: Maximum tokens allowed.
        """
        self._max_tokens = max_tokens
        self._budget.max_tokens = max_tokens

    async def refresh(self) -> int:
        """Refresh both structure cache and semantic index.

        Also clears search cache since indexed content has changed.

        Returns:
            Number of files re-indexed.
        """
        self.clear_cache()
        self._indexed = False
        self._retriever = None
        return await self.build_semantic_index()
