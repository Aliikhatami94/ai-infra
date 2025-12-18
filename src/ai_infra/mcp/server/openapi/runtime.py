from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any

from .models import OpenAPISpec, Operation


def sanitize_tool_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "op"


def op_tool_name(path: str, method: str, opid: str | None) -> str:
    if opid:
        return sanitize_tool_name(opid)
    return sanitize_tool_name(f"{method.lower()}_{path.strip('/').replace('/', '_')}")


# =============================================================================
# Array Parameter Serialization
# =============================================================================


def serialize_query_param(
    name: str,
    value: Any,
    style: str | None = None,
    explode: bool | None = None,
) -> dict[str, Any]:
    """Serialize a query parameter according to OpenAPI style/explode.

    Supports:
    - style=form (default): ?ids=1,2,3 or ?ids=1&ids=2&ids=3
    - style=spaceDelimited: ?ids=1%202%203
    - style=pipeDelimited: ?ids=1|2|3
    - style=deepObject: ?ids[0]=1&ids[1]=2 (for objects)

    Args:
        name: Parameter name
        value: Parameter value (may be list/array)
        style: OpenAPI serialization style
        explode: Whether to explode arrays

    Returns:
        Dict suitable for httpx params
    """
    # Default style is "form" for query params
    style = style or "form"

    # Default explode depends on style
    if explode is None:
        explode = style == "form"

    # Handle non-array values
    if not isinstance(value, (list, tuple)):
        return {name: value}

    # Empty array
    if not value:
        return {}

    # Explode: each value gets its own key
    if explode:
        # Return list - httpx will serialize as ?key=val1&key=val2
        return {name: list(value)}

    # Non-exploded: join with delimiter
    if style == "form":
        return {name: ",".join(str(v) for v in value)}
    elif style == "spaceDelimited":
        return {name: " ".join(str(v) for v in value)}
    elif style == "pipeDelimited":
        return {name: "|".join(str(v) for v in value)}
    elif style == "deepObject":
        # deepObject: ?ids[0]=1&ids[1]=2
        return {f"{name}[{i}]": v for i, v in enumerate(value)}
    else:
        # Unknown style, default to comma-separated
        return {name: ",".join(str(v) for v in value)}


def pick_effective_base_url_with_source(
    spec: OpenAPISpec,
    path_item: dict[str, Any] | None,
    op: Operation | None,
    override: str | None,
) -> tuple[str, str]:
    """
    Returns (url, source) where source ∈ {"override","operation","path","root","none"}.
    """
    if override:
        return override.rstrip("/"), "override"
    for source, node in (
        ("operation", op or {}),
        ("path", path_item or {}),
        ("root", spec or {}),
    ):
        servers = node.get("servers") or []
        if servers:
            url = str(servers[0].get("url", "")).rstrip("/")
            if url:
                return url, source
    return "", "none"


def collect_params(op: Operation) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"path": [], "query": [], "header": []}
    for p in op.get("parameters") or []:
        loc = p.get("in")
        if loc in out:
            out[loc].append(p)
    return out


def has_request_body(op: Operation) -> bool:
    return bool(op.get("requestBody", {}).get("content"))


def extract_body_content_type(op: Operation) -> str:
    content = op.get("requestBody", {}).get("content", {})
    for ct in ("application/json", "application/x-www-form-urlencoded", "text/plain"):
        if ct in content:
            return ct
    return next(iter(content.keys())) if content else "application/json"


def merge_parameters(path_item: dict[str, Any] | None, op: Operation) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for src in (path_item.get("parameters") if path_item else []) or []:
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            merged.append(src)
            seen.add((src["in"], src["name"]))
    for src in op.get("parameters") or []:
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            key = (src["in"], src["name"])
            if key in seen:
                for i, existing in enumerate(merged):
                    if (existing.get("in"), existing.get("name")) == key:
                        merged[i] = src
                        break
            else:
                merged.append(src)
                seen.add(key)
    return merged


def split_params(params: list[dict[str, Any]]):
    path_params: list[dict[str, Any]] = []
    query_params: list[dict[str, Any]] = []
    header_params: list[dict[str, Any]] = []
    cookie_params: list[dict[str, Any]] = []
    for p in params:
        loc = p.get("in")
        if loc == "path":
            path_params.append(p)
        elif loc == "query":
            query_params.append(p)
        elif loc == "header":
            header_params.append(p)
        elif loc == "cookie":
            cookie_params.append(p)
    return path_params, query_params, header_params, cookie_params


def pick_effective_base_url(
    spec: OpenAPISpec,
    path_item: dict[str, Any] | None,
    op: Operation | None,
    override: str | None,
) -> str:
    if override:
        return override.rstrip("/")
    for node in (op or {}, path_item or {}, spec):  # op → path → root
        servers = node.get("servers") or []
        if servers:
            return str(servers[0].get("url", "")).rstrip("/") or ""
    return ""


# =============================================================================
# Response Cache (TTL-based)
# =============================================================================


class ResponseCache:
    """Simple in-memory TTL cache for HTTP responses.

    Thread-safe for async usage. Stores responses by cache key.

    Example:
        cache = ResponseCache(ttl=300, methods=["GET"])

        # Check cache
        key = cache.make_key("GET", "/users/1", {"page": 1})
        if cached := cache.get(key):
            return cached

        # Store response
        cache.set(key, response_data)
    """

    def __init__(
        self,
        ttl: float = 300.0,
        methods: list[str] | None = None,
        max_size: int = 1000,
    ):
        """Initialize cache.

        Args:
            ttl: Time-to-live in seconds
            methods: HTTP methods to cache (default: ["GET"])
            max_size: Maximum number of entries
        """
        self.ttl = ttl
        self.methods = set(m.upper() for m in (methods or ["GET"]))
        self.max_size = max_size
        self._cache: dict[str, tuple[float, Any]] = {}

    def should_cache(self, method: str) -> bool:
        """Check if method should be cached."""
        return method.upper() in self.methods

    def make_key(self, method: str, url: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key from request details."""
        key_parts = [method.upper(), url]
        if params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None

        expires_at, value = self._cache[key]
        if time.time() > expires_at:
            # Expired - remove and return None
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with TTL."""
        # Simple LRU: remove oldest entries if at max size
        if len(self._cache) >= self.max_size:
            # Remove ~10% of oldest entries
            to_remove = self.max_size // 10 or 1
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[:to_remove]
            for k in sorted_keys:
                del self._cache[k]

        expires_at = time.time() + self.ttl
        self._cache[key] = (expires_at, value)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries (including expired)."""
        return len(self._cache)


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Thread-safe for async usage. Limits requests per second.

    Example:
        limiter = RateLimiter(rate=10)  # 10 requests/second

        async def make_request():
            await limiter.acquire()  # Blocks if rate exceeded
            return await do_request()
    """

    def __init__(self, rate: float, burst: float | None = None):
        """Initialize rate limiter.

        Args:
            rate: Maximum requests per second
            burst: Maximum burst size (default: rate)
        """
        self.rate = rate
        self.burst = burst or rate
        self._tokens = self.burst
        self._last_update = time.time()
        self._lock: asyncio.Lock | None = None  # Lazy init for async

    async def _get_lock(self):
        """Get or create async lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        import asyncio

        lock = await self._get_lock()

        async with lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return

            # Wait for tokens
            wait_time = (tokens - self._tokens) / self.rate
            await asyncio.sleep(wait_time)
            self._refill()
            self._tokens -= tokens

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if acquired, False if rate limited
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False


# =============================================================================
# Request Deduplicator
# =============================================================================


class RequestDeduplicator:
    """Deduplicates concurrent identical requests.

    When multiple identical requests are made concurrently, only one
    actually executes and the result is shared with all waiters.

    Example:
        dedup = RequestDeduplicator()

        async def fetch_user(id: int):
            key = f"user:{id}"
            async with dedup.dedupe(key):
                return await actual_fetch(id)
    """

    def __init__(self):
        self._pending: dict[str, Any] = {}  # key -> asyncio.Future
        self._lock = None  # Lazy init

    async def _get_lock(self):
        """Get or create async lock."""
        if self._lock is None:
            import asyncio

            self._lock = asyncio.Lock()
        return self._lock

    async def execute(
        self,
        key: str,
        fn: Any,  # Callable[[], Awaitable[T]]
    ) -> Any:
        """Execute function, deduplicating concurrent calls.

        Args:
            key: Unique key for this request
            fn: Async function to execute

        Returns:
            Result of the function
        """
        import asyncio

        lock = await self._get_lock()

        async with lock:
            if key in self._pending:
                # Another request is in progress, wait for it
                pending_future = self._pending[key]

        if key in self._pending:
            return await pending_future

        # We're the first, create future and execute
        async with lock:
            if key in self._pending:
                # Race condition, another request started
                return await self._pending[key]

            new_future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[key] = new_future

        try:
            result = await fn()
            new_future.set_result(result)
            return result
        except Exception as e:
            new_future.set_exception(e)
            raise
        finally:
            async with lock:
                self._pending.pop(key, None)
