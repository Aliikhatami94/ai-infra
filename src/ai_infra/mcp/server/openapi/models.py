from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "AuthConfig",
    "BuildReport",
    "OpReport",
    "OpenAPIOptions",
    "OpenAPISpec",
    "Operation",
    "OperationContext",
]

OpenAPISpec = dict[str, Any]
Operation = dict[str, Any]


# =============================================================================
# Authentication Configuration
# =============================================================================


@dataclass
class AuthConfig:
    """Authentication configuration for OpenAPI→MCP.

    Supports multiple auth schemes:
    - Header-based API keys
    - Query parameter API keys
    - Basic auth (username/password)
    - Bearer tokens
    - Dynamic auth (callable that returns token)

    Example:
        # API Key in header
        auth = AuthConfig(headers={"Authorization": "Bearer sk-xxx"})

        # API Key in query
        auth = AuthConfig(query={"api_key": "xxx"})

        # Basic auth
        auth = AuthConfig(basic=("username", "password"))

        # Bearer token
        auth = AuthConfig(bearer="sk-xxx")

        # Dynamic auth (called before each request)
        async def get_token() -> str:
            return await refresh_my_token()
        auth = AuthConfig(bearer_fn=get_token)
    """

    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    basic: tuple | None = None  # (username, password)
    bearer: str | None = None
    bearer_fn: Callable[[], Any] | None = None  # Async or sync callable

    @classmethod
    def from_value(cls, value: Any) -> AuthConfig:
        """Create AuthConfig from various input types.

        Supports:
        - AuthConfig: Returns as-is
        - Dict: Headers
        - Tuple: Basic auth (username, password)
        - String: Bearer token
        - Callable: Dynamic auth function
        - None: No auth
        """
        if value is None:
            return cls()
        if isinstance(value, AuthConfig):
            return value
        if isinstance(value, dict):
            return cls(headers=value)
        if isinstance(value, tuple) and len(value) == 2:
            return cls(basic=value)
        if isinstance(value, str):
            return cls(bearer=value)
        if callable(value):
            return cls(bearer_fn=value)
        raise ValueError(f"Unsupported auth type: {type(value)}")


# =============================================================================
# Filtering Options
# =============================================================================


@dataclass
class OpenAPIOptions:
    """Options for OpenAPI→MCP tool generation.

    Provides flexible filtering and customization:
    - Filter by paths (glob patterns)
    - Filter by HTTP methods
    - Filter by OpenAPI tags
    - Filter by operationId
    - Custom tool naming and descriptions

    Example:
        options = OpenAPIOptions(
            tool_prefix="github",
            include_paths=["/repos/*", "/users/*"],
            exclude_paths=["/admin/*"],
            include_methods=["GET", "POST"],
            exclude_tags=["deprecated"],
        )
    """

    # Tool naming
    tool_prefix: str | None = None
    tool_name_fn: Callable[[str, str, dict], str] | None = None
    tool_description_fn: Callable[[dict], str] | None = None

    # Path filtering (glob patterns)
    include_paths: list[str] | None = None
    exclude_paths: list[str] | None = None

    # Method filtering
    include_methods: list[str] | None = None
    exclude_methods: list[str] | None = None

    # Tag filtering
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None

    # OperationId filtering
    include_operations: list[str] | None = None
    exclude_operations: list[str] | None = None

    # Auth
    auth: AuthConfig | None = None
    endpoint_auth: dict[str, Any] | None = None  # Pattern -> AuthConfig

    # Request configuration
    timeout: float | None = None  # Request timeout in seconds (default: 30)
    retries: int = 0  # Number of retries on transient failures

    # Rate limiting
    rate_limit: float | None = None  # Max requests per second (None = unlimited)
    rate_limit_retry: bool = True  # Retry on 429 Too Many Requests
    rate_limit_max_retries: int = 3  # Max retries on 429

    # Caching & Performance
    cache_ttl: float | None = None  # Cache TTL in seconds (None = no caching)
    cache_methods: list[str] | None = None  # Methods to cache (default: ["GET"])
    dedupe_requests: bool = False  # Deduplicate concurrent identical requests

    # Pagination
    auto_paginate: bool = False  # Automatically fetch all pages
    max_pages: int = 10  # Maximum pages to fetch when auto-paginating

    def should_include_operation(
        self,
        path: str,
        method: str,
        operation: dict,
    ) -> bool:
        """Check if an operation should be included based on filters."""
        method_upper = method.upper()

        # Method filters
        if self.include_methods:
            if method_upper not in [m.upper() for m in self.include_methods]:
                return False
        if self.exclude_methods:
            if method_upper in [m.upper() for m in self.exclude_methods]:
                return False

        # Path filters
        if self.include_paths:
            if not any(fnmatch.fnmatch(path, p) for p in self.include_paths):
                return False
        if self.exclude_paths:
            if any(fnmatch.fnmatch(path, p) for p in self.exclude_paths):
                return False

        # Tag filters
        tags = operation.get("tags") or []
        if self.include_tags:
            if not any(t in self.include_tags for t in tags):
                return False
        if self.exclude_tags:
            if any(t in self.exclude_tags for t in tags):
                return False

        # OperationId filters
        op_id = operation.get("operationId")
        if self.include_operations:
            if op_id not in self.include_operations:
                return False
        if self.exclude_operations:
            if op_id in self.exclude_operations:
                return False

        return True

    def get_tool_name(
        self,
        default_name: str,
        method: str,
        path: str,
        operation: dict,
    ) -> str:
        """Get tool name, applying prefix and custom function."""
        # Custom function takes precedence
        if self.tool_name_fn:
            name = self.tool_name_fn(method, path, operation)
        else:
            name = default_name

        # Apply prefix
        if self.tool_prefix:
            name = f"{self.tool_prefix}_{name}"

        return name

    def get_tool_description(
        self,
        default_description: str,
        operation: dict,
    ) -> str:
        """Get tool description, applying custom function."""
        if self.tool_description_fn:
            return self.tool_description_fn(operation)
        return default_description

    def get_auth_for_path(self, path: str) -> AuthConfig | None:
        """Get auth config for a specific path."""
        if self.endpoint_auth:
            for pattern, auth_value in self.endpoint_auth.items():
                if fnmatch.fnmatch(path, pattern):
                    if auth_value is None:
                        return None  # Explicitly no auth
                    return AuthConfig.from_value(auth_value)
        return self.auth


class OperationContext(BaseModel):
    name: str
    description: str
    method: str
    path: str
    path_params: list[dict[str, Any]] = Field(default_factory=list)
    query_params: list[dict[str, Any]] = Field(default_factory=list)
    header_params: list[dict[str, Any]] = Field(default_factory=list)
    cookie_params: list[dict[str, Any]] = Field(default_factory=list)
    wants_body: bool = False
    body_content_type: str | None = None
    body_required: bool = False

    def full_description(self) -> str:
        return self.description


@dataclass
class OpReport:
    operation_id: str | None
    tool_name: str
    method: str
    path: str
    base_url: str
    base_url_source: str  # override | operation | path | root | none
    has_body: bool
    body_content_type: str | None
    body_required: bool
    params: dict[str, int]
    security: dict[str, Any]
    input_model_fields: int = 0  # number of input fields
    media_types_seen: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BuildReport:
    title: str
    total_ops: int = 0
    registered_tools: int = 0
    skipped_ops: int = 0
    filtered_ops: int = 0  # Operations filtered by options
    warnings: list[str] = field(default_factory=list)
    ops: list[OpReport] = field(default_factory=list)

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, (BuildReport, OpReport)):
                return o.__dict__
            return str(o)

        return json.dumps(self, default=_default, indent=2)
