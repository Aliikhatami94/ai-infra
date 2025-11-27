from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "OpenAPISpec",
    "OperationContext",
    "Operation",
    "OpReport",
    "BuildReport",
    "OpenAPIOptions",
    "AuthConfig",
]

OpenAPISpec = Dict[str, Any]
Operation = Dict[str, Any]


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

    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, str] = field(default_factory=dict)
    basic: Optional[tuple] = None  # (username, password)
    bearer: Optional[str] = None
    bearer_fn: Optional[Callable[[], Any]] = None  # Async or sync callable

    @classmethod
    def from_value(cls, value: Any) -> "AuthConfig":
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
    tool_prefix: Optional[str] = None
    tool_name_fn: Optional[Callable[[str, str, dict], str]] = None
    tool_description_fn: Optional[Callable[[dict], str]] = None

    # Path filtering (glob patterns)
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None

    # Method filtering
    include_methods: Optional[List[str]] = None
    exclude_methods: Optional[List[str]] = None

    # Tag filtering
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None

    # OperationId filtering
    include_operations: Optional[List[str]] = None
    exclude_operations: Optional[List[str]] = None

    # Auth
    auth: Optional[AuthConfig] = None
    endpoint_auth: Optional[Dict[str, Any]] = None  # Pattern -> AuthConfig

    # Request configuration
    timeout: Optional[float] = None  # Request timeout in seconds (default: 30)
    retries: int = 0  # Number of retries on transient failures

    # Rate limiting
    rate_limit: Optional[float] = None  # Max requests per second (None = unlimited)
    rate_limit_retry: bool = True  # Retry on 429 Too Many Requests
    rate_limit_max_retries: int = 3  # Max retries on 429

    # Caching & Performance
    cache_ttl: Optional[float] = None  # Cache TTL in seconds (None = no caching)
    cache_methods: Optional[List[str]] = None  # Methods to cache (default: ["GET"])
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

    def get_auth_for_path(self, path: str) -> Optional[AuthConfig]:
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
    path_params: List[Dict[str, Any]] = Field(default_factory=list)
    query_params: List[Dict[str, Any]] = Field(default_factory=list)
    header_params: List[Dict[str, Any]] = Field(default_factory=list)
    cookie_params: List[Dict[str, Any]] = Field(default_factory=list)
    wants_body: bool = False
    body_content_type: Optional[str] = None
    body_required: bool = False

    def full_description(self) -> str:
        return self.description


@dataclass
class OpReport:
    operation_id: Optional[str]
    tool_name: str
    method: str
    path: str
    base_url: str
    base_url_source: str  # override | operation | path | root | none
    has_body: bool
    body_content_type: Optional[str]
    body_required: bool
    params: Dict[str, int]
    security: Dict[str, Any]
    input_model_fields: int = 0  # number of input fields
    media_types_seen: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BuildReport:
    title: str
    total_ops: int = 0
    registered_tools: int = 0
    skipped_ops: int = 0
    filtered_ops: int = 0  # Operations filtered by options
    warnings: List[str] = field(default_factory=list)
    ops: List[OpReport] = field(default_factory=list)

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, (BuildReport, OpReport)):
                return o.__dict__
            return str(o)

        return json.dumps(self, default=_default, indent=2)
