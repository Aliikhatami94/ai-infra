from __future__ import annotations

import inspect
import os
import textwrap
from collections.abc import Awaitable, Callable, Iterable, Sequence

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

ToolFn = Callable[..., str | Awaitable[str]]


class ToolDef(BaseModel):
    fn: ToolFn | None = Field(default=None, exclude=True)
    name: str | None = None
    description: str | None = None


def _auto_detect_hosts() -> list[str]:
    """Auto-detect allowed hosts from environment.

    Works across all deployment platforms: Railway, Render, Fly.io, Heroku, etc.

    Returns:
        List of host patterns to allow (e.g., ["127.0.0.1:*", "api.example.com:*"])
    """
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]  # Always allow localhost

    # Common environment variables across platforms
    env_vars = [
        "RAILWAY_PUBLIC_DOMAIN",  # Railway
        "RAILWAY_STATIC_URL",  # Railway (includes https://)
        "RENDER_EXTERNAL_HOSTNAME",  # Render
        "FLY_APP_NAME",  # Fly.io (construct from app name)
        "HEROKU_APP_NAME",  # Heroku
        "VERCEL_URL",  # Vercel
        "HOST",  # Generic
        "PUBLIC_URL",  # Generic
        "APP_URL",  # Generic
    ]

    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if not value:
            continue

        # Clean up the value
        value = value.replace("https://", "").replace("http://", "").strip("/")

        # Special handling for Fly.io (needs .fly.dev suffix)
        if var == "FLY_APP_NAME" and value and ".fly.dev" not in value:
            value = f"{value}.fly.dev"

        # Special handling for Heroku (needs .herokuapp.com suffix)
        if var == "HEROKU_APP_NAME" and value and ".herokuapp.com" not in value:
            value = f"{value}.herokuapp.com"

        if value:
            hosts.append(f"{value}:*")

    return hosts


def _auto_detect_origins() -> list[str]:
    """Auto-detect allowed origins from environment.

    Returns:
        List of origin patterns (e.g., ["http://localhost:*", "https://api.example.com"])
    """
    origins = [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
        "https://127.0.0.1:*",
        "https://localhost:*",
    ]

    # Check for public URLs in environment
    env_vars = [
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_STATIC_URL",
        "RENDER_EXTERNAL_HOSTNAME",
        "PUBLIC_URL",
        "APP_URL",
        "VERCEL_URL",
    ]

    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if not value:
            continue

        # If it already has a protocol, use as-is
        if value.startswith("http://") or value.startswith("https://"):
            origins.append(value.rstrip("/"))
        else:
            # Assume https for production domains
            origins.append(f"https://{value}".rstrip("/"))

    return origins


class MCPSecuritySettings:
    """Security settings for MCP servers with automatic environment detection.

    By default, auto-detects the deployment environment and configures
    appropriate security settings. Works with Railway, Render, Fly.io,
    Heroku, Vercel, and other platforms.

    Examples:
        # Auto-detect (recommended - works everywhere)
        mcp = mcp_from_functions(name="my-mcp", functions=[my_tool])

        # Disable security for development
        security = MCPSecuritySettings(enable_security=False)

        # Custom domains (overrides auto-detection)
        security = MCPSecuritySettings(domains=["api.example.com"])
    """

    def __init__(
        self,
        *,
        domains: Sequence[str] | None = None,
        enable_security: bool = True,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
    ):
        """Create security settings.

        Args:
            domains: Custom domains to allow (e.g., ["api.example.com"]).
                    If not provided, auto-detects from environment.
            enable_security: Whether to enable DNS rebinding protection.
                           Set to False to allow all hosts (dev only).
            allowed_hosts: Advanced: Override auto-detected host patterns.
            allowed_origins: Advanced: Override auto-detected origin patterns.
        """
        self._enable_dns_rebinding_protection = enable_security

        if not enable_security:
            # Disable security - allow everything
            self._allowed_hosts = []
            self._allowed_origins = []
        elif allowed_hosts is not None or allowed_origins is not None:
            # Manual override
            self._allowed_hosts = list(allowed_hosts) if allowed_hosts else []
            self._allowed_origins = list(allowed_origins) if allowed_origins else []
        elif domains:
            # Use provided domains
            hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
            origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
            for domain in domains:
                hosts.append(f"{domain}:*")
                origins.append(f"https://{domain}")
            self._allowed_hosts = hosts
            self._allowed_origins = origins
        else:
            # Auto-detect from environment
            self._allowed_hosts = _auto_detect_hosts()
            self._allowed_origins = _auto_detect_origins()

    def to_transport_settings(self) -> TransportSecuritySettings:
        """Convert to the underlying TransportSecuritySettings.

        Always returns a TransportSecuritySettings object to ensure
        proper security configuration is applied.
        """
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=self._enable_dns_rebinding_protection,
            allowed_hosts=self._allowed_hosts,
            allowed_origins=self._allowed_origins,
        )

    @property
    def allowed_hosts(self) -> list[str]:
        """Get the list of allowed hosts."""
        return self._allowed_hosts.copy()

    @property
    def allowed_origins(self) -> list[str]:
        """Get the list of allowed origins."""
        return self._allowed_origins.copy()

    @property
    def enabled(self) -> bool:
        """Check if security is enabled."""
        return self._enable_dns_rebinding_protection


def _describe(fn: Callable[..., object], fallback: str) -> str:
    doc = inspect.getdoc(fn) or ""
    doc = textwrap.dedent(doc).strip()
    return doc or f"{fallback} tool"


def mcp_from_functions(
    *,
    name: str | None,
    functions: Iterable[ToolFn | ToolDef] | None,
    security: MCPSecuritySettings | None = None,
) -> FastMCP:
    """Create a FastMCP server from plain functions with automatic security.

    Security is auto-configured based on the deployment environment:
    - Detects Railway, Render, Fly.io, Heroku, Vercel, etc.
    - Always allows localhost for local development
    - No manual configuration needed in most cases

    Args:
        name: Server name.
        functions: Tool functions or ToolDef objects.
        security: Optional security settings. If not provided, auto-detects
                 from environment. Use MCPSecuritySettings(enable_security=False)
                 to disable security (development only).

    Examples:
        # Auto-detect security (recommended)
        mcp = mcp_from_functions(name="my-mcp", functions=[my_tool])

        # Disable security for dev
        mcp = mcp_from_functions(
            name="my-mcp",
            functions=[my_tool],
            security=MCPSecuritySettings(enable_security=False)
        )

        # Custom domains
        mcp = mcp_from_functions(
            name="my-mcp",
            functions=[my_tool],
            security=MCPSecuritySettings(domains=["api.example.com"])
        )
    """
    # Default to auto-detect security if not provided
    if security is None:
        security = MCPSecuritySettings()

    resolved_security = security.to_transport_settings()
    server = FastMCP(name=name, transport_security=resolved_security)
    if not functions:
        return server

    seen: set[str] = set()
    for item in functions:
        if isinstance(item, ToolDef):
            fn = getattr(item, "fn", None)
            if fn is None:
                continue  # or raise ValueError("ToolDef.fn is required")
            tool_name = getattr(item, "name", None) or fn.__name__
            desc = (getattr(item, "description", None) or _describe(fn, tool_name)).strip()
        else:
            fn = item
            tool_name = fn.__name__
            desc = _describe(fn, tool_name)

        # best-effort dedupe; last one wins
        if tool_name in seen:
            # If FastMCP ever supports removal/replacement, we could call it here.
            pass
        seen.add(tool_name)

        server.add_tool(name=tool_name, description=desc, fn=fn)

    return server
