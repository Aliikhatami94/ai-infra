"""Unit tests for MCP Server core functionality.

Tests cover:
- MCPServer initialization (strict mode, health path)
- Mount management (add_app, add_fastmcp, add_from_module, add_openapi, add_tools)
- Server lifecycle (lifespan, session managers)
- Transport handling (streamable_http, sse, websocket)
- Tool execution and registration
- Resource handling
- Utility functions

Phase 5.1 of ai-infra test plan.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.mcp.server.models import MCPMount

# Import the module
from ai_infra.mcp.server.server import (
    MCPServer,
    _py_type_to_json,
    import_object,
    normalize_mount,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server():
    """Create an MCPServer instance."""
    return MCPServer()


@pytest.fixture
def mock_app():
    """Create a mock ASGI app."""
    app = MagicMock()
    app.state = MagicMock()
    return app


@pytest.fixture
def mock_fastmcp():
    """Create a mock FastMCP instance."""
    mcp = MagicMock()
    mcp.streamable_http_app = MagicMock(return_value=MagicMock())
    mcp.sse_app = MagicMock(return_value=MagicMock())
    mcp.websocket_app = MagicMock(return_value=MagicMock())
    mcp.session_manager = MagicMock()
    return mcp


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    sm = MagicMock()
    sm.run = MagicMock(return_value=AsyncMock())
    return sm


# =============================================================================
# MCPServer Initialization Tests
# =============================================================================


class TestMCPServerInit:
    """Tests for MCPServer initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        server = MCPServer()

        assert server._strict is True
        assert server._health_path == "/health"
        assert server._mounts == []

    def test_init_strict_mode_false(self):
        """Test initialization with strict mode disabled."""
        server = MCPServer(strict=False)

        assert server._strict is False

    def test_init_custom_health_path(self):
        """Test initialization with custom health path."""
        server = MCPServer(health_path="/healthz")

        assert server._health_path == "/healthz"

    def test_init_empty_health_path(self):
        """Test initialization with empty health path disables health endpoint."""
        server = MCPServer(health_path="")

        assert server._health_path == ""


# =============================================================================
# MCPServer Mount Management Tests
# =============================================================================


class TestMCPServerMountManagement:
    """Tests for mount management methods."""

    def test_add_single_mount(self, server):
        """Test adding a single mount."""
        mount = MCPMount(path="/api", app=MagicMock())

        result = server.add(mount)

        assert result is server  # Fluent interface
        assert len(server._mounts) == 1
        assert server._mounts[0] == mount

    def test_add_multiple_mounts(self, server):
        """Test adding multiple mounts."""
        mount1 = MCPMount(path="/api1", app=MagicMock())
        mount2 = MCPMount(path="/api2", app=MagicMock())

        server.add(mount1, mount2)

        assert len(server._mounts) == 2

    def test_add_app_basic(self, server, mock_app):
        """Test add_app with basic parameters."""
        result = server.add_app("/api", mock_app)

        assert result is server
        assert len(server._mounts) == 1
        assert server._mounts[0].path == "/api"
        assert server._mounts[0].app is mock_app

    def test_add_app_with_name(self, server, mock_app):
        """Test add_app with custom name."""
        server.add_app("/api", mock_app, name="my-api")

        assert server._mounts[0].name == "my-api"

    def test_add_app_with_session_manager(self, server, mock_app, mock_session_manager):
        """Test add_app with session manager."""
        server.add_app("/api", mock_app, session_manager=mock_session_manager)

        assert server._mounts[0].session_manager is mock_session_manager
        assert server._mounts[0].require_manager is True

    def test_add_app_with_async_cleanup(self, server, mock_app):
        """Test add_app with async cleanup callback."""
        cleanup = AsyncMock()

        server.add_app("/api", mock_app, async_cleanup=cleanup)

        assert server._mounts[0].async_cleanup is cleanup

    def test_add_app_normalizes_path(self, server, mock_app):
        """Test add_app normalizes mount path."""
        server.add_app("api/", mock_app)

        assert server._mounts[0].path == "/api"


# =============================================================================
# MCPServer FastMCP Integration Tests
# =============================================================================


class TestMCPServerFastMCPIntegration:
    """Tests for FastMCP integration."""

    def test_add_fastmcp_streamable_http(self, server, mock_fastmcp):
        """Test add_fastmcp with streamable_http transport."""
        server.add_fastmcp(mock_fastmcp, "/api", transport="streamable_http")

        mock_fastmcp.streamable_http_app.assert_called_once()
        assert len(server._mounts) == 1

    def test_add_fastmcp_sse(self, server, mock_fastmcp):
        """Test add_fastmcp with SSE transport."""
        server.add_fastmcp(mock_fastmcp, "/api", transport="sse")

        mock_fastmcp.sse_app.assert_called_once()

    def test_add_fastmcp_websocket(self, server, mock_fastmcp):
        """Test add_fastmcp with WebSocket transport."""
        server.add_fastmcp(mock_fastmcp, "/api", transport="websocket")

        mock_fastmcp.websocket_app.assert_called_once()

    def test_add_fastmcp_invalid_transport(self, server, mock_fastmcp):
        """Test add_fastmcp with invalid transport raises error."""
        with pytest.raises(ValueError, match="Unknown transport"):
            server.add_fastmcp(mock_fastmcp, "/api", transport="invalid")

    def test_add_fastmcp_with_name(self, server, mock_fastmcp):
        """Test add_fastmcp with custom name."""
        server.add_fastmcp(mock_fastmcp, "/api", name="my-mcp")

        assert server._mounts[0].name == "my-mcp"

    def test_add_fastmcp_with_async_cleanup(self, server, mock_fastmcp):
        """Test add_fastmcp with async cleanup."""
        cleanup = AsyncMock()

        server.add_fastmcp(mock_fastmcp, "/api", async_cleanup=cleanup)

        assert server._mounts[0].async_cleanup is cleanup

    def test_add_fastmcp_sets_session_manager_on_state(self, server, mock_fastmcp):
        """Test add_fastmcp sets session manager on app state."""
        sub_app = mock_fastmcp.streamable_http_app.return_value
        sub_app.state = MagicMock(spec=[])
        # Initially no session_manager on state

        server.add_fastmcp(mock_fastmcp, "/api", transport="streamable_http")

        # Session manager should be set on the sub_app state
        assert hasattr(sub_app.state, "session_manager")


# =============================================================================
# MCPServer Module Import Tests
# =============================================================================


class TestMCPServerFromModule:
    """Tests for add_from_module method."""

    def test_add_from_module_with_fastmcp(self, server):
        """Test add_from_module with FastMCP object."""
        mock_mcp = MagicMock()
        mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
        mock_mcp.session_manager = MagicMock()

        with patch("ai_infra.mcp.server.server.import_object", return_value=mock_mcp):
            server.add_from_module("my.module", "/api", transport="streamable_http")

        assert len(server._mounts) == 1

    def test_add_from_module_with_asgi_app(self, server, mock_app):
        """Test add_from_module with ASGI app."""
        with patch("ai_infra.mcp.server.server.import_object", return_value=mock_app):
            server.add_from_module("my.module", "/api")

        assert len(server._mounts) == 1


# =============================================================================
# MCPServer Tools Tests
# =============================================================================


class TestMCPServerTools:
    """Tests for tool registration and handling."""

    def test_add_tools_basic(self, server):
        """Test add_tools with basic tools."""

        def my_tool(x: int) -> int:
            return x * 2

        with patch("ai_infra.mcp.server.server.mcp_from_functions") as mock_mcp_from:
            mock_mcp = MagicMock()
            mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
            mock_mcp.session_manager = MagicMock()
            mock_mcp_from.return_value = mock_mcp

            server.add_tools("/tools", tools=[my_tool], name="my-tools")

        mock_mcp_from.assert_called_once()
        assert len(server._mounts) == 1

    def test_add_tools_with_transport(self, server):
        """Test add_tools with custom transport."""
        with patch("ai_infra.mcp.server.server.mcp_from_functions") as mock_mcp_from:
            mock_mcp = MagicMock()
            mock_mcp.sse_app = MagicMock(return_value=MagicMock())
            mock_mcp_from.return_value = mock_mcp

            server.add_tools("/tools", tools=[], transport="sse")

        assert len(server._mounts) == 1


# =============================================================================
# MCPServer OpenAPI Integration Tests
# =============================================================================


class TestMCPServerOpenAPI:
    """Tests for OpenAPI integration."""

    def test_add_openapi_basic(self, server):
        """Test add_openapi with basic spec."""
        mock_mcp = MagicMock()
        mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
        mock_mcp.session_manager = MagicMock()

        with patch("ai_infra.mcp.server.server._mcp_from_openapi") as mock_from_openapi:
            mock_from_openapi.return_value = (mock_mcp, None)

            server.add_openapi("/api", {"openapi": "3.0.0"})

        mock_from_openapi.assert_called_once()
        assert len(server._mounts) == 1

    def test_add_openapi_with_cleanup(self, server):
        """Test add_openapi with async cleanup returned."""
        mock_mcp = MagicMock()
        mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
        mock_mcp.session_manager = MagicMock()
        mock_cleanup = AsyncMock()

        with patch("ai_infra.mcp.server.server._mcp_from_openapi") as mock_from_openapi:
            mock_from_openapi.return_value = (mock_mcp, mock_cleanup)

            server.add_openapi("/api", {"openapi": "3.0.0"})

        assert server._mounts[0].async_cleanup is mock_cleanup

    def test_add_openapi_with_report(self, server):
        """Test add_openapi with build report."""
        mock_mcp = MagicMock()
        mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
        mock_mcp.session_manager = MagicMock()
        mock_report = MagicMock()

        with patch("ai_infra.mcp.server.server._mcp_from_openapi") as mock_from_openapi:
            mock_from_openapi.return_value = (mock_mcp, None, mock_report)

            server.add_openapi("/api", {"openapi": "3.0.0"})

        assert mock_mcp.openapi_build_report == mock_report

    def test_add_openapi_with_filtering(self, server):
        """Test add_openapi with path filtering."""
        mock_mcp = MagicMock()
        mock_mcp.streamable_http_app = MagicMock(return_value=MagicMock())
        mock_mcp.session_manager = MagicMock()

        with patch("ai_infra.mcp.server.server._mcp_from_openapi") as mock_from_openapi:
            mock_from_openapi.return_value = (mock_mcp, None)

            server.add_openapi(
                "/api",
                {"openapi": "3.0.0"},
                include_paths=["/users/*"],
                exclude_methods=["DELETE"],
            )

        call_kwargs = mock_from_openapi.call_args[1]
        assert call_kwargs["include_paths"] == ["/users/*"]
        assert call_kwargs["exclude_methods"] == ["DELETE"]


# =============================================================================
# MCPServer Lifecycle Tests
# =============================================================================


class TestMCPServerLifecycle:
    """Tests for server lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifespan_starts_session_managers(self, server, mock_session_manager):
        """Test lifespan starts session managers."""
        mock_app = MagicMock()
        mock_app.state.session_manager = mock_session_manager

        # Create a proper async context manager for run()
        @contextlib.asynccontextmanager
        async def mock_run():
            yield

        mock_session_manager.run = mock_run

        server.add_app("/api", mock_app, session_manager=mock_session_manager, require_manager=True)

        async with server.lifespan(None):
            pass

        # Session manager should have been used

    @pytest.mark.asyncio
    async def test_lifespan_runs_async_cleanup(self, server, mock_app):
        """Test lifespan runs async cleanup on shutdown."""
        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        server.add_app("/api", mock_app, async_cleanup=cleanup, require_manager=False)

        async with server.lifespan(None):
            pass

        assert cleanup_called

    @pytest.mark.asyncio
    async def test_lifespan_multiple_cleanups(self, server):
        """Test lifespan runs multiple cleanups."""
        cleanup_order = []

        async def cleanup1():
            cleanup_order.append(1)

        async def cleanup2():
            cleanup_order.append(2)

        mock_app1 = MagicMock()
        mock_app2 = MagicMock()

        server.add_app("/api1", mock_app1, async_cleanup=cleanup1, require_manager=False)
        server.add_app("/api2", mock_app2, async_cleanup=cleanup2, require_manager=False)

        async with server.lifespan(None):
            pass

        assert len(cleanup_order) == 2


# =============================================================================
# MCPServer Session Manager Tests
# =============================================================================


class TestMCPServerSessionManager:
    """Tests for session manager handling."""

    def test_iter_unique_session_managers_empty(self, server):
        """Test iterating session managers with no mounts."""
        managers = list(server._iter_unique_session_managers())

        assert managers == []

    def test_iter_unique_session_managers_skips_not_required(self, server, mock_app):
        """Test skips mounts that don't require session manager."""
        server.add_app("/api", mock_app, require_manager=False)

        managers = list(server._iter_unique_session_managers())

        assert managers == []

    def test_iter_unique_session_managers_deduplicates(self, server, mock_session_manager):
        """Test deduplicates shared session managers."""
        mock_app1 = MagicMock()
        mock_app1.state.session_manager = mock_session_manager
        mock_app2 = MagicMock()
        mock_app2.state.session_manager = mock_session_manager

        server.add_app(
            "/api1", mock_app1, session_manager=mock_session_manager, require_manager=True
        )
        server.add_app(
            "/api2", mock_app2, session_manager=mock_session_manager, require_manager=True
        )

        managers = list(server._iter_unique_session_managers())

        assert len(managers) == 1

    def test_iter_unique_session_managers_strict_mode_raises(self):
        """Test strict mode raises when session manager required but missing."""
        server = MCPServer(strict=True)
        # Create app with no session_manager attribute at all
        mock_app = MagicMock()
        mock_app.state = MagicMock(spec=[])  # No session_manager
        del mock_app.state.session_manager  # Ensure it doesn't exist
        mount = MCPMount(path="/api", app=mock_app, require_manager=True, session_manager=None)
        server._mounts.append(mount)

        with pytest.raises(RuntimeError, match="has no session_manager"):
            list(server._iter_unique_session_managers())

    def test_iter_unique_session_managers_non_strict_skips(self):
        """Test non-strict mode skips mounts without session manager."""
        server = MCPServer(strict=False)
        # Create app with no session_manager attribute at all
        mock_app = MagicMock()
        mock_app.state = MagicMock(spec=[])  # No session_manager
        mount = MCPMount(path="/api", app=mock_app, require_manager=True, session_manager=None)
        server._mounts.append(mount)

        managers = list(server._iter_unique_session_managers())

        assert managers == []


# =============================================================================
# MCPServer Mount All Tests
# =============================================================================


class TestMCPServerMountAll:
    """Tests for mount_all method."""

    def test_mount_all_mounts_apps(self, server, mock_app):
        """Test mount_all mounts all apps."""
        root_app = MagicMock()
        server.add_app("/api", mock_app)

        server.mount_all(root_app)

        root_app.mount.assert_called_once_with("/api", mock_app)

    def test_mount_all_multiple_apps(self, server):
        """Test mount_all with multiple apps."""
        root_app = MagicMock()
        app1 = MagicMock()
        app2 = MagicMock()

        server.add_app("/api1", app1)
        server.add_app("/api2", app2)

        server.mount_all(root_app)

        assert root_app.mount.call_count == 2


# =============================================================================
# MCPServer FastAPI Attachment Tests
# =============================================================================


class TestMCPServerFastAPIAttachment:
    """Tests for FastAPI attachment."""

    def test_attach_to_fastapi_mounts_all(self, server, mock_app):
        """Test attach_to_fastapi mounts all apps."""
        fastapi_app = MagicMock()
        fastapi_app.router.lifespan_context = None

        server.add_app("/api", mock_app)
        server.attach_to_fastapi(fastapi_app)

        fastapi_app.mount.assert_called_once()

    def test_attach_to_fastapi_sets_lifespan(self, server):
        """Test attach_to_fastapi sets lifespan context."""
        fastapi_app = MagicMock()
        fastapi_app.router.lifespan_context = None

        server.attach_to_fastapi(fastapi_app)

        assert fastapi_app.router.lifespan_context is not None

    def test_attach_to_fastapi_preserves_existing_lifespan(self, server):
        """Test attach_to_fastapi preserves existing lifespan."""
        fastapi_app = MagicMock()
        existing_lifespan = MagicMock()
        fastapi_app.router.lifespan_context = existing_lifespan

        server.attach_to_fastapi(fastapi_app)

        # Should set a new merged lifespan
        assert fastapi_app.router.lifespan_context is not None
        assert fastapi_app.router.lifespan_context is not existing_lifespan


# =============================================================================
# MCPServer OpenMCP Discovery Tests
# =============================================================================


class TestMCPServerOpenMCP:
    """Tests for OpenMCP discovery."""

    def test_get_openmcp_empty(self, server):
        """Test get_openmcp with no mounts."""
        spec = server.get_openmcp()

        assert spec["openmcp"] == "1.0.0"
        assert spec["tools"] == []
        assert spec["servers"] == []

    def test_get_openmcp_with_mounts(self, server, mock_app):
        """Test get_openmcp with mounts."""
        server.add_app("/api", mock_app, name="test-api")

        spec = server.get_openmcp()

        assert len(spec["servers"]) == 1
        assert spec["servers"][0]["path"] == "/api"
        assert spec["servers"][0]["name"] == "test-api"

    def test_get_openmcp_extracts_tools(self, server):
        """Test get_openmcp extracts tools from mounted apps."""
        mock_app = MagicMock()
        mock_tool = MagicMock()
        mock_tool.description = "Test tool"

        mock_tool_manager = MagicMock()
        mock_tool_manager._tools = {"test_tool": mock_tool}

        mock_mcp = MagicMock()
        mock_mcp._tool_manager = mock_tool_manager

        mock_app.state.mcp = mock_mcp
        server.add_app("/api", mock_app)

        spec = server.get_openmcp()

        assert len(spec["tools"]) == 1
        assert spec["tools"][0]["name"] == "test_tool"
        assert spec["tools"][0]["description"] == "Test tool"


# =============================================================================
# MCPServer ASGI Root Tests
# =============================================================================


class TestMCPServerASGIRoot:
    """Tests for ASGI root app building."""

    def test_build_asgi_root_creates_starlette_app(self, server):
        """Test build_asgi_root creates Starlette app."""
        app = server.build_asgi_root()

        # Should return an app object
        assert app is not None

    def test_build_asgi_root_has_routes(self, server):
        """Test build_asgi_root creates app with routes."""
        app = server.build_asgi_root()

        # App should have router with routes
        assert hasattr(app, "router")

    def test_build_asgi_root_with_mounts(self, server, mock_app):
        """Test build_asgi_root with mounted apps."""
        server.add_app("/api", mock_app)

        app = server.build_asgi_root()

        assert app is not None


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestNormalizeMountPath:
    """Tests for normalize_mount utility."""

    def test_normalize_adds_leading_slash(self):
        """Test adds leading slash."""
        assert normalize_mount("api") == "/api"

    def test_normalize_removes_trailing_slash(self):
        """Test removes trailing slash."""
        assert normalize_mount("/api/") == "/api"

    def test_normalize_handles_both_slashes(self):
        """Test handles both leading and trailing."""
        assert normalize_mount("api/") == "/api"

    def test_normalize_removes_mcp_suffix(self):
        """Test removes /mcp suffix."""
        assert normalize_mount("/api/mcp") == "/api"

    def test_normalize_root_path(self):
        """Test normalizes to root path."""
        assert normalize_mount("/") == "/"
        assert normalize_mount("") == "/"

    def test_normalize_nested_path(self):
        """Test handles nested paths."""
        assert normalize_mount("/v1/api/tools") == "/v1/api/tools"


class TestImportObject:
    """Tests for import_object utility."""

    def test_import_object_with_colon_notation(self):
        """Test import with module:attr notation."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.my_obj = "test_object"
            mock_import.return_value = mock_module

            result = import_object("my.module:my_obj")

            assert result == "test_object"

    def test_import_object_with_attr_param(self):
        """Test import with separate attr parameter."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.my_obj = "test_object"
            mock_import.return_value = mock_module

            result = import_object("my.module", attr="my_obj")

            assert result == "test_object"

    def test_import_object_finds_mcp_attr(self):
        """Test import finds 'mcp' attr by default."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=["mcp"])
            mock_module.mcp = "mcp_object"
            mock_import.return_value = mock_module

            result = import_object("my.module")

            assert result == "mcp_object"

    def test_import_object_finds_app_attr(self):
        """Test import finds 'app' attr as fallback."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=["app"])
            mock_module.app = "app_object"
            mock_import.return_value = mock_module

            result = import_object("my.module")

            assert result == "app_object"

    def test_import_object_raises_on_missing_attr(self):
        """Test raises when attribute not found."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=[])  # Empty spec = no attributes
            mock_import.return_value = mock_module

            with pytest.raises(ImportError, match="not found"):
                import_object("my.module:my_missing_attr")

    def test_import_object_raises_on_no_obvious_object(self):
        """Test raises when no mcp/app found."""
        with patch("ai_infra.mcp.server.server.importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=[])  # No mcp or app
            mock_import.return_value = mock_module

            with pytest.raises(ImportError, match="No obvious object found"):
                import_object("my.module")


class TestPyTypeToJson:
    """Tests for _py_type_to_json utility."""

    def test_str_type(self):
        """Test string type conversion."""
        assert _py_type_to_json(str) == "string"

    def test_int_type(self):
        """Test integer type conversion."""
        assert _py_type_to_json(int) == "integer"

    def test_float_type(self):
        """Test float type conversion."""
        assert _py_type_to_json(float) == "number"

    def test_bool_type(self):
        """Test boolean type conversion."""
        assert _py_type_to_json(bool) == "boolean"

    def test_list_type(self):
        """Test list type conversion."""
        assert _py_type_to_json(list) == "array"

    def test_dict_type(self):
        """Test dict type conversion."""
        assert _py_type_to_json(dict) == "object"

    def test_none_type(self):
        """Test None type conversion."""
        assert _py_type_to_json(None) == "null"

    def test_unknown_type_defaults_to_string(self):
        """Test unknown type defaults to string."""

        class CustomType:
            pass

        assert _py_type_to_json(CustomType) == "string"


# =============================================================================
# MCPServer Error Handling Tests
# =============================================================================


class TestMCPServerErrorHandling:
    """Tests for error handling."""

    def test_add_openapi_handles_errors_gracefully(self, server):
        """Test add_openapi handles errors."""
        with patch("ai_infra.mcp.server.server._mcp_from_openapi") as mock_from_openapi:
            mock_from_openapi.side_effect = ValueError("Invalid spec")

            with pytest.raises(ValueError, match="Invalid spec"):
                server.add_openapi("/api", {"invalid": "spec"})


# =============================================================================
# MCPServer Uvicorn Runner Tests
# =============================================================================


class TestMCPServerUvicornRunner:
    """Tests for Uvicorn runner."""

    def test_run_uvicorn_calls_uvicorn(self, server):
        """Test run_uvicorn calls uvicorn.run."""
        import sys

        mock_uvicorn = MagicMock()
        sys.modules["uvicorn"] = mock_uvicorn

        try:
            server.run_uvicorn(host="127.0.0.1", port=9000, log_level="debug")

            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "127.0.0.1"
            assert call_args[1]["port"] == 9000
            assert call_args[1]["log_level"] == "debug"
        finally:
            del sys.modules["uvicorn"]


# =============================================================================
# MCPServer Edge Cases Tests
# =============================================================================


class TestMCPServerEdgeCases:
    """Tests for edge cases."""

    def test_chained_add_calls(self, server):
        """Test chained add calls (fluent interface)."""
        mount1 = MCPMount(path="/api1", app=MagicMock())
        mount2 = MCPMount(path="/api2", app=MagicMock())

        result = server.add(mount1).add(mount2)

        assert result is server
        assert len(server._mounts) == 2

    def test_empty_mounts_lifespan(self, server):
        """Test lifespan with no mounts."""

        async def run_lifespan():
            async with server.lifespan(None):
                pass

        asyncio.get_event_loop().run_until_complete(run_lifespan())

    def test_add_app_auto_require_manager_detection(self, server, mock_session_manager):
        """Test add_app auto-detects require_manager from session_manager."""
        mock_app = MagicMock()
        mock_app.state.session_manager = mock_session_manager

        server.add_app("/api", mock_app, session_manager=mock_session_manager)

        assert server._mounts[0].require_manager is True

    def test_add_app_no_session_manager(self, server, mock_app):
        """Test add_app without session manager sets require_manager to False."""
        mock_app.state = MagicMock(spec=[])  # No session_manager

        server.add_app("/api", mock_app)

        assert server._mounts[0].require_manager is False
