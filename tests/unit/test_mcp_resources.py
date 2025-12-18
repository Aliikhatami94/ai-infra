"""Unit tests for MCP resources support."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.mcp.client.resources import (
    MCPResource,
    ResourceInfo,
    convert_mcp_resource,
    list_mcp_resources,
    load_mcp_resources,
)

# ---------------------------------------------------------------------------
# Mock MCP types for testing
# ---------------------------------------------------------------------------


class MockTextResourceContents:
    """Mock MCP TextResourceContents."""

    def __init__(self, text: str, mime_type: str | None = None):
        self.text = text
        self.mimeType = mime_type


class MockBlobResourceContents:
    """Mock MCP BlobResourceContents (base64 encoded)."""

    def __init__(self, data: bytes, mime_type: str | None = None):
        self.blob = base64.b64encode(data).decode("ascii")
        self.mimeType = mime_type


class MockResource:
    """Mock MCP Resource."""

    def __init__(
        self,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
    ):
        self.uri = uri
        self.name = name
        self.description = description
        self.mimeType = mime_type


class MockReadResourceResult:
    """Mock MCP ReadResourceResult."""

    def __init__(self, contents: list):
        self.contents = contents


class MockListResourcesResult:
    """Mock MCP ListResourcesResult."""

    def __init__(self, resources: list[MockResource]):
        self.resources = resources


# ---------------------------------------------------------------------------
# ResourceInfo tests
# ---------------------------------------------------------------------------


class TestResourceInfo:
    """Tests for ResourceInfo dataclass."""

    def test_create_resource_info(self):
        """Test creating ResourceInfo."""
        info = ResourceInfo(
            uri="file:///config.json",
            name="Configuration",
            description="App configuration file",
            mime_type="application/json",
        )
        assert info.uri == "file:///config.json"
        assert info.name == "Configuration"
        assert info.description == "App configuration file"
        assert info.mime_type == "application/json"

    def test_from_mcp_resource(self):
        """Test creating ResourceInfo from MCP Resource."""
        mcp_resource = MockResource(
            uri="file:///data.csv",
            name="Data File",
            description="Raw data",
            mime_type="text/csv",
        )
        info = ResourceInfo.from_mcp_resource(mcp_resource)

        assert info.uri == "file:///data.csv"
        assert info.name == "Data File"
        assert info.description == "Raw data"
        assert info.mime_type == "text/csv"

    def test_from_mcp_resource_minimal(self):
        """Test creating ResourceInfo from minimal MCP Resource."""
        mcp_resource = MockResource(uri="file:///test.txt")
        info = ResourceInfo.from_mcp_resource(mcp_resource)

        assert info.uri == "file:///test.txt"
        assert info.name is None
        assert info.description is None
        assert info.mime_type is None


# ---------------------------------------------------------------------------
# MCPResource tests
# ---------------------------------------------------------------------------


class TestMCPResource:
    """Tests for MCPResource dataclass."""

    def test_text_resource(self):
        """Test creating text resource."""
        resource = MCPResource(
            uri="file:///config.json",
            mime_type="application/json",
            data='{"key": "value"}',
        )
        assert resource.is_text is True
        assert resource.is_binary is False
        assert resource.data == '{"key": "value"}'

    def test_binary_resource(self):
        """Test creating binary resource."""
        binary_data = b"\x89PNG\r\n\x1a\n"
        resource = MCPResource(
            uri="file:///image.png",
            mime_type="image/png",
            data=binary_data,
        )
        assert resource.is_text is False
        assert resource.is_binary is True
        assert resource.data == binary_data

    def test_size_text(self):
        """Test size property for text resource."""
        resource = MCPResource(
            uri="file:///test.txt",
            mime_type="text/plain",
            data="Hello, World!",
        )
        assert resource.size == 13

    def test_size_binary(self):
        """Test size property for binary resource."""
        resource = MCPResource(
            uri="file:///data.bin",
            mime_type="application/octet-stream",
            data=b"\x00\x01\x02\x03",
        )
        assert resource.size == 4

    def test_as_text_from_text(self):
        """Test as_text for text resource."""
        resource = MCPResource(
            uri="file:///test.txt",
            mime_type="text/plain",
            data="Hello!",
        )
        assert resource.as_text() == "Hello!"

    def test_as_text_from_binary(self):
        """Test as_text for binary resource."""
        resource = MCPResource(
            uri="file:///test.txt",
            mime_type="text/plain",
            data=b"Hello!",
        )
        assert resource.as_text() == "Hello!"

    def test_as_bytes_from_bytes(self):
        """Test as_bytes for binary resource."""
        resource = MCPResource(
            uri="file:///data.bin",
            mime_type="application/octet-stream",
            data=b"\x00\x01\x02",
        )
        assert resource.as_bytes() == b"\x00\x01\x02"

    def test_as_bytes_from_text(self):
        """Test as_bytes for text resource."""
        resource = MCPResource(
            uri="file:///test.txt",
            mime_type="text/plain",
            data="Hello!",
        )
        assert resource.as_bytes() == b"Hello!"


# ---------------------------------------------------------------------------
# convert_mcp_resource tests
# ---------------------------------------------------------------------------


class TestConvertMcpResource:
    """Tests for convert_mcp_resource function."""

    def test_convert_text_resource(self):
        """Test converting text resource contents."""
        contents = MockTextResourceContents(
            text='{"key": "value"}',
            mime_type="application/json",
        )
        resource = convert_mcp_resource("file:///config.json", contents)

        assert resource.uri == "file:///config.json"
        assert resource.mime_type == "application/json"
        assert resource.is_text is True
        assert resource.data == '{"key": "value"}'

    def test_convert_binary_resource(self):
        """Test converting binary (blob) resource contents."""
        original_data = b"\x89PNG\r\n\x1a\n\x00\x00"
        contents = MockBlobResourceContents(
            data=original_data,
            mime_type="image/png",
        )
        resource = convert_mcp_resource("file:///image.png", contents)

        assert resource.uri == "file:///image.png"
        assert resource.mime_type == "image/png"
        assert resource.is_binary is True
        assert resource.data == original_data

    def test_convert_unsupported_type_raises(self):
        """Test unsupported content type raises TypeError."""
        unknown_contents = MagicMock()
        # Remove text and blob attributes
        del unknown_contents.text
        del unknown_contents.blob

        with pytest.raises(TypeError, match="Unsupported resource content type"):
            convert_mcp_resource("file:///unknown", unknown_contents)


# ---------------------------------------------------------------------------
# load_mcp_resources tests
# ---------------------------------------------------------------------------


class TestLoadMcpResources:
    """Tests for load_mcp_resources function."""

    @pytest.mark.asyncio
    async def test_load_specific_uri(self):
        """Test loading a specific resource by URI."""
        session = AsyncMock()
        session.read_resource.return_value = MockReadResourceResult(
            contents=[MockTextResourceContents(text="Hello, World!", mime_type="text/plain")]
        )

        resources = await load_mcp_resources(session, uris="file:///test.txt")

        session.read_resource.assert_called_once_with("file:///test.txt")
        assert len(resources) == 1
        assert resources[0].uri == "file:///test.txt"
        assert resources[0].data == "Hello, World!"

    @pytest.mark.asyncio
    async def test_load_multiple_uris(self):
        """Test loading multiple resources by URI list."""
        session = AsyncMock()
        session.read_resource.side_effect = [
            MockReadResourceResult(
                contents=[MockTextResourceContents(text="File 1", mime_type="text/plain")]
            ),
            MockReadResourceResult(
                contents=[MockTextResourceContents(text="File 2", mime_type="text/plain")]
            ),
        ]

        resources = await load_mcp_resources(session, uris=["file:///a.txt", "file:///b.txt"])

        assert len(resources) == 2
        assert resources[0].data == "File 1"
        assert resources[1].data == "File 2"

    @pytest.mark.asyncio
    async def test_load_all_resources(self):
        """Test loading all resources when no URIs specified."""
        session = AsyncMock()
        session.list_resources.return_value = MockListResourcesResult(
            resources=[
                MockResource(uri="file:///a.txt"),
                MockResource(uri="file:///b.txt"),
            ]
        )
        session.read_resource.side_effect = [
            MockReadResourceResult(contents=[MockTextResourceContents(text="A")]),
            MockReadResourceResult(contents=[MockTextResourceContents(text="B")]),
        ]

        resources = await load_mcp_resources(session, uris=None)

        session.list_resources.assert_called_once()
        assert len(resources) == 2


# ---------------------------------------------------------------------------
# list_mcp_resources tests
# ---------------------------------------------------------------------------


class TestListMcpResources:
    """Tests for list_mcp_resources function."""

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing available resources."""
        session = AsyncMock()
        session.list_resources.return_value = MockListResourcesResult(
            resources=[
                MockResource(
                    uri="file:///config.json",
                    name="Config",
                    description="App config",
                    mime_type="application/json",
                ),
                MockResource(
                    uri="file:///data.csv",
                    name="Data",
                    mime_type="text/csv",
                ),
            ]
        )

        resources = await list_mcp_resources(session)

        assert len(resources) == 2
        assert resources[0].uri == "file:///config.json"
        assert resources[0].name == "Config"
        assert resources[1].uri == "file:///data.csv"

    @pytest.mark.asyncio
    async def test_list_resources_empty(self):
        """Test listing resources when none available."""
        session = AsyncMock()
        session.list_resources.return_value = MockListResourcesResult(resources=[])

        resources = await list_mcp_resources(session)

        assert resources == []

    @pytest.mark.asyncio
    async def test_list_resources_handles_none(self):
        """Test listing resources handles None result."""
        session = AsyncMock()
        result = MagicMock()
        result.resources = None
        session.list_resources.return_value = result

        resources = await list_mcp_resources(session)

        assert resources == []


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestResourcesIntegration:
    """Integration-style tests for resources functionality."""

    @pytest.mark.asyncio
    async def test_full_resource_workflow(self):
        """Test complete workflow: list, select, load resources."""
        session = AsyncMock()

        # List resources
        session.list_resources.return_value = MockListResourcesResult(
            resources=[
                MockResource(
                    uri="file:///app/config.json",
                    name="App Config",
                    mime_type="application/json",
                ),
                MockResource(
                    uri="file:///app/logo.png",
                    name="Logo",
                    mime_type="image/png",
                ),
            ]
        )

        resources_info = await list_mcp_resources(session)
        assert len(resources_info) == 2

        # Load text resource
        config_json = '{"app": "test", "version": "1.0"}'
        session.read_resource.return_value = MockReadResourceResult(
            contents=[MockTextResourceContents(text=config_json, mime_type="application/json")]
        )

        loaded = await load_mcp_resources(session, uris="file:///app/config.json")
        assert len(loaded) == 1
        assert loaded[0].is_text
        assert '"app": "test"' in loaded[0].data

        # Load binary resource
        logo_bytes = b"\x89PNG\r\n\x1a\n"
        session.read_resource.return_value = MockReadResourceResult(
            contents=[MockBlobResourceContents(data=logo_bytes, mime_type="image/png")]
        )

        loaded = await load_mcp_resources(session, uris="file:///app/logo.png")
        assert len(loaded) == 1
        assert loaded[0].is_binary
        assert loaded[0].data == logo_bytes

    def test_resource_info_repr(self):
        """Test ResourceInfo has useful string representation."""
        info = ResourceInfo(uri="file:///test.txt", name="Test")
        repr_str = repr(info)

        assert "file:///test.txt" in repr_str
        assert "ResourceInfo" in repr_str

    def test_mcp_resource_repr(self):
        """Test MCPResource has useful string representation."""
        resource = MCPResource(
            uri="file:///test.txt",
            mime_type="text/plain",
            data="Hello",
        )
        repr_str = repr(resource)

        assert "file:///test.txt" in repr_str
        assert "MCPResource" in repr_str
