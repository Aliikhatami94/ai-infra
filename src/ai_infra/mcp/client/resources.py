"""Resources support for MCP client.

This module provides functionality to fetch and work with MCP resources -
files, data, and static content exposed by MCP servers.

Resources can be text (e.g., code files, configs) or binary (e.g., images, PDFs).

Example:
    ```python
    from ai_infra.mcp import MCPClient

    async with MCPClient([config]) as mcp:
        # List available resources
        resources = await mcp.list_resources("my-server")
        for uri, info_list in resources.items():
            for info in info_list:
                print(f"{info.uri}: {info.name}")

        # Get specific resources
        data = await mcp.get_resources(
            "my-server",
            uris=["file:///path/to/config.json"],
        )
        for resource in data:
            print(f"{resource.uri}: {len(resource.data)} bytes")
    ```
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Resource
    from pydantic import AnyUrl


@dataclass
class ResourceInfo:
    """Information about an available MCP resource.

    Attributes:
        uri: The resource URI (e.g., "file:///path/to/file.txt").
        name: Human-readable name for the resource.
        description: Optional description of the resource.
        mime_type: Optional MIME type (e.g., "text/plain", "image/png").
    """

    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None

    @classmethod
    def from_mcp_resource(cls, resource: Resource) -> ResourceInfo:
        """Create ResourceInfo from MCP Resource object.

        Args:
            resource: The MCP Resource object.

        Returns:
            ResourceInfo with extracted fields.
        """
        return cls(
            uri=str(resource.uri),
            name=getattr(resource, "name", None),
            description=getattr(resource, "description", None),
            mime_type=getattr(resource, "mimeType", None),
        )


@dataclass
class MCPResource:
    """A loaded MCP resource with its content.

    Resources can contain either text or binary data.
    Use `is_text` and `is_binary` properties to check the type.

    Attributes:
        uri: The resource URI.
        mime_type: Optional MIME type of the resource.
        data: The resource content (str for text, bytes for binary).

    Example:
        ```python
        resource = await mcp.get_resources("server", uris=["file:///config.json"])
        if resource[0].is_text:
            config = json.loads(resource[0].data)
        ```
    """

    uri: str
    mime_type: str | None
    data: str | bytes

    @property
    def is_text(self) -> bool:
        """Check if resource contains text data."""
        return isinstance(self.data, str)

    @property
    def is_binary(self) -> bool:
        """Check if resource contains binary data."""
        return isinstance(self.data, bytes)

    @property
    def size(self) -> int:
        """Get the size of the resource data in bytes."""
        if isinstance(self.data, str):
            return len(self.data.encode("utf-8"))
        return len(self.data)

    def as_text(self, encoding: str = "utf-8") -> str:
        """Get resource data as text.

        Args:
            encoding: Encoding to use if data is binary.

        Returns:
            The resource data as a string.

        Raises:
            UnicodeDecodeError: If binary data cannot be decoded.
        """
        if isinstance(self.data, str):
            return self.data
        return self.data.decode(encoding)

    def as_bytes(self, encoding: str = "utf-8") -> bytes:
        """Get resource data as bytes.

        Args:
            encoding: Encoding to use if data is text.

        Returns:
            The resource data as bytes.
        """
        if isinstance(self.data, bytes):
            return self.data
        return self.data.encode(encoding)


def convert_mcp_resource(uri: str, contents: Any) -> MCPResource:
    """Convert MCP resource contents to MCPResource.

    Handles both text and binary (blob) resource contents.
    Binary resources are base64-decoded.

    Args:
        uri: The resource URI.
        contents: The MCP resource contents object.

    Returns:
        MCPResource with the converted data.

    Raises:
        TypeError: If the contents type is not supported.

    Example:
        ```python
        # For text content
        text_resource = convert_mcp_resource(
            "file:///config.json",
            TextResourceContents(text='{"key": "value"}', mimeType="application/json")
        )
        assert text_resource.is_text

        # For binary content (base64 encoded)
        binary_resource = convert_mcp_resource(
            "file:///image.png",
            BlobResourceContents(blob="iVBORw0KGgo=...", mimeType="image/png")
        )
        assert binary_resource.is_binary
        ```
    """
    # Check for text content
    if hasattr(contents, "text"):
        return MCPResource(
            uri=uri,
            mime_type=getattr(contents, "mimeType", None),
            data=contents.text,
        )

    # Check for blob (binary) content
    if hasattr(contents, "blob"):
        decoded_data = base64.b64decode(contents.blob)
        return MCPResource(
            uri=uri,
            mime_type=getattr(contents, "mimeType", None),
            data=decoded_data,
        )

    raise TypeError(
        f"Unsupported resource content type: {type(contents)}. "
        "Expected TextResourceContents or BlobResourceContents."
    )


async def load_mcp_resources(
    session: ClientSession,
    *,
    uris: str | list[str] | None = None,
) -> list[MCPResource]:
    """Load resources from an MCP server.

    If no URIs are specified, loads all available resources.

    Args:
        session: The MCP ClientSession to use.
        uris: Optional URI(s) to load. Can be a single URI string,
            a list of URIs, or None to load all resources.

    Returns:
        List of MCPResource objects with loaded content.

    Example:
        ```python
        # Load all resources
        resources = await load_mcp_resources(session)

        # Load specific resource
        resources = await load_mcp_resources(
            session,
            uris="file:///config.json"
        )

        # Load multiple resources
        resources = await load_mcp_resources(
            session,
            uris=["file:///config.json", "file:///data.csv"]
        )
        ```
    """
    # Build URI list
    if uris is None:
        # List all resources and get their URIs
        result = await session.list_resources()
        resources_list = getattr(result, "resources", result) or []
        uri_list: list[str] = []
        for r in resources_list:
            uri: Any = None
            if hasattr(r, "uri"):
                uri = r.uri
            elif isinstance(r, dict):
                uri = r.get("uri")
            elif isinstance(r, (tuple, list)) and len(r) >= 1:
                uri = r[0]
            if uri is not None:
                uri_list.append(str(uri))
    elif isinstance(uris, str):
        uri_list = [uris]
    else:
        uri_list = list(uris)

    # Load each resource
    resources = []
    for uri in uri_list:
        # read_resource expects AnyUrl but we have str - cast for type checker
        read_result = await session.read_resource(cast("AnyUrl", uri))
        contents_list = getattr(read_result, "contents", []) or []
        for content in contents_list:
            resources.append(convert_mcp_resource(uri, content))

    return resources


async def list_mcp_resources(session: ClientSession) -> list[ResourceInfo]:
    """List available resources from an MCP server.

    Args:
        session: The MCP ClientSession to use.

    Returns:
        List of ResourceInfo objects describing available resources.

    Example:
        ```python
        async with client_session as session:
            resources = await list_mcp_resources(session)
            for r in resources:
                print(f"{r.uri}: {r.name} ({r.mime_type})")
        ```
    """
    result = await session.list_resources()
    resources_list = getattr(result, "resources", result) or []
    # Each r is expected to be a Resource object
    return [ResourceInfo.from_mcp_resource(cast("Resource", r)) for r in resources_list]
