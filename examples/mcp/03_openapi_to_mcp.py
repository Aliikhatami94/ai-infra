#!/usr/bin/env python
"""OpenAPI to MCP Conversion Example.

This example demonstrates:
- Converting any OpenAPI spec into MCP tools automatically
- Filtering endpoints (paths, methods, tags, operations)
- Authentication configuration (API keys, Bearer tokens)
- Custom tool naming and descriptions
- Pagination handling

This is a flagship feature of ai-infra! Any REST API with an OpenAPI
spec can instantly become available to AI agents via MCP.

OpenAPI sources:
- URL: https://api.example.com/openapi.json
- Local file: ./openapi.yaml
- Python dict: {"openapi": "3.0.0", ...}
- FastAPI app: app.openapi()
"""

import os

from ai_infra import MCPServer

# =============================================================================
# Example 1: Zero-Config OpenAPI Conversion
# =============================================================================


def simple_openapi_to_mcp():
    """Convert an OpenAPI spec to MCP with zero configuration."""
    print("=" * 60)
    print("1. Zero-Config OpenAPI to MCP")
    print("=" * 60)

    server = MCPServer()

    # Just point to an OpenAPI spec - that's it!
    # All endpoints become MCP tools automatically
    server.add_openapi(
        path="/petstore",
        spec="https://petstore3.swagger.io/api/v3/openapi.json",
    )

    print("\n✓ Created MCP server from Petstore OpenAPI spec")
    print("  All endpoints are now available as MCP tools!")
    print()
    print("  Example tools generated:")
    print("    - getPetById(petId) → GET /pet/{petId}")
    print("    - addPet(body) → POST /pet")
    print("    - findPetsByStatus(status) → GET /pet/findByStatus")
    print("    - updatePet(body) → PUT /pet")
    print("    - deletePet(petId) → DELETE /pet/{petId}")

    return server


# =============================================================================
# Example 2: Filtered Endpoints
# =============================================================================


def filtered_openapi_to_mcp():
    """Convert only specific endpoints from an OpenAPI spec."""
    print("\n" + "=" * 60)
    print("2. Filtered Endpoints")
    print("=" * 60)

    server = MCPServer()

    server.add_openapi(
        path="/github",
        spec="https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
        # Only include specific paths
        include_paths=["/repos/*", "/users/*", "/search/*"],
        # Exclude dangerous operations
        exclude_paths=["/repos/*/delete", "/admin/*"],
        # Only allow safe methods
        include_methods=["GET", "POST"],
        exclude_methods=["DELETE", "PATCH"],
        # Filter by tags
        include_tags=["repos", "users"],
        # Or by operationId
        include_operations=["repos/get", "repos/list-for-org"],
        # Prefix all tool names
        tool_prefix="github",  # Tools become: github_repos_get, etc.
    )

    print("\n✓ Created filtered MCP server from GitHub API")
    print("  Filters applied:")
    print("    - Paths: /repos/*, /users/*, /search/*")
    print("    - Methods: GET, POST only")
    print("    - Tags: repos, users")
    print("    - Prefix: github_")

    return server


# =============================================================================
# Example 3: Authentication
# =============================================================================


def authenticated_openapi_to_mcp():
    """Configure authentication for OpenAPI calls."""
    print("\n" + "=" * 60)
    print("3. Authentication Configuration")
    print("=" * 60)

    server = MCPServer()

    # Get API key from environment
    api_key = os.getenv("GITHUB_TOKEN", "your-token-here")

    # Method 1: Bearer token (most common)
    server.add_openapi(
        path="/github",
        spec="https://api.github.com/openapi",
        auth={"Authorization": f"Bearer {api_key}"},  # Header auth
    )

    print("\n✓ Authentication methods supported:")
    print()
    print("  # Header-based (Bearer token, API key)")
    print('  auth={"Authorization": "Bearer token123"}')
    print('  auth={"X-API-Key": "key123"}')
    print()
    print("  # Basic auth")
    print('  auth=("username", "password")')
    print()
    print("  # Query parameter")
    print('  auth={"api_key": "key123"}, auth_location="query"')
    print()
    print("  # Dynamic auth (refresh tokens)")
    print("  async def get_token(): return await refresh_token()")
    print("  auth=get_token")

    # Method 2: Per-endpoint auth overrides
    server.add_openapi(
        path="/mixed-auth",
        spec="./openapi.json",
        auth={"Authorization": "Bearer default-token"},
        endpoint_auth={
            "/admin/*": {"Authorization": "Bearer admin-token"},
            "/public/*": None,  # No auth needed
        },
    )

    print()
    print("  Per-endpoint auth:")
    print("    /admin/*  → Uses admin token")
    print("    /public/* → No authentication")
    print("    others    → Default token")

    return server


# =============================================================================
# Example 4: Custom Tool Naming
# =============================================================================


def custom_naming_openapi():
    """Customize how OpenAPI operations become tool names."""
    print("\n" + "=" * 60)
    print("4. Custom Tool Naming")
    print("=" * 60)

    server = MCPServer()

    # Custom naming function
    def custom_tool_name(method: str, path: str, operation: dict) -> str:
        """Generate custom tool name from OpenAPI operation.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (/users/{id})
            operation: Full OpenAPI operation object

        Returns:
            Custom tool name
        """
        # Use operationId if available
        if "operationId" in operation:
            return operation["operationId"].replace("-", "_")

        # Otherwise, build from method + path
        path_parts = path.strip("/").replace("{", "").replace("}", "").split("/")
        return f"{method.lower()}_{'_'.join(path_parts)}"

    # Custom description function
    def custom_description(operation: dict) -> str:
        """Generate custom tool description.

        Args:
            operation: Full OpenAPI operation object

        Returns:
            Custom description
        """
        summary = operation.get("summary", "")
        description = operation.get("description", "")[:100]
        return f"{summary}. {description}" if description else summary

    server.add_openapi(
        path="/custom",
        spec="./openapi.json",
        tool_name_fn=custom_tool_name,
        tool_description_fn=custom_description,
    )

    print("\n✓ Custom naming functions applied")
    print()
    print("  tool_name_fn(method, path, operation) → str")
    print("  tool_description_fn(operation) → str")
    print()
    print("  Examples:")
    print("    GET /users/{id}  → get_users_id")
    print("    POST /orders     → create_order (from operationId)")

    return server


# =============================================================================
# Example 5: From Local File or Dict
# =============================================================================


def local_openapi_sources():
    """Load OpenAPI from different sources."""
    print("\n" + "=" * 60)
    print("5. Different OpenAPI Sources")
    print("=" * 60)

    server = MCPServer()

    print("\n✓ Supported OpenAPI sources:")
    print()
    print("  # From URL")
    print('  spec="https://api.example.com/openapi.json"')
    print()
    print("  # From local JSON file")
    print('  spec="./openapi.json"')
    print()
    print("  # From local YAML file")
    print('  spec="./openapi.yaml"')
    print()
    print("  # From Path object")
    print("  from pathlib import Path")
    print('  spec=Path("./api/openapi.json")')
    print()
    print("  # From dict")
    print("  spec={")
    print('      "openapi": "3.0.0",')
    print('      "info": {"title": "My API", "version": "1.0.0"},')
    print('      "paths": {...}')
    print("  }")
    print()
    print("  # From FastAPI app")
    print("  from fastapi import FastAPI")
    print("  app = FastAPI()")
    print("  spec=app.openapi()")

    # Example with dict
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "Demo API", "version": "1.0.0"},
        "paths": {
            "/hello": {
                "get": {
                    "operationId": "sayHello",
                    "summary": "Say hello",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }

    server.add_openapi(
        path="/demo",
        spec=spec_dict,
        base_url="http://localhost:3000",  # Required for dict specs
    )

    print()
    print("  ✓ Created server from dict spec with base_url")

    return server


# =============================================================================
# Example 6: Running the OpenAPI MCP Server
# =============================================================================


def run_openapi_server(port: int = 8000):
    """Run an OpenAPI-based MCP server."""
    print("\n" + "=" * 60)
    print("6. Running the Server")
    print("=" * 60)

    server = MCPServer()

    # Add Petstore API
    server.add_openapi(
        path="/petstore",
        spec="https://petstore3.swagger.io/api/v3/openapi.json",
        tool_prefix="pet",
        include_methods=["GET", "POST"],  # Safe operations only
    )

    print(f"\nStarting server at http://127.0.0.1:{port}")
    print()
    print("Mount points:")
    print("  /petstore → Petstore API tools")
    print()
    print("Connect with MCPClient:")
    print("  mcp = MCPClient([{")
    print('      "transport": "streamable_http",')
    print(f'      "url": "http://127.0.0.1:{port}/petstore/mcp"')
    print("  }])")
    print()
    print("(Press Ctrl+C to stop)")

    # Build and run the combined app
    app = server.build()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port)


# =============================================================================
# Example 7: Multiple OpenAPI Specs
# =============================================================================


def multiple_openapi_specs():
    """Combine multiple OpenAPI specs into one MCP server."""
    print("\n" + "=" * 60)
    print("7. Multiple OpenAPI Specs")
    print("=" * 60)

    server = MCPServer()

    # Add multiple APIs under different paths
    server.add_openapi(
        path="/petstore",
        spec="https://petstore3.swagger.io/api/v3/openapi.json",
        tool_prefix="pet",
    )

    # Could add more APIs:
    # server.add_openapi(
    #     path="/github",
    #     spec="https://api.github.com/openapi",
    #     tool_prefix="github",
    #     auth={"Authorization": "Bearer token"},
    # )
    #
    # server.add_openapi(
    #     path="/stripe",
    #     spec="https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
    #     tool_prefix="stripe",
    #     auth={"Authorization": "Bearer sk-test-xxx"},
    # )

    print("\n✓ Combined MCP server with multiple APIs:")
    print("  /petstore → Petstore tools (pet_*)")
    print("  /github   → GitHub tools (github_*)")
    print("  /stripe   → Stripe tools (stripe_*)")
    print()
    print("  Each API mounted at its own path,")
    print("  all accessible via a single MCPClient!")

    return server


# =============================================================================
# Main
# =============================================================================


def main():
    """Run examples or start server."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--run":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            run_openapi_server(port)
        elif sys.argv[1] == "--help":
            print("OpenAPI to MCP Example")
            print()
            print("Usage:")
            print("  python 03_openapi_to_mcp.py          # Show examples")
            print("  python 03_openapi_to_mcp.py --run    # Run server")
            print("  python 03_openapi_to_mcp.py --run 8080  # Custom port")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
    else:
        # Show all examples
        print("\n" + "=" * 60)
        print("OpenAPI to MCP Conversion Examples")
        print("=" * 60)
        print("\nThis example shows how to convert OpenAPI specs to MCP tools.")
        print("Run with --run to start a real server.\n")

        simple_openapi_to_mcp()
        filtered_openapi_to_mcp()
        authenticated_openapi_to_mcp()
        custom_naming_openapi()
        local_openapi_sources()
        multiple_openapi_specs()

        print("\n" + "=" * 60)
        print("To start a server:")
        print("  python 03_openapi_to_mcp.py --run")
        print("=" * 60)


if __name__ == "__main__":
    main()
