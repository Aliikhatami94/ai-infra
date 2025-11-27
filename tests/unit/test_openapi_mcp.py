"""Tests for OpenAPI→MCP conversion.

Tests 3.5.1-3.5.4:
- 3.5.1: Zero-Config Conversion
- 3.5.2: Full Flexibility (filters, naming)
- 3.5.3: Authentication
- 3.5.4: Schema Handling
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_infra.mcp.server.openapi import (
    AuthConfig,
    BuildReport,
    OpenAPIOptions,
    _mcp_from_openapi,
    load_openapi,
)
from ai_infra.mcp.server.openapi.builder import (
    _merge_allof_schemas,
    _py_type_from_schema,
    _resolve_ref,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_spec() -> Dict[str, Any]:
    """Simple OpenAPI spec for testing."""
    return {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List all users",
                    "tags": ["users", "public"],
                    "responses": {"200": {"description": "OK"}},
                },
                "post": {
                    "operationId": "createUser",
                    "summary": "Create a user",
                    "tags": ["users", "admin"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"},
                                    },
                                    "required": ["name", "email"],
                                }
                            }
                        },
                    },
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/users/{id}": {
                "get": {
                    "operationId": "getUser",
                    "summary": "Get user by ID",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
                "delete": {
                    "operationId": "deleteUser",
                    "summary": "Delete user",
                    "tags": ["users", "admin"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                    "responses": {"204": {"description": "Deleted"}},
                },
            },
            "/admin/settings": {
                "get": {
                    "operationId": "getSettings",
                    "summary": "Get admin settings",
                    "tags": ["admin"],
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/internal/health": {
                "get": {
                    "operationId": "healthCheck",
                    "summary": "Health check",
                    "tags": ["internal", "deprecated"],
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    }


@pytest.fixture
def complex_schema_spec() -> Dict[str, Any]:
    """Spec with complex schema compositions."""
    return {
        "openapi": "3.1.0",
        "info": {"title": "Complex Schema API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "components": {
            "schemas": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"$ref": "#/components/schemas/Address"},
                    },
                },
                "Employee": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Person"},
                        {
                            "type": "object",
                            "properties": {
                                "employeeId": {"type": "integer"},
                                "department": {"type": "string"},
                            },
                        },
                    ],
                },
                "Status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                },
                "Result": {
                    "oneOf": [
                        {"type": "object", "properties": {"success": {"type": "boolean"}}},
                        {"type": "object", "properties": {"error": {"type": "string"}}},
                    ],
                },
                # Circular reference
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/TreeNode"},
                        },
                    },
                },
            },
        },
        "paths": {
            "/employees": {
                "post": {
                    "operationId": "createEmployee",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Employee"}
                            }
                        },
                    },
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/status": {
                "get": {
                    "operationId": "getStatus",
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Status"}
                                }
                            },
                        }
                    },
                },
            },
        },
    }


# =============================================================================
# 3.5.1: Zero-Config Conversion Tests
# =============================================================================


class TestZeroConfig:
    """Test 3.5.1: Zero-Config OpenAPI→MCP conversion."""

    def test_dict_spec(self, simple_spec):
        """Spec as dict works directly."""
        mcp, cleanup, report = _mcp_from_openapi(simple_spec)

        assert report.title == "Test API"
        assert report.registered_tools > 0
        assert report.total_ops == 6  # 6 operations in the spec

    def test_load_openapi_dict_passthrough(self):
        """load_openapi returns dict as-is."""
        spec = {"openapi": "3.1.0", "info": {"title": "Test"}}
        result = load_openapi(spec)
        assert result == spec

    def test_load_openapi_json_string(self):
        """load_openapi parses JSON string."""
        json_str = '{"openapi": "3.1.0", "info": {"title": "Test"}}'
        result = load_openapi(json_str)
        assert result["openapi"] == "3.1.0"
        assert result["info"]["title"] == "Test"

    def test_load_openapi_yaml_string(self):
        """load_openapi parses YAML string."""
        yaml_str = "openapi: '3.1.0'\ninfo:\n  title: Test"
        result = load_openapi(yaml_str)
        assert result["openapi"] == "3.1.0"
        assert result["info"]["title"] == "Test"

    def test_all_endpoints_become_tools(self, simple_spec):
        """All endpoints become MCP tools by default."""
        mcp, cleanup, report = _mcp_from_openapi(simple_spec)

        # Should have tools for all operations
        tool_names = [op.tool_name for op in report.ops]
        assert "listUsers" in tool_names
        assert "createUser" in tool_names
        assert "getUser" in tool_names
        assert "deleteUser" in tool_names
        assert "getSettings" in tool_names
        assert "healthCheck" in tool_names


# =============================================================================
# 3.5.2: Full Flexibility Tests
# =============================================================================


class TestFiltering:
    """Test 3.5.2: Filtering and customization options."""

    def test_tool_prefix(self, simple_spec):
        """tool_prefix prepends to all tool names."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            tool_prefix="github",
        )

        tool_names = [op.tool_name for op in report.ops]
        for name in tool_names:
            assert name.startswith("github_"), f"Expected prefix 'github_' in {name}"

    def test_include_paths(self, simple_spec):
        """include_paths filters to matching paths."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            include_paths=["/users", "/users/*"],
        )

        # Should only have /users endpoints
        tool_names = [op.tool_name for op in report.ops]
        assert "listUsers" in tool_names
        assert "createUser" in tool_names
        assert "getUser" in tool_names
        assert "deleteUser" in tool_names

        # Should NOT have /admin or /internal
        assert "getSettings" not in tool_names
        assert "healthCheck" not in tool_names

        # Check filtered count
        assert report.filtered_ops == 2  # admin + internal

    def test_exclude_paths(self, simple_spec):
        """exclude_paths filters out matching paths."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            exclude_paths=["/admin/*", "/internal/*"],
        )

        tool_names = [op.tool_name for op in report.ops]
        assert "getSettings" not in tool_names
        assert "healthCheck" not in tool_names

        # Users should still be there
        assert "listUsers" in tool_names

    def test_include_methods(self, simple_spec):
        """include_methods filters to specific HTTP methods."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            include_methods=["GET"],
        )

        tool_names = [op.tool_name for op in report.ops]

        # GET methods only
        assert "listUsers" in tool_names
        assert "getUser" in tool_names
        assert "getSettings" in tool_names
        assert "healthCheck" in tool_names

        # No POST or DELETE
        assert "createUser" not in tool_names
        assert "deleteUser" not in tool_names

    def test_exclude_methods(self, simple_spec):
        """exclude_methods filters out specific HTTP methods."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            exclude_methods=["DELETE"],
        )

        tool_names = [op.tool_name for op in report.ops]
        assert "deleteUser" not in tool_names
        assert "listUsers" in tool_names  # Other methods still there

    def test_include_tags(self, simple_spec):
        """include_tags filters to operations with specific tags."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            include_tags=["public"],
        )

        tool_names = [op.tool_name for op in report.ops]
        # Only listUsers has "public" tag
        assert "listUsers" in tool_names
        assert len(tool_names) == 1

    def test_exclude_tags(self, simple_spec):
        """exclude_tags filters out operations with specific tags."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            exclude_tags=["deprecated", "admin"],
        )

        tool_names = [op.tool_name for op in report.ops]

        # healthCheck has "deprecated", should be excluded
        assert "healthCheck" not in tool_names

        # createUser and deleteUser have "admin", should be excluded
        assert "createUser" not in tool_names
        assert "deleteUser" not in tool_names

        # getSettings has only "admin", should be excluded
        assert "getSettings" not in tool_names

        # listUsers and getUser should remain
        assert "listUsers" in tool_names
        assert "getUser" in tool_names

    def test_include_operations(self, simple_spec):
        """include_operations filters to specific operationIds."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            include_operations=["listUsers", "getUser"],
        )

        tool_names = [op.tool_name for op in report.ops]
        assert tool_names == ["listUsers", "getUser"]

    def test_exclude_operations(self, simple_spec):
        """exclude_operations filters out specific operationIds."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            exclude_operations=["deleteUser", "healthCheck"],
        )

        tool_names = [op.tool_name for op in report.ops]
        assert "deleteUser" not in tool_names
        assert "healthCheck" not in tool_names
        assert "listUsers" in tool_names

    def test_custom_tool_name_fn(self, simple_spec):
        """tool_name_fn customizes tool names."""

        def custom_name(method: str, path: str, operation: dict) -> str:
            return f"{method.lower()}_{path.replace('/', '_').strip('_')}"

        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            tool_name_fn=custom_name,
            include_paths=["/users"],  # Just test one path
        )

        tool_names = [op.tool_name for op in report.ops]
        assert "get_users" in tool_names
        assert "post_users" in tool_names

    def test_custom_description_fn(self, simple_spec):
        """tool_description_fn customizes descriptions."""

        def custom_desc(operation: dict) -> str:
            return f"API: {operation.get('summary', 'No desc')}"

        options = OpenAPIOptions(tool_description_fn=custom_desc)
        mcp, cleanup, report = _mcp_from_openapi(simple_spec, options=options)

        # Descriptions should be customized (checked during build)
        assert report.registered_tools > 0


# =============================================================================
# 3.5.3: Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test 3.5.3: Authentication configuration."""

    def test_auth_config_from_dict(self):
        """AuthConfig.from_value handles dict as headers."""
        auth = AuthConfig.from_value({"Authorization": "Bearer xxx"})
        assert auth.headers == {"Authorization": "Bearer xxx"}

    def test_auth_config_from_tuple(self):
        """AuthConfig.from_value handles tuple as basic auth."""
        auth = AuthConfig.from_value(("user", "pass"))
        assert auth.basic == ("user", "pass")

    def test_auth_config_from_string(self):
        """AuthConfig.from_value handles string as bearer token."""
        auth = AuthConfig.from_value("my-token")
        assert auth.bearer == "my-token"

    def test_auth_config_from_callable(self):
        """AuthConfig.from_value handles callable as bearer_fn."""

        def get_token():
            return "dynamic-token"

        auth = AuthConfig.from_value(get_token)
        assert auth.bearer_fn is get_token

    def test_auth_in_options(self, simple_spec):
        """auth parameter is passed through options."""
        mcp, cleanup, report = _mcp_from_openapi(
            simple_spec,
            auth={"X-API-Key": "secret"},
        )

        # Build should succeed
        assert report.registered_tools > 0

    def test_endpoint_auth(self, simple_spec):
        """endpoint_auth allows per-path auth."""
        options = OpenAPIOptions(
            auth=AuthConfig(headers={"Authorization": "Bearer default"}),
            endpoint_auth={
                "/admin/*": {"Authorization": "Bearer admin-token"},
                "/internal/*": None,  # No auth
            },
        )

        # Test the matching logic
        assert options.get_auth_for_path("/users").headers == {"Authorization": "Bearer default"}

        admin_auth = options.get_auth_for_path("/admin/settings")
        assert admin_auth.headers == {"Authorization": "Bearer admin-token"}

        assert options.get_auth_for_path("/internal/health") is None


# =============================================================================
# 3.5.4: Schema Handling Tests
# =============================================================================


class TestSchemaHandling:
    """Test 3.5.4: Schema composition and handling."""

    def test_resolve_ref_simple(self, complex_schema_spec):
        """$ref resolution works."""
        schema = {"$ref": "#/components/schemas/Address"}
        resolved = _resolve_ref(schema, complex_schema_spec)

        assert resolved["type"] == "object"
        assert "street" in resolved["properties"]
        assert "city" in resolved["properties"]

    def test_resolve_ref_circular(self, complex_schema_spec):
        """Circular refs don't cause infinite loop."""
        schema = {"$ref": "#/components/schemas/TreeNode"}
        resolved = _resolve_ref(schema, complex_schema_spec)

        # Should resolve without hanging
        assert resolved["type"] == "object"
        assert "value" in resolved["properties"]

    def test_allof_merge(self, complex_schema_spec):
        """allOf schemas are merged correctly."""
        employee_schema = complex_schema_spec["components"]["schemas"]["Employee"]
        merged = _merge_allof_schemas(employee_schema["allOf"], complex_schema_spec)

        # Should have properties from both schemas
        assert "name" in merged["properties"]  # From Person
        assert "address" in merged["properties"]  # From Person
        assert "employeeId" in merged["properties"]  # From Employee-specific
        assert "department" in merged["properties"]  # From Employee-specific

    def test_py_type_string(self):
        """String schema becomes str."""
        schema = {"type": "string"}
        assert _py_type_from_schema(schema) is str

    def test_py_type_integer(self):
        """Integer schema becomes int."""
        schema = {"type": "integer"}
        assert _py_type_from_schema(schema) is int

    def test_py_type_number(self):
        """Number schema becomes float."""
        schema = {"type": "number"}
        assert _py_type_from_schema(schema) is float

    def test_py_type_boolean(self):
        """Boolean schema becomes bool."""
        schema = {"type": "boolean"}
        assert _py_type_from_schema(schema) is bool

    def test_py_type_array(self):
        """Array schema becomes List[T]."""
        schema = {"type": "array", "items": {"type": "string"}}
        result = _py_type_from_schema(schema)

        # Check it's a List type
        assert hasattr(result, "__origin__")
        assert result.__origin__ is list

    def test_py_type_object(self):
        """Object schema becomes Pydantic model."""
        from pydantic import BaseModel

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = _py_type_from_schema(schema)

        # Should be a Pydantic model class
        assert issubclass(result, BaseModel)
        assert "name" in result.model_fields
        assert "age" in result.model_fields

    def test_py_type_enum(self):
        """Enum schema becomes Literal type."""
        from typing import get_args

        schema = {"type": "string", "enum": ["active", "inactive", "pending"]}
        result = _py_type_from_schema(schema)

        # Should be a Literal type with the enum values
        args = get_args(result)
        assert "active" in args
        assert "inactive" in args
        assert "pending" in args

    def test_py_type_oneof(self):
        """oneOf schema becomes Union type."""
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        }
        result = _py_type_from_schema(schema)

        # Should be Union[str, int]
        assert hasattr(result, "__origin__")
        from typing import Union, get_origin

        assert get_origin(result) is Union

    def test_py_type_with_spec_ref(self, complex_schema_spec):
        """Schema with $ref resolves through spec."""
        from pydantic import BaseModel

        schema = {"$ref": "#/components/schemas/Address"}
        result = _py_type_from_schema(schema, complex_schema_spec)

        # Should resolve to the Address model
        assert issubclass(result, BaseModel)
        assert "street" in result.model_fields

    def test_nested_object_with_ref(self, complex_schema_spec):
        """Nested objects with refs work."""
        from pydantic import BaseModel

        schema = {"$ref": "#/components/schemas/Person"}
        result = _py_type_from_schema(schema, complex_schema_spec)

        assert issubclass(result, BaseModel)
        assert "name" in result.model_fields
        assert "address" in result.model_fields

    def test_binary_format(self):
        """Binary format becomes bytes."""
        schema = {"type": "string", "format": "binary"}
        assert _py_type_from_schema(schema) is bytes

    def test_build_with_complex_schemas(self, complex_schema_spec):
        """Full build works with complex schemas."""
        mcp, cleanup, report = _mcp_from_openapi(complex_schema_spec)

        assert report.registered_tools == 2  # createEmployee, getStatus
        assert len(report.warnings) == 0 or all("base URL" not in w for w in report.warnings[:5])


# =============================================================================
# OpenAPIOptions Tests
# =============================================================================


class TestOpenAPIOptions:
    """Test OpenAPIOptions dataclass."""

    def test_should_include_default(self):
        """Default options include everything."""
        options = OpenAPIOptions()

        assert options.should_include_operation("/users", "GET", {})
        assert options.should_include_operation("/admin", "DELETE", {})

    def test_should_include_method_filter(self):
        """Method filtering works."""
        options = OpenAPIOptions(include_methods=["GET", "POST"])

        assert options.should_include_operation("/users", "GET", {})
        assert options.should_include_operation("/users", "POST", {})
        assert not options.should_include_operation("/users", "DELETE", {})

    def test_should_include_path_filter(self):
        """Path glob filtering works."""
        options = OpenAPIOptions(include_paths=["/users/*"])

        assert options.should_include_operation("/users/123", "GET", {})
        assert not options.should_include_operation("/admin/settings", "GET", {})

    def test_should_include_tag_filter(self):
        """Tag filtering works."""
        options = OpenAPIOptions(exclude_tags=["deprecated"])

        assert options.should_include_operation("/users", "GET", {"tags": ["users"]})
        assert not options.should_include_operation("/old", "GET", {"tags": ["deprecated"]})

    def test_get_tool_name_with_prefix(self):
        """Prefix is applied to tool name."""
        options = OpenAPIOptions(tool_prefix="github")

        name = options.get_tool_name("getUser", "GET", "/users", {})
        assert name == "github_getUser"

    def test_get_tool_name_with_custom_fn(self):
        """Custom function overrides default name."""
        options = OpenAPIOptions(tool_name_fn=lambda m, p, o: f"custom_{m.lower()}")

        name = options.get_tool_name("getUser", "GET", "/users", {})
        assert name == "custom_get"

    def test_get_tool_name_prefix_with_custom_fn(self):
        """Prefix is applied after custom function."""
        options = OpenAPIOptions(tool_prefix="api", tool_name_fn=lambda m, p, o: f"{m.lower()}_op")

        name = options.get_tool_name("getUser", "GET", "/users", {})
        assert name == "api_get_op"
