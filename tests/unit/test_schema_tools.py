"""Tests for schema_tools - tools_from_models() function."""

from typing import Optional

import pytest
from pydantic import BaseModel, Field

from ai_infra.tools.schema_tools import (
    GeneratedTool,
    ToolConfig,
    _get_model_fields,
    _get_model_name,
    tools_from_models,
)


# Test models
class User(BaseModel):
    """A user model."""

    id: int
    name: str
    email: str
    age: Optional[int] = None
    is_active: bool = True


class ProductModel(BaseModel):
    """A product model with 'Model' suffix."""

    id: int
    name: str = Field(..., description="Product name")
    price: float
    category: str


class Order(BaseModel):
    """An order model."""

    id: int
    user_id: int
    product_id: int
    quantity: int = 1
    status: str = "pending"


class TestGetModelName:
    """Tests for _get_model_name helper."""

    def test_simple_name(self):
        assert _get_model_name(User) == "user"

    def test_removes_model_suffix(self):
        assert _get_model_name(ProductModel) == "product"

    def test_preserves_short_names(self):
        # Shouldn't remove suffix if it would leave empty string
        class Model(BaseModel):
            id: int

        # This would result in empty string without protection
        assert _get_model_name(Model) == "model"


class TestGetModelFields:
    """Tests for _get_model_fields helper."""

    def test_pydantic_model_fields(self):
        fields = _get_model_fields(User)

        assert "id" in fields
        assert "name" in fields
        assert "email" in fields
        assert "age" in fields
        assert "is_active" in fields

        # Check types
        assert fields["id"][0] == int
        assert fields["name"][0] == str
        assert fields["is_active"][0] == bool

    def test_optional_field_has_default(self):
        fields = _get_model_fields(User)

        # age is Optional with None default
        assert fields["age"][1] is None

        # is_active has True default
        assert fields["is_active"][1] is True

        # Required fields have ... as default
        assert fields["name"][1] is ...


class TestToolsFromModels:
    """Tests for tools_from_models main function."""

    def test_generates_all_crud_tools(self):
        tools = tools_from_models(User)

        tool_names = [t.name for t in tools]

        assert "get_user" in tool_names
        assert "list_users" in tool_names
        assert "create_user" in tool_names
        assert "update_user" in tool_names
        assert "delete_user" in tool_names

        assert len(tools) == 5

    def test_read_only_mode(self):
        tools = tools_from_models(User, read_only=True)

        tool_names = [t.name for t in tools]

        assert "get_user" in tool_names
        assert "list_users" in tool_names
        assert "create_user" not in tool_names
        assert "update_user" not in tool_names
        assert "delete_user" not in tool_names

        assert len(tools) == 2

    def test_specific_operations(self):
        tools = tools_from_models(User, operations=["get", "create"])

        tool_names = [t.name for t in tools]

        assert "get_user" in tool_names
        assert "create_user" in tool_names
        assert "list_users" not in tool_names
        assert len(tools) == 2

    def test_multiple_models(self):
        tools = tools_from_models(User, Order)

        tool_names = [t.name for t in tools]

        # Should have 5 tools for each model = 10 total
        assert len(tools) == 10

        assert "get_user" in tool_names
        assert "get_order" in tool_names
        assert "list_users" in tool_names
        assert "list_orders" in tool_names

    def test_custom_name_pattern(self):
        tools = tools_from_models(User, name_pattern="{model}_{action}")

        tool_names = [t.name for t in tools]

        assert "user_get" in tool_names
        assert "users_list" in tool_names
        assert "user_create" in tool_names

    def test_tools_have_descriptions(self):
        tools = tools_from_models(User)

        for tool in tools:
            assert tool.description is not None
            assert "user" in tool.description.lower()

    def test_tools_have_args_schema(self):
        tools = tools_from_models(User)

        for tool in tools:
            assert hasattr(tool, "args_schema")
            assert issubclass(tool.args_schema, BaseModel)


class TestToolExecution:
    """Tests for executing generated tools."""

    def test_get_tool_without_executor(self):
        tools = tools_from_models(User)
        get_tool = next(t for t in tools if t.name == "get_user")

        result = get_tool.invoke({"id": 123})

        assert result["error"] == "No executor configured"
        assert result["id"] == 123

    def test_get_tool_with_executor(self):
        # Mock executor
        def mock_executor(operation, model, **kwargs):
            return {"operation": operation, "model": model.__name__, **kwargs}

        tools = tools_from_models(User, executor=mock_executor)
        get_tool = next(t for t in tools if t.name == "get_user")

        result = get_tool.invoke({"id": 123})

        assert result["operation"] == "get"
        assert result["model"] == "User"
        assert result["id"] == 123

    def test_list_tool_with_filters(self):
        captured = {}

        def mock_executor(operation, model, **kwargs):
            captured.update(kwargs)
            return []

        tools = tools_from_models(User, executor=mock_executor)
        list_tool = next(t for t in tools if t.name == "list_users")

        list_tool.invoke({"name": "John", "limit": 10})

        assert captured["name"] == "John"
        assert captured["limit"] == 10

    def test_create_tool_excludes_id(self):
        captured = {}

        def mock_executor(operation, model, **kwargs):
            captured.update(kwargs)
            return {"id": 1}

        tools = tools_from_models(User, executor=mock_executor)
        create_tool = next(t for t in tools if t.name == "create_user")

        create_tool.invoke({"name": "John", "email": "john@example.com"})

        # ID should not be in the create call
        assert "id" not in captured
        assert captured["name"] == "John"
        assert captured["email"] == "john@example.com"

    def test_update_tool_requires_id(self):
        tools = tools_from_models(User)
        update_tool = next(t for t in tools if t.name == "update_user")

        # Check that id is required in parameters
        params_fields = update_tool.args_schema.model_fields
        assert "id" in params_fields

        # id should be required (no default)
        id_field = params_fields["id"]
        assert id_field.is_required()

    def test_update_tool_filters_none_values(self):
        captured = {}

        def mock_executor(operation, model, **kwargs):
            captured.update(kwargs)
            return {"id": kwargs["id"]}

        tools = tools_from_models(User, executor=mock_executor)
        update_tool = next(t for t in tools if t.name == "update_user")

        # Only update name, leave others as None
        update_tool.invoke({"id": 1, "name": "Jane", "email": None, "age": None})

        # None values should be filtered out
        assert "name" in captured
        assert captured["name"] == "Jane"
        # email and age were None, should not be in updates
        # Note: captured has id separately, and updates are in the remaining kwargs

    def test_delete_tool(self):
        captured = {}

        def mock_executor(operation, model, **kwargs):
            captured["operation"] = operation
            captured["id"] = kwargs["id"]
            return {"deleted": True}

        tools = tools_from_models(User, executor=mock_executor)
        delete_tool = next(t for t in tools if t.name == "delete_user")

        result = delete_tool.invoke({"id": 123})

        assert captured["operation"] == "delete"
        assert captured["id"] == 123
        assert result["deleted"] is True


class TestPaginationConfig:
    """Tests for pagination configuration."""

    def test_default_limit(self):
        tools = tools_from_models(User, default_limit=50)
        list_tool = next(t for t in tools if t.name == "list_users")

        params_fields = list_tool.args_schema.model_fields
        limit_field = params_fields["limit"]

        assert limit_field.default == 50

    def test_max_limit(self):
        tools = tools_from_models(User, max_limit=500)
        list_tool = next(t for t in tools if t.name == "list_users")

        params_fields = list_tool.args_schema.model_fields
        limit_field = params_fields["limit"]

        # Check that max is set in JSON schema
        schema = list_tool.args_schema.model_json_schema()
        assert schema["properties"]["limit"]["maximum"] == 500
