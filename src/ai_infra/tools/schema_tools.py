"""
Schema-to-Tools: Automatically generate CRUD tools from SQLAlchemy/Pydantic models.

This module provides `tools_from_models()` which generates get, list, create, update,
and delete tools from database models.

Example:
    ```python
    from sqlalchemy.orm import DeclarativeBase
    from ai_infra import Agent, tools_from_models

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str]
        email: Mapped[str]

    # Generate CRUD tools
    tools = tools_from_models(User)
    # Creates: get_user, list_users, create_user, update_user, delete_user

    agent = Agent(tools=tools)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeVar, get_type_hints

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

# Type for model classes
ModelT = TypeVar("ModelT")


@dataclass
class ToolConfig:
    """Configuration for generated tools."""

    # Tool naming pattern: {prefix}_{model_name} or {model_name}_{suffix}
    name_pattern: str = "{action}_{model}"

    # Include these operations (None = all)
    operations: set[str] | None = None

    # Default page size for list operations
    default_limit: int = 20

    # Max page size for list operations
    max_limit: int = 100


@dataclass
class GeneratedTool:
    """A tool generated from a model schema."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: type[BaseModel]
    operation: str  # get, list, create, update, delete

    def to_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain StructuredTool with proper schema."""
        return StructuredTool.from_function(
            func=self.func,
            name=self.name,
            description=self.description,
            args_schema=self.parameters,
        )


def _get_model_name(model: type) -> str:
    """Extract model name, handling SQLAlchemy and Pydantic models."""
    name = model.__name__

    # Remove common suffixes
    for suffix in ("Model", "Schema", "Table"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]

    return name.lower()


def _get_model_fields(model: type) -> dict[str, tuple[type, Any]]:
    """
    Extract fields from a model (SQLAlchemy or Pydantic).

    Returns: dict of {field_name: (type, default_value_or_...)}
    """
    fields: dict[str, tuple[type, Any]] = {}

    # Check if it's a Pydantic model
    if hasattr(model, "model_fields"):
        # Pydantic v2
        from pydantic_core import PydanticUndefined

        for name, field_info in model.model_fields.items():
            annotation = field_info.annotation or Any
            # Check for PydanticUndefined (required field with no default)
            if field_info.default is PydanticUndefined:
                default = ...
            else:
                default = field_info.default
            fields[name] = (annotation, default)
        return fields

    # Check if it's a SQLAlchemy model
    if hasattr(model, "__table__"):
        # SQLAlchemy 2.x with type annotations
        try:
            hints = get_type_hints(model)
        except Exception:
            hints = {}

        for column in model.__table__.columns:
            col_name = column.name
            # Get type from annotations or infer from column type
            if col_name in hints:
                col_type = hints[col_name]
            else:
                col_type = _sqlalchemy_type_to_python(column.type)

            # Primary key and autoincrement fields have default
            if column.primary_key or column.autoincrement:
                default = None
            elif column.default is not None:
                default = None  # Has default
            elif column.nullable:
                default = None
            else:
                default = ...  # Required

            fields[col_name] = (col_type, default)
        return fields

    # Fallback: try to get type hints
    try:
        hints = get_type_hints(model)
        for name, hint in hints.items():
            if not name.startswith("_"):
                fields[name] = (hint, ...)
    except Exception:
        pass

    return fields


def _sqlalchemy_type_to_python(sa_type: Any) -> type:
    """Convert SQLAlchemy column type to Python type."""
    type_name = type(sa_type).__name__.upper()

    type_mapping = {
        "INTEGER": int,
        "BIGINTEGER": int,
        "SMALLINTEGER": int,
        "FLOAT": float,
        "REAL": float,
        "DOUBLE": float,
        "NUMERIC": float,
        "DECIMAL": float,
        "STRING": str,
        "TEXT": str,
        "VARCHAR": str,
        "CHAR": str,
        "BOOLEAN": bool,
        "DATE": str,  # ISO format string for LLM
        "DATETIME": str,
        "TIME": str,
        "TIMESTAMP": str,
        "JSON": dict,
        "JSONB": dict,
        "ARRAY": list,
        "UUID": str,
    }

    return type_mapping.get(type_name, str)


def _create_get_tool(
    model: type,
    model_name: str,
    config: ToolConfig,
    executor: Callable[..., Any] | None,
) -> GeneratedTool:
    """Create a get_<model> tool."""
    # Build parameter model
    GetParams = create_model(
        f"Get{model_name.title()}Params",
        id=(int, Field(..., description=f"The ID of the {model_name} to retrieve")),
    )

    def get_func(id: int) -> dict[str, Any]:
        """Get a record by ID."""
        if executor:
            return executor("get", model, id=id)
        return {"error": "No executor configured", "id": id}

    tool_name = config.name_pattern.format(action="get", model=model_name)

    return GeneratedTool(
        name=tool_name,
        description=f"Get a {model_name} by ID",
        func=get_func,
        parameters=GetParams,
        operation="get",
    )


def _create_list_tool(
    model: type,
    model_name: str,
    fields: dict[str, tuple[type, Any]],
    config: ToolConfig,
    executor: Callable[..., Any] | None,
) -> GeneratedTool:
    """Create a list_<model>s tool."""
    # Build filter parameters from fields
    filter_fields: dict[str, Any] = {
        "limit": (
            int,
            Field(
                default=config.default_limit,
                ge=1,
                le=config.max_limit,
                description=f"Maximum number of results (default: {config.default_limit}, max: {config.max_limit})",
            ),
        ),
        "offset": (
            int,
            Field(default=0, ge=0, description="Number of records to skip"),
        ),
    }

    # Add optional filter for each field
    for field_name, (field_type, _) in fields.items():
        # Skip complex types for filtering
        if field_type in (dict, list, set):
            continue
        # Make optional for filtering
        filter_fields[field_name] = (
            Optional[field_type],
            Field(default=None, description=f"Filter by {field_name}"),
        )

    ListParams = create_model(f"List{model_name.title()}sParams", **filter_fields)

    def list_func(**kwargs) -> list[dict[str, Any]]:
        """List records with optional filters."""
        # Extract pagination
        limit = kwargs.pop("limit", config.default_limit)
        offset = kwargs.pop("offset", 0)
        # Remove None filters
        filters = {k: v for k, v in kwargs.items() if v is not None}

        if executor:
            return executor("list", model, limit=limit, offset=offset, **filters)
        return {"error": "No executor configured", "filters": filters}

    tool_name = config.name_pattern.format(action="list", model=f"{model_name}s")

    return GeneratedTool(
        name=tool_name,
        description=f"List {model_name}s with optional filters",
        func=list_func,
        parameters=ListParams,
        operation="list",
    )


def _create_create_tool(
    model: type,
    model_name: str,
    fields: dict[str, tuple[type, Any]],
    config: ToolConfig,
    executor: Callable[..., Any] | None,
) -> GeneratedTool:
    """Create a create_<model> tool."""
    # Build create parameters from fields (excluding id/primary key)
    create_fields: dict[str, Any] = {}

    for field_name, (field_type, default) in fields.items():
        # Skip primary key for create
        if field_name == "id":
            continue
        create_fields[field_name] = (
            field_type if default is ... else Optional[field_type],
            Field(default=default, description=f"The {field_name}"),
        )

    CreateParams = create_model(f"Create{model_name.title()}Params", **create_fields)

    def create_func(**kwargs) -> dict[str, Any]:
        """Create a new record."""
        if executor:
            return executor("create", model, **kwargs)
        return {"error": "No executor configured", "data": kwargs}

    tool_name = config.name_pattern.format(action="create", model=model_name)

    return GeneratedTool(
        name=tool_name,
        description=f"Create a new {model_name}",
        func=create_func,
        parameters=CreateParams,
        operation="create",
    )


def _create_update_tool(
    model: type,
    model_name: str,
    fields: dict[str, tuple[type, Any]],
    config: ToolConfig,
    executor: Callable[..., Any] | None,
) -> GeneratedTool:
    """Create an update_<model> tool."""
    # Build update parameters (all optional except id)
    update_fields: dict[str, Any] = {
        "id": (int, Field(..., description=f"The ID of the {model_name} to update")),
    }

    for field_name, (field_type, _) in fields.items():
        if field_name == "id":
            continue
        update_fields[field_name] = (
            Optional[field_type],
            Field(default=None, description=f"New value for {field_name}"),
        )

    UpdateParams = create_model(f"Update{model_name.title()}Params", **update_fields)

    def update_func(**kwargs) -> dict[str, Any]:
        """Update a record by ID."""
        id = kwargs.pop("id")
        # Remove None values (not being updated)
        updates = {k: v for k, v in kwargs.items() if v is not None}

        if executor:
            return executor("update", model, id=id, **updates)
        return {"error": "No executor configured", "id": id, "updates": updates}

    tool_name = config.name_pattern.format(action="update", model=model_name)

    return GeneratedTool(
        name=tool_name,
        description=f"Update a {model_name} by ID",
        func=update_func,
        parameters=UpdateParams,
        operation="update",
    )


def _create_delete_tool(
    model: type,
    model_name: str,
    config: ToolConfig,
    executor: Callable[..., Any] | None,
) -> GeneratedTool:
    """Create a delete_<model> tool."""
    DeleteParams = create_model(
        f"Delete{model_name.title()}Params",
        id=(int, Field(..., description=f"The ID of the {model_name} to delete")),
    )

    def delete_func(id: int) -> dict[str, Any]:
        """Delete a record by ID."""
        if executor:
            return executor("delete", model, id=id)
        return {"error": "No executor configured", "id": id}

    tool_name = config.name_pattern.format(action="delete", model=model_name)

    return GeneratedTool(
        name=tool_name,
        description=f"Delete a {model_name} by ID",
        func=delete_func,
        parameters=DeleteParams,
        operation="delete",
    )


def tools_from_models(
    *models: type,
    executor: Callable[..., Any] | None = None,
    read_only: bool = False,
    operations: Sequence[str] | None = None,
    name_pattern: str = "{action}_{model}",
    default_limit: int = 20,
    max_limit: int = 100,
) -> list[Callable[..., Any]]:
    """
    Generate CRUD tools from SQLAlchemy or Pydantic models.

    Args:
        *models: One or more model classes to generate tools for
        executor: Function to execute operations. Signature:
            executor(operation: str, model: type, **kwargs) -> Any
            If None, tools return placeholder dicts (useful for testing/mocking)
        read_only: If True, only generate get/list tools
        operations: Specific operations to generate (get, list, create, update, delete)
        name_pattern: Pattern for tool names. Use {action} and {model} placeholders
        default_limit: Default page size for list operations
        max_limit: Maximum page size for list operations

    Returns:
        List of callable tools ready for use with Agent

    Example:
        ```python
        from ai_infra import Agent, tools_from_models

        # Simple usage (generates tools with placeholder executor)
        tools = tools_from_models(User, Product)

        # With SQLAlchemy session
        def execute_crud(operation, model, **kwargs):
            if operation == "get":
                return session.get(model, kwargs["id"])
            elif operation == "list":
                query = session.query(model)
                for k, v in kwargs.items():
                    if k not in ("limit", "offset"):
                        query = query.filter(getattr(model, k) == v)
                return query.offset(kwargs["offset"]).limit(kwargs["limit"]).all()
            # ... etc

        tools = tools_from_models(User, executor=execute_crud)
        agent = Agent(tools=tools)
        ```
    """
    config = ToolConfig(
        name_pattern=name_pattern,
        operations=set(operations) if operations else None,
        default_limit=default_limit,
        max_limit=max_limit,
    )

    # Determine which operations to generate
    if read_only:
        allowed_ops = {"get", "list"}
    elif config.operations:
        allowed_ops = config.operations
    else:
        allowed_ops = {"get", "list", "create", "update", "delete"}

    all_tools: list[StructuredTool] = []

    for model in models:
        model_name = _get_model_name(model)
        fields = _get_model_fields(model)

        if "get" in allowed_ops:
            tool = _create_get_tool(model, model_name, config, executor)
            all_tools.append(tool.to_langchain_tool())

        if "list" in allowed_ops:
            tool = _create_list_tool(model, model_name, fields, config, executor)
            all_tools.append(tool.to_langchain_tool())

        if "create" in allowed_ops:
            tool = _create_create_tool(model, model_name, fields, config, executor)
            all_tools.append(tool.to_langchain_tool())

        if "update" in allowed_ops:
            tool = _create_update_tool(model, model_name, fields, config, executor)
            all_tools.append(tool.to_langchain_tool())

        if "delete" in allowed_ops:
            tool = _create_delete_tool(model, model_name, config, executor)
            all_tools.append(tool.to_langchain_tool())

    return all_tools
