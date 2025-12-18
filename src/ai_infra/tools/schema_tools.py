"""
Schema-to-Tools: Automatically generate CRUD tools from SQLAlchemy/Pydantic models.

This module provides two ways to generate CRUD tools:

1. **tools_from_models()** - Flexible, bring-your-own executor
2. **tools_from_models_sql()** - Zero-config with svc-infra SqlRepository (RECOMMENDED)

Quick Start (Recommended):
    ```python
    from ai_infra import Agent, tools_from_models_sql
    from svc_infra.api.fastapi.db.sql.session import get_session

    async with get_session() as session:
        tools = tools_from_models_sql(User, Product, session=session)
        agent = Agent(tools=tools)
        result = await agent.arun("Create a user named Alice")
    ```

Quick Start (Custom Executor):
    ```python
    from ai_infra import Agent, tools_from_models

    # With custom execution logic
    def my_executor(operation, model, **kwargs):
        if operation == "get":
            return db.query(model).get(kwargs["id"])
        # ... handle other operations

    tools = tools_from_models(User, executor=my_executor)
    agent = Agent(tools=tools)
    ```

FastAPI Integration:
    ```python
    from fastapi import Depends
    from svc_infra.api.fastapi.db.sql.session import SqlSessionDep

    @app.post("/chat")
    async def chat(message: str, session: SqlSessionDep):
        tools = tools_from_models_sql(User, Order, session=session)
        agent = Agent(tools=tools)
        return await agent.arun(message)
    ```

Generated Tools:
    For a model named `User`, generates:
    - get_user(id) - Retrieve by ID
    - list_users(limit, offset) - List with pagination
    - create_user(**fields) - Create new record
    - update_user(id, **fields) - Update existing record
    - delete_user(id) - Delete/soft-delete record
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, get_type_hints

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
                field_default = ...
            else:
                field_default = field_info.default
            fields[name] = (annotation, field_default)
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
            # Default to required (...), then override to None for optional fields
            column_default: object
            if column.primary_key or column.autoincrement:
                column_default = None
            elif column.default is not None:
                column_default = None  # Has default
            elif column.nullable:
                column_default = None
            else:
                column_default = ...  # Required field

            fields[col_name] = (col_type, column_default)
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
            from typing import cast

            return cast("dict[str, Any]", executor("get", model, id=id))
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
            field_type | None,
            Field(default=None, description=f"Filter by {field_name}"),
        )

    ListParams = create_model(f"List{model_name.title()}sParams", **filter_fields)

    def list_func(**kwargs: Any) -> list[dict[str, Any]]:
        """List records with optional filters."""
        # Extract pagination
        limit = int(kwargs.pop("limit", config.default_limit))
        offset = int(kwargs.pop("offset", 0))
        # Remove None filters
        filters = {k: v for k, v in kwargs.items() if v is not None}

        if executor:
            from typing import cast

            return cast(
                "list[dict[str, Any]]",
                executor("list", model, limit=limit, offset=offset, **filters),
            )
        return [{"error": "No executor configured", "filters": filters}]

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
            field_type if default is ... else field_type | None,
            Field(default=default, description=f"The {field_name}"),
        )

    CreateParams = create_model(f"Create{model_name.title()}Params", **create_fields)

    def create_func(**kwargs) -> dict[str, Any]:
        """Create a new record."""
        if executor:
            from typing import cast

            return cast("dict[str, Any]", executor("create", model, **kwargs))
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
            field_type | None,
            Field(default=None, description=f"New value for {field_name}"),
        )

    UpdateParams = create_model(f"Update{model_name.title()}Params", **update_fields)

    def update_func(**kwargs: Any) -> dict[str, Any]:
        """Update a record by ID."""
        id = int(kwargs.pop("id"))
        # Remove None values (not being updated)
        updates = {k: v for k, v in kwargs.items() if v is not None}

        if executor:
            from typing import cast

            return cast("dict[str, Any]", executor("update", model, id=id, **updates))
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
            from typing import cast

            return cast("dict[str, Any]", executor("delete", model, id=id))
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
) -> list[StructuredTool]:
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


def tools_from_models_sql(
    *models: type,
    session: Any,
    read_only: bool = False,
    operations: Sequence[str] | None = None,
    name_pattern: str = "{action}_{model}",
    default_limit: int = 20,
    max_limit: int = 100,
    soft_delete: bool = False,
    id_attr: str = "id",
) -> list[StructuredTool]:
    """
    Generate CRUD tools from SQLAlchemy models with automatic database execution.

    This is the **recommended** way to generate tools for database models. It uses
    svc-infra's SqlRepository for production-ready async CRUD operations with:
    - Automatic session management
    - Soft-delete support
    - Proper error handling
    - Type-safe operations

    Args:
        *models: One or more SQLAlchemy model classes
        session: AsyncSession from SQLAlchemy (use svc-infra's get_session())
        read_only: If True, only generate get/list tools
        operations: Specific operations to generate (get, list, create, update, delete)
        name_pattern: Pattern for tool names. Use {action} and {model} placeholders
        default_limit: Default page size for list operations
        max_limit: Maximum page size for list operations
        soft_delete: If True, delete operations set deleted_at instead of removing
        id_attr: Name of the ID column (default: "id")

    Returns:
        List of callable tools ready for use with Agent

    Example - Basic usage:
        ```python
        from ai_infra import Agent, tools_from_models_sql
        from svc_infra.api.fastapi.db.sql.session import get_session

        async with get_session() as session:
            tools = tools_from_models_sql(User, Product, session=session)
            agent = Agent(tools=tools)
            result = await agent.arun("List all users")
        ```

    Example - FastAPI endpoint:
        ```python
        from fastapi import Depends
        from svc_infra.api.fastapi.db.sql.session import SqlSessionDep

        @app.post("/chat")
        async def chat(message: str, session: SqlSessionDep):
            tools = tools_from_models_sql(User, Order, session=session)
            agent = Agent(tools=tools)
            return await agent.arun(message)
        ```

    Example - Read-only with soft-delete:
        ```python
        tools = tools_from_models_sql(
            User,
            session=session,
            read_only=True,
            soft_delete=True,
        )
        ```

    Note:
        Requires svc-infra package (included as ai-infra dependency).
        For custom execution logic, use `tools_from_models()` with an executor.
    """
    try:
        from svc_infra.db.sql.repository import SqlRepository
    except ImportError as e:
        raise ImportError(
            "tools_from_models_sql requires svc-infra. Install with: pip install svc-infra"
        ) from e

    # Cache repositories per model for efficiency
    _repos: dict[type, SqlRepository] = {}

    def _get_repo(model: type) -> SqlRepository:
        if model not in _repos:
            _repos[model] = SqlRepository(
                model=model,
                id_attr=id_attr,
                soft_delete=soft_delete,
            )
        return _repos[model]

    async def async_executor(operation: str, model: type, **kwargs: Any) -> Any:
        """Execute CRUD operation using svc-infra SqlRepository."""
        repo = _get_repo(model)

        if operation == "get":
            result = await repo.get(session, kwargs["id"])
            if result is None:
                return {"error": f"Not found: id={kwargs['id']}"}
            return _model_to_dict(result)

        elif operation == "list":
            limit = min(kwargs.get("limit", default_limit), max_limit)
            offset = kwargs.get("offset", 0)
            results = await repo.list(session, limit=limit, offset=offset)
            return [_model_to_dict(r) for r in results]

        elif operation == "create":
            # Remove operation metadata from kwargs
            create_data = {k: v for k, v in kwargs.items() if k not in ("limit", "offset")}
            result = await repo.create(session, create_data)
            return _model_to_dict(result)

        elif operation == "update":
            id_value = kwargs.pop("id")
            update_data = {k: v for k, v in kwargs.items() if k not in ("limit", "offset")}
            result = await repo.update(session, id_value, update_data)
            if result is None:
                return {"error": f"Not found: id={id_value}"}
            return _model_to_dict(result)

        elif operation == "delete":
            success = await repo.delete(session, kwargs["id"])
            return {"success": success, "id": kwargs["id"]}

        else:
            return {"error": f"Unknown operation: {operation}"}

    # Sync wrapper for LangChain tools (they expect sync functions)
    def sync_executor(operation: str, model: type, **kwargs: Any) -> Any:
        """Sync wrapper that runs async executor."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside async context - use thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, async_executor(operation, model, **kwargs))
                return future.result()
        else:
            # No running loop - run directly
            return asyncio.run(async_executor(operation, model, **kwargs))

    return tools_from_models(
        *models,
        executor=sync_executor,
        read_only=read_only,
        operations=operations,
        name_pattern=name_pattern,
        default_limit=default_limit,
        max_limit=max_limit,
    )


def _model_to_dict(obj: Any) -> dict[str, Any]:
    """Convert SQLAlchemy model instance to dictionary."""
    if hasattr(obj, "__table__"):
        # SQLAlchemy model
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
    elif hasattr(obj, "model_dump"):
        # Pydantic v2
        from typing import cast

        return cast("dict[str, Any]", obj.model_dump())
    elif hasattr(obj, "dict"):
        # Pydantic v1
        from typing import cast

        return cast("dict[str, Any]", obj.dict())
    elif hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return {"value": str(obj)}
