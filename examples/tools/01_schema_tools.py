#!/usr/bin/env python
"""Schema-to-Tools: Auto-Generate CRUD Tools from Models.

This example demonstrates:
- Automatically generating CRUD tools from Pydantic/SQLAlchemy models
- Using tools_from_models() with custom executors
- Using tools_from_models_sql() for zero-config database operations
- Configuring read-only mode, custom operations, and pagination
- Integrating generated tools with an Agent for natural language database access

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY

The tools module provides two main functions:
1. tools_from_models() - Flexible, bring-your-own executor
2. tools_from_models_sql() - Zero-config with SQLAlchemy session (recommended)

Generated tools follow a consistent naming pattern:
- get_{model}(id) - Retrieve by ID
- list_{model}s(limit, offset, **filters) - List with pagination
- create_{model}(**fields) - Create new record
- update_{model}(id, **fields) - Update existing record
- delete_{model}(id) - Delete record
"""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ai_infra import Agent
from ai_infra.tools import tools_from_models

# =============================================================================
# Example 1: Basic Usage with Pydantic Models
# =============================================================================


class User(BaseModel):
    """A user in the system."""

    id: int = Field(default=None, description="User ID (auto-generated)")
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    role: str = Field(default="user", description="User role: admin, user, guest")
    created_at: datetime = Field(default_factory=datetime.now)


class Product(BaseModel):
    """A product in the catalog."""

    id: int = Field(default=None, description="Product ID (auto-generated)")
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price in USD")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(default=True, description="Whether product is in stock")


def example_basic_tools() -> None:
    """Generate tools without an executor (for testing/schema inspection)."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Tool Generation")
    print("=" * 60)

    # Generate tools - without executor, they return placeholder dicts
    tools = tools_from_models(User, Product)

    print(f"\nGenerated {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\nTool schemas (what the LLM sees):")
    for tool in tools[:3]:  # Show first 3
        schema = tool.args_schema.model_json_schema()
        print(f"\n  {tool.name}:")
        print(f"    Parameters: {list(schema.get('properties', {}).keys())}")


# =============================================================================
# Example 2: With In-Memory Database (Custom Executor)
# =============================================================================


class InMemoryDB:
    """Simple in-memory database for demonstration."""

    def __init__(self) -> None:
        self.tables: dict[str, dict[int, dict]] = {}
        self.counters: dict[str, int] = {}

    def execute(
        self, operation: str, model: type, **kwargs: Any
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Execute a CRUD operation."""
        table_name = model.__name__.lower()

        if table_name not in self.tables:
            self.tables[table_name] = {}
            self.counters[table_name] = 0

        table = self.tables[table_name]

        if operation == "get":
            record = table.get(kwargs["id"])
            if record:
                return record
            return {"error": f"{model.__name__} with id={kwargs['id']} not found"}

        elif operation == "list":
            limit = kwargs.pop("limit", 20)
            offset = kwargs.pop("offset", 0)
            # Apply filters
            results = []
            for record in table.values():
                match = all(record.get(k) == v for k, v in kwargs.items() if v is not None)
                if match:
                    results.append(record)
            return results[offset : offset + limit]

        elif operation == "create":
            self.counters[table_name] += 1
            new_id = self.counters[table_name]
            record = {"id": new_id, **kwargs}
            if "created_at" not in record:
                record["created_at"] = datetime.now().isoformat()
            table[new_id] = record
            return record

        elif operation == "update":
            record_id = kwargs.pop("id")
            if record_id not in table:
                return {"error": f"{model.__name__} with id={record_id} not found"}
            table[record_id].update(kwargs)
            return table[record_id]

        elif operation == "delete":
            record_id = kwargs["id"]
            if record_id in table:
                deleted = table.pop(record_id)
                return {"deleted": True, "record": deleted}
            return {"error": f"{model.__name__} with id={record_id} not found"}

        return {"error": f"Unknown operation: {operation}"}


def example_with_executor() -> None:
    """Generate tools with a custom in-memory database executor."""
    print("\n" + "=" * 60)
    print("Example 2: Tools with Custom Executor (In-Memory DB)")
    print("=" * 60)

    # Create in-memory database
    db = InMemoryDB()

    # Generate tools with executor
    tools = tools_from_models(User, Product, executor=db.execute)

    print(f"\nGenerated {len(tools)} tools with database backend")

    # Test the tools directly
    print("\n--- Direct Tool Execution ---")

    # Find create_user tool
    create_user = next(t for t in tools if t.name == "create_user")
    list_users = next(t for t in tools if t.name == "list_users")
    get_user = next(t for t in tools if t.name == "get_user")

    # Create some users
    user1 = create_user.invoke({"name": "Alice", "email": "alice@example.com"})
    print(f"Created: {user1}")

    user2 = create_user.invoke({"name": "Bob", "email": "bob@example.com", "role": "admin"})
    print(f"Created: {user2}")

    # List users
    users = list_users.invoke({"limit": 10})
    print(f"All users: {users}")

    # Get specific user
    user = get_user.invoke({"id": 1})
    print(f"User 1: {user}")


# =============================================================================
# Example 3: Read-Only Mode
# =============================================================================


def example_read_only() -> None:
    """Generate read-only tools (get and list only)."""
    print("\n" + "=" * 60)
    print("Example 3: Read-Only Tools")
    print("=" * 60)

    # Generate read-only tools
    tools = tools_from_models(User, Product, read_only=True)

    print(f"\nGenerated {len(tools)} read-only tools:")
    for tool in tools:
        print(f"  - {tool.name}")

    print("\nNote: No create, update, or delete tools generated!")


# =============================================================================
# Example 4: Custom Operations
# =============================================================================


def example_custom_operations() -> None:
    """Generate only specific operations."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Operations")
    print("=" * 60)

    # Generate only get and create tools
    tools = tools_from_models(User, operations=["get", "create"])

    print("\nGenerated tools with custom operations (get, create only):")
    for tool in tools:
        print(f"  - {tool.name}")


# =============================================================================
# Example 5: Agent with Schema Tools
# =============================================================================


async def example_with_agent() -> None:
    """Use generated tools with an Agent for natural language database access."""
    print("\n" + "=" * 60)
    print("Example 5: Agent with Schema Tools")
    print("=" * 60)

    # Create database with some initial data
    db = InMemoryDB()

    # Seed some data
    db.execute("create", User, name="Alice Smith", email="alice@company.com", role="admin")
    db.execute("create", User, name="Bob Johnson", email="bob@company.com", role="user")
    db.execute("create", User, name="Charlie Brown", email="charlie@company.com", role="user")

    db.execute("create", Product, name="Laptop", price=999.99, category="Electronics")
    db.execute("create", Product, name="Mouse", price=29.99, category="Electronics")
    db.execute("create", Product, name="Desk Chair", price=249.99, category="Furniture")

    # Generate tools
    tools = tools_from_models(User, Product, executor=db.execute)

    # Create agent
    agent = Agent(
        tools=tools,
        system_prompt="""You are a helpful database assistant.
You can query and manage users and products in the database.
When asked about data, use the appropriate tools to fetch and display information.
Format results in a clear, readable way.""",
    )

    print("\n--- Natural Language Database Queries ---\n")

    # Query examples
    queries = [
        "How many users do we have? List them all.",
        "Show me all Electronics products",
        "Create a new user named Diana with email diana@company.com",
        "What's the most expensive product?",
    ]

    for query in queries:
        print(f"Query: {query}")
        try:
            result = await agent.arun(query)
            print(f"Response: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# =============================================================================
# Example 6: Pagination Configuration
# =============================================================================


def example_pagination() -> None:
    """Configure pagination settings for list operations."""
    print("\n" + "=" * 60)
    print("Example 6: Pagination Configuration")
    print("=" * 60)

    db = InMemoryDB()

    # Create many products
    for i in range(50):
        db.execute(
            "create", Product, name=f"Product {i + 1}", price=9.99 * (i + 1), category="Test"
        )

    # Generate tools with custom pagination
    tools = tools_from_models(
        Product,
        executor=db.execute,
        default_limit=5,  # Default page size
        max_limit=10,  # Maximum allowed page size
    )

    list_products = next(t for t in tools if t.name == "list_products")

    print("\nWith default_limit=5, max_limit=10:")

    # Default pagination
    page1 = list_products.invoke({"limit": 5, "offset": 0})
    print(f"Page 1 (5 items): {[p['name'] for p in page1]}")

    page2 = list_products.invoke({"limit": 5, "offset": 5})
    print(f"Page 2 (5 items): {[p['name'] for p in page2]}")


# =============================================================================
# Example 7: Custom Naming Pattern
# =============================================================================


def example_custom_naming() -> None:
    """Use custom naming patterns for generated tools."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Naming Pattern")
    print("=" * 60)

    # Default pattern: {action}_{model}
    default_tools = tools_from_models(User, name_pattern="{action}_{model}")
    print("\nDefault pattern '{action}_{model}':")
    for tool in default_tools:
        print(f"  - {tool.name}")

    # Custom pattern: db_{action}_{model}
    custom_tools = tools_from_models(User, name_pattern="db_{action}_{model}")
    print("\nCustom pattern 'db_{action}_{model}':")
    for tool in custom_tools:
        print(f"  - {tool.name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Schema-to-Tools Examples")
    print("Auto-Generate CRUD Tools from Pydantic/SQLAlchemy Models")
    print("=" * 60)

    # Sync examples
    example_basic_tools()
    example_with_executor()
    example_read_only()
    example_custom_operations()
    example_pagination()
    example_custom_naming()

    # Async example
    print("\n" + "=" * 60)
    print("Running async Agent example...")
    print("(Requires API key - will skip if not available)")
    print("=" * 60)

    try:
        asyncio.run(example_with_agent())
    except Exception as e:
        print(f"Agent example skipped: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
