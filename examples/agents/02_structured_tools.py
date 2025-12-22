#!/usr/bin/env python
"""Structured Tools with Pydantic Schemas Example.

This example demonstrates:
- Tools with Pydantic model input schemas
- Complex nested input validation
- tools_from_functions utility
- Structured output from agents

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ai_infra import Agent
from ai_infra.llm.tools import tools_from_functions

# =============================================================================
# Pydantic Schemas for Tool Inputs
# =============================================================================


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskInput(BaseModel):
    """Input schema for creating a task."""

    title: str = Field(description="Brief title of the task")
    description: str = Field(description="Detailed task description")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority level")
    assignee: str | None = Field(default=None, description="Person to assign the task to")
    due_days: int = Field(default=7, ge=1, le=365, description="Days until due (1-365)")


class SearchFilters(BaseModel):
    """Search filters for database queries."""

    query: str = Field(description="Search query text")
    category: str | None = Field(default=None, description="Filter by category")
    min_price: float | None = Field(default=None, ge=0, description="Minimum price")
    max_price: float | None = Field(default=None, ge=0, description="Maximum price")
    in_stock: bool = Field(default=True, description="Only show in-stock items")
    sort_by: Literal["relevance", "price_asc", "price_desc", "newest"] = Field(
        default="relevance", description="Sort order"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results (1-100)")


class EmailRequest(BaseModel):
    """Email sending request."""

    to: list[str] = Field(description="List of recipient email addresses")
    subject: str = Field(description="Email subject line", max_length=200)
    body: str = Field(description="Email body content")
    cc: list[str] = Field(default_factory=list, description="CC recipients")
    is_html: bool = Field(default=False, description="Whether body is HTML")


# =============================================================================
# Tools with Pydantic Input Schemas
# =============================================================================


# Simulated database
_tasks: list[dict] = []
_products = [
    {"id": 1, "name": "Laptop", "category": "Electronics", "price": 999.99, "stock": 10},
    {"id": 2, "name": "Headphones", "category": "Electronics", "price": 149.99, "stock": 25},
    {"id": 3, "name": "Desk Chair", "category": "Furniture", "price": 299.99, "stock": 5},
    {"id": 4, "name": "Monitor", "category": "Electronics", "price": 399.99, "stock": 0},
    {"id": 5, "name": "Keyboard", "category": "Electronics", "price": 79.99, "stock": 50},
]


def create_task(task: TaskInput) -> str:
    """Create a new task in the task management system.

    Args:
        task: Task details including title, description, priority, and assignee.

    Returns:
        Confirmation message with task ID.
    """
    task_id = len(_tasks) + 1
    task_dict = task.model_dump()
    task_dict["id"] = task_id
    _tasks.append(task_dict)

    return (
        f"✓ Created task #{task_id}: '{task.title}' "
        f"(Priority: {task.priority.value}, Due in {task.due_days} days)"
        + (f", Assigned to: {task.assignee}" if task.assignee else "")
    )


def search_products(filters: SearchFilters) -> str:
    """Search the product catalog with filters.

    Args:
        filters: Search filters including query, category, price range, etc.

    Returns:
        Formatted search results.
    """
    results = _products.copy()

    # Apply filters
    if filters.query:
        query_lower = filters.query.lower()
        results = [p for p in results if query_lower in p["name"].lower()]

    if filters.category:
        results = [p for p in results if p["category"] == filters.category]

    if filters.min_price is not None:
        results = [p for p in results if p["price"] >= filters.min_price]

    if filters.max_price is not None:
        results = [p for p in results if p["price"] <= filters.max_price]

    if filters.in_stock:
        results = [p for p in results if p["stock"] > 0]

    # Sort
    if filters.sort_by == "price_asc":
        results.sort(key=lambda p: p["price"])
    elif filters.sort_by == "price_desc":
        results.sort(key=lambda p: p["price"], reverse=True)
    elif filters.sort_by == "newest":
        results.sort(key=lambda p: p["id"], reverse=True)

    # Limit
    results = results[: filters.limit]

    if not results:
        return "No products found matching your criteria."

    lines = [f"Found {len(results)} product(s):"]
    for p in results:
        stock_status = f"({p['stock']} in stock)" if p["stock"] > 0 else "(Out of stock)"
        lines.append(f"  • {p['name']} - ${p['price']:.2f} {stock_status}")

    return "\n".join(lines)


def send_email(request: EmailRequest) -> str:
    """Send an email to specified recipients.

    Args:
        request: Email details including recipients, subject, and body.

    Returns:
        Confirmation message.
    """
    # In real app, this would send actual emails
    recipients = ", ".join(request.to[:3])
    if len(request.to) > 3:
        recipients += f" (+{len(request.to) - 3} more)"

    cc_info = f" (CC: {len(request.cc)})" if request.cc else ""
    format_info = " [HTML]" if request.is_html else ""

    return f"✓ Email sent to {recipients}{cc_info}{format_info}\n  Subject: {request.subject}"


# =============================================================================
# Using tools_from_functions Utility
# =============================================================================


def list_tasks() -> str:
    """List all tasks in the system.

    Returns:
        Formatted list of all tasks.
    """
    if not _tasks:
        return "No tasks found."

    lines = ["Current tasks:"]
    for task in _tasks:
        lines.append(
            f"  #{task['id']}: {task['title']} "
            f"[{task['priority']}] - Due in {task['due_days']} days"
        )
    return "\n".join(lines)


def get_task(task_id: int) -> str:
    """Get details of a specific task.

    Args:
        task_id: The task ID to retrieve.

    Returns:
        Task details or error message.
    """
    for task in _tasks:
        if task["id"] == task_id:
            return (
                f"Task #{task_id}:\n"
                f"  Title: {task['title']}\n"
                f"  Description: {task['description']}\n"
                f"  Priority: {task['priority']}\n"
                f"  Assignee: {task['assignee'] or 'Unassigned'}\n"
                f"  Due in: {task['due_days']} days"
            )
    return f"Task #{task_id} not found."


# =============================================================================
# Examples
# =============================================================================


def main():
    print("=" * 60)
    print("Structured Tools with Pydantic Schemas")
    print("=" * 60)

    # Create agent with structured tools
    agent = Agent(tools=[create_task, search_products, send_email])

    # Complex task creation
    print("\n1. Creating a task with complex input:")
    result = agent.run(
        "Create a high priority task titled 'Fix login bug' with description "
        "'Users can\\'t log in with SSO' and assign it to Alice. Due in 3 days."
    )
    print(f"   {result}")

    # Product search with filters
    print("\n2. Searching products with filters:")
    result = agent.run(
        "Search for electronics under $200 that are in stock, sorted by price low to high"
    )
    print(f"   {result}")

    # Email with multiple recipients
    print("\n3. Sending a formatted email:")
    result = agent.run(
        "Send an email to alice@example.com and bob@example.com with subject "
        "'Meeting Tomorrow' and body 'Don\\'t forget our 10am sync!'"
    )
    print(f"   {result}")


def tools_from_functions_example():
    """Using tools_from_functions to convert multiple functions at once."""
    print("\n" + "=" * 60)
    print("Using tools_from_functions Utility")
    print("=" * 60)

    # Convert multiple functions to tools at once
    tools = tools_from_functions(
        [
            create_task,
            list_tasks,
            get_task,
        ]
    )

    print(f"\nConverted {len(tools)} functions to tools")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:50]}...")

    # Use with agent
    agent = Agent(tools=tools)

    # Create some tasks
    agent.run(
        "Create a medium priority task 'Review PR #123' with description 'Code review needed'"
    )
    agent.run("Create a low priority task 'Update docs' assigned to Bob")

    # List and get tasks
    result = agent.run("List all tasks and show details of task #1")
    print(f"\nResult:\n{result}")


def nested_schema_example():
    """Tools with deeply nested Pydantic schemas."""
    print("\n" + "=" * 60)
    print("Nested Pydantic Schemas")
    print("=" * 60)

    class Address(BaseModel):
        """Mailing address."""

        street: str
        city: str
        state: str
        zip_code: str
        country: str = "USA"

    class Customer(BaseModel):
        """Customer information."""

        name: str
        email: str
        phone: str | None = None
        addresses: list[Address] = Field(default_factory=list)

    class OrderItem(BaseModel):
        """Item in an order."""

        product_id: int
        quantity: int = Field(ge=1, le=100)
        unit_price: float = Field(ge=0)

    class OrderRequest(BaseModel):
        """Complete order request."""

        customer: Customer
        items: list[OrderItem]
        shipping_address: Address
        notes: str | None = None
        express_shipping: bool = False

    def create_order(order: OrderRequest) -> str:
        """Create a new order with customer and items.

        Args:
            order: Complete order details including customer, items, and shipping.

        Returns:
            Order confirmation with total.
        """
        total = sum(item.unit_price * item.quantity for item in order.items)
        shipping = 9.99 if not order.express_shipping else 24.99

        return (
            f"✓ Order created for {order.customer.name}\n"
            f"  Items: {len(order.items)}\n"
            f"  Subtotal: ${total:.2f}\n"
            f"  Shipping: ${shipping:.2f}\n"
            f"  Total: ${total + shipping:.2f}\n"
            f"  Ship to: {order.shipping_address.city}, {order.shipping_address.state}"
        )

    agent = Agent(tools=[create_order])

    # The agent can handle complex nested structures
    result = agent.run(
        "Create an order for John Doe (john@example.com) with 2 laptops at $999 each. "
        "Ship to 123 Main St, San Francisco, CA 94102. Use express shipping."
    )
    print(f"\nOrder result:\n{result}")


def structured_agent_output():
    """Agent with structured output schema."""
    print("\n" + "=" * 60)
    print("Agent with Structured Output")
    print("=" * 60)

    class AnalysisResult(BaseModel):
        """Structured analysis result."""

        summary: str = Field(description="Brief summary of findings")
        recommendations: list[str] = Field(description="List of recommendations")
        confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
        categories: list[str] = Field(description="Relevant categories")

    # Note: Agent structured output uses output_schema on run/arun
    # This is different from tool input schemas
    agent = Agent(tools=[search_products])

    # Regular run - returns string
    result = agent.run("What electronics do you have in stock?")
    print(f"\nString result:\n{result}")

    # For structured output, you can use LLM directly after getting context
    print("\n(For structured agent output, use LLM.chat with output_schema)")


async def async_structured_example():
    """Async agent with structured tools."""
    print("\n" + "=" * 60)
    print("Async Structured Tools")
    print("=" * 60)

    agent = Agent(tools=[create_task, list_tasks])

    # Create tasks concurrently
    await asyncio.gather(
        agent.arun("Create task 'Deploy v2.0' with high priority"),
        agent.arun("Create task 'Write tests' with medium priority"),
        agent.arun("Create task 'Update changelog' with low priority"),
    )

    # List all tasks
    result = await agent.arun("Show me all tasks")
    print(f"\nAll tasks:\n{result}")


if __name__ == "__main__":
    main()
    tools_from_functions_example()
    nested_schema_example()
    structured_agent_output()

    # Clear tasks for async example
    _tasks.clear()
    asyncio.run(async_structured_example())
