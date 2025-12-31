# Object Tools

> Auto-generate AI tools from any Python object's methods.

## Quick Start

```python
from ai_infra import Agent, tools_from_object

class Calculator:
    """A simple calculator."""

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

calc = Calculator()
tools = tools_from_object(calc)

agent = Agent(tools=tools)
result = agent.run("What is 5 + 3?")
```

---

## Overview

`tools_from_object()` automatically converts an object's methods into AI-compatible function tools. This enables agents to interact with any Python object without manual tool definition.

**Key Benefits:**
- Zero boilerplate — just pass your object
- Automatic schema extraction from type hints
- Method docstrings become tool descriptions (LLM sees these!)
- Works with any class — services, controllers, managers, etc.

---

## Function Signature

```python
def tools_from_object(
    obj: Any,
    *,
    methods: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
    include_private: bool = False,
    async_wrapper: bool = True,
) -> list[Callable]:
    """Convert an object's methods into AI function tools.

    Args:
        obj: The object instance to convert.
        methods: Specific methods to include (None = all public methods).
        exclude: Methods to exclude from conversion.
        prefix: Tool name prefix (default: class name in snake_case).
        include_private: Include _underscore methods (default: False).
        async_wrapper: Wrap sync methods for async compatibility (default: True).

    Returns:
        List of callable functions compatible with ai-infra Agent.

    Example:
        >>> tools = tools_from_object(my_service)
        >>> agent = Agent(tools=tools)
        >>> agent.run("Get the user with ID 123")
    """
```

---

## Parameters

### `obj` (required)

The object instance whose methods will be converted to tools.

```python
# Pass an instance, not a class
service = UserService(db=database)
tools = tools_from_object(service)  # [OK] Correct

# NOT the class itself
tools = tools_from_object(UserService)  # [X] Wrong
```

### `methods`

Explicitly specify which methods to include. If `None`, includes all public methods.

```python
class OrderService:
    def create_order(self, items: list[str]) -> Order: ...
    def get_order(self, order_id: str) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def _internal_validate(self, order: Order) -> bool: ...

# Only expose specific methods
tools = tools_from_object(
    service,
    methods=["create_order", "get_order"],  # Excludes cancel_order
)
```

### `exclude`

Methods to exclude from conversion.

```python
# Expose all except cancel
tools = tools_from_object(
    service,
    exclude=["cancel_order", "_internal_validate"],
)
```

### `prefix`

Tool name prefix. Default is the class name in snake_case.

```python
class RobotArm:
    def move(self, x: float, y: float) -> str: ...
    def home(self) -> str: ...

arm = RobotArm()

# Default: tools named "robot_arm_move", "robot_arm_home"
tools = tools_from_object(arm)

# Custom prefix: tools named "arm_move", "arm_home"
tools = tools_from_object(arm, prefix="arm")

# No prefix: tools named "move", "home"
tools = tools_from_object(arm, prefix="")
```

### `include_private`

Whether to include methods starting with `_` (underscore).

```python
class Service:
    def public_method(self) -> str: ...
    def _helper_method(self) -> str: ...
    def __dunder_method__(self) -> str: ...  # Never included

# Default: only public_method
tools = tools_from_object(service)

# Include _helper_method too
tools = tools_from_object(service, include_private=True)
```

### `async_wrapper`

Whether to wrap synchronous methods for async compatibility.

```python
class SyncService:
    def fetch_data(self) -> dict: ...  # Sync method

# Default: wraps in async-compatible function
tools = tools_from_object(service, async_wrapper=True)

# Keep as sync (for sync-only agents)
tools = tools_from_object(service, async_wrapper=False)
```

---

## Method Filtering Rules

Methods are filtered in this order:

1. **Exclude dunder methods** (`__init__`, `__str__`, etc.) — always excluded
2. **Exclude private methods** (starting with `_`) — unless `include_private=True`
3. **Apply `methods` filter** — if specified, only include these
4. **Apply `exclude` filter** — remove any in this list
5. **Exclude non-callables** — skip properties, class attributes

```python
class Example:
    def __init__(self): ...           # [X] Dunder - excluded
    def __str__(self): ...            # [X] Dunder - excluded
    def _private(self): ...           # [X] Private - excluded by default
    def public_action(self): ...      # [OK] Included

    @property
    def value(self): ...              # [X] Property - excluded
```

---

## Docstring Inheritance

Tool descriptions are derived from method docstrings. The LLM sees these descriptions when deciding which tool to use.

### Basic Docstrings

```python
class OrderService:
    def create_order(self, items: list[str]) -> Order:
        """Create a new order with the given items.

        Args:
            items: List of product IDs to order.

        Returns:
            The created Order object.
        """
        ...

# LLM sees: "Create a new order with the given items..."
```

### Dynamic Docstring Enhancement

When using `prefix`, docstrings are enhanced with context:

```python
arm = RobotArm()
tools = tools_from_object(arm, prefix="arm")

# Original docstring: "Move to position."
# Enhanced: "Move arm to position."  (if no prefix in original)
```

### No Docstring Fallback

If a method has no docstring, a default is generated:

```python
def move(self, x: float, y: float) -> str:
    return "moved"

# Generated: "Call move on {class_name}. Parameters: x (float), y (float)"
```

---

## Type Hint Extraction

Parameter types are extracted from type hints and used for:
- Agent schema generation
- Input validation (when using structured output)
- Documentation

```python
class Service:
    def process(
        self,
        user_id: str,                           # Required string
        count: int = 10,                        # Optional int with default
        options: dict[str, Any] | None = None,  # Optional dict
    ) -> ProcessResult:
        ...

# Extracted schema:
# {
#     "user_id": {"type": "string", "required": True},
#     "count": {"type": "integer", "default": 10},
#     "options": {"type": "object", "nullable": True}
# }
```

### Pydantic Model Parameters

Methods with Pydantic model parameters are fully supported:

```python
from pydantic import BaseModel

class MoveRequest(BaseModel):
    x: float
    y: float
    speed: float = 1.0

class Robot:
    def move(self, request: MoveRequest) -> str:
        """Move the robot to a position."""
        ...

tools = tools_from_object(robot)
# Tool accepts: {"request": {"x": 10, "y": 20, "speed": 0.5}}
```

---

## Decorators (Advanced)

For fine-grained control, use decorators to configure individual methods:

### `@tool` Decorator

```python
from ai_infra.tools import tool

class Service:
    @tool(name="fetch_user", description="Get a user by their ID")
    def get_user(self, user_id: str) -> User:
        ...

    @tool(exclude=True)  # Exclude from tool generation
    def internal_method(self) -> None:
        ...
```

### `@tool_exclude` Decorator

```python
from ai_infra.tools import tool_exclude

class Service:
    def public_action(self) -> str:
        """This becomes a tool."""
        ...

    @tool_exclude
    def helper(self) -> str:
        """This is NOT a tool."""
        ...
```

---

## Usage Examples

### Service Layer

```python
class UserService:
    def __init__(self, db: Database):
        self.db = db

    def get_user(self, user_id: str) -> User:
        """Retrieve a user by ID."""
        return self.db.users.get(user_id)

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        return self.db.users.create(name=name, email=email)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID."""
        return self.db.users.delete(user_id)

# Convert to tools
service = UserService(db=database)
tools = tools_from_object(service, prefix="user")

# Use with agent
agent = Agent(tools=tools)
agent.run("Create a user named Alice with email alice@example.com")
```

### Controller Pattern (Robotics/IoT)

```python
class RobotController:
    def __init__(self, name: str):
        self.name = name
        self.position = {"x": 0, "y": 0}

    def move(self, x: float, y: float) -> str:
        """Move robot to coordinates."""
        self.position = {"x": x, "y": y}
        return f"Moved to ({x}, {y})"

    def home(self) -> str:
        """Return robot to home position."""
        self.position = {"x": 0, "y": 0}
        return "Homed"

    def status(self) -> dict:
        """Get current robot status."""
        return {"name": self.name, "position": self.position}

robot = RobotController("arm-1")
tools = tools_from_object(robot, prefix="robot")

agent = Agent(tools=tools)
agent.run("Move the robot to position (10, 20)")
```

### Financial Services

```python
class PortfolioManager:
    def get_allocation(self) -> dict[str, float]:
        """Get current portfolio allocation percentages."""
        ...

    def rebalance(self, target: dict[str, float]) -> str:
        """Rebalance portfolio to target allocation."""
        ...

    def buy(self, symbol: str, amount: Decimal) -> str:
        """Buy shares of a stock."""
        ...

    def sell(self, symbol: str, amount: Decimal) -> str:
        """Sell shares of a stock."""
        ...

manager = PortfolioManager(account_id="12345")
tools = tools_from_object(manager, prefix="portfolio")

agent = Agent(tools=tools)
agent.run("Rebalance my portfolio to 60% stocks and 40% bonds")
```

---

## Comparison with `tools_from_models`

| Feature | `tools_from_object` | `tools_from_models` |
|---------|---------------------|---------------------|
| Input | Object instance | Pydantic model class |
| Output | Tools from methods | CRUD tools for model |
| Use Case | Services, controllers | Data models, entities |
| Schema Source | Type hints on methods | Pydantic field definitions |

**Use `tools_from_object`** when you have a service class with business logic methods.

**Use `tools_from_models`** when you need CRUD operations on data models.

```python
# Service with methods → tools_from_object
class OrderService:
    def create_order(self, items: list[str]) -> Order: ...
    def ship_order(self, order_id: str) -> bool: ...

tools = tools_from_object(OrderService(db))

# Data model → tools_from_models
class Order(BaseModel):
    id: str
    items: list[str]
    status: str

tools = tools_from_models([Order])
```

---

## Error Handling

Tools generated by `tools_from_object` propagate exceptions from the underlying methods:

```python
class Service:
    def risky_action(self) -> str:
        raise ValueError("Something went wrong")

tools = tools_from_object(Service())
agent = Agent(tools=tools)

try:
    agent.run("Do the risky action")
except ValueError as e:
    print(f"Tool error: {e}")
```

For custom error handling, wrap in the method:

```python
class Service:
    def safe_action(self) -> str:
        try:
            # risky operation
            return "success"
        except Exception as e:
            return f"Error: {e}"
```

---

## Best Practices

1. **Write good docstrings** — LLMs use them to understand tool purpose
2. **Use type hints** — Required for proper schema generation
3. **Keep methods focused** — One action per method
4. **Use descriptive names** — `create_user` not `cu`
5. **Return useful messages** — Confirmations the LLM can relay to users
6. **Handle errors gracefully** — Don't let tools crash the agent

---

## See Also

- [Schema Tools](./schema-tools.md) — CRUD tools from Pydantic models
- [Progress Streaming](./progress.md) — Stream progress updates from tools
