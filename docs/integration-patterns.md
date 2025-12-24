# Integration Patterns — tools_from_object()

> **Purpose**: Cross-package guide for exposing any domain service to LLM agents
>
> **Module**: `ai_infra.tools.object_tools`
>
> **See Also**: [Object Tools Reference](tools/object-tools.md)

---

## Overview

`tools_from_object()` is a generic utility designed for use by any domain package
(fin-infra, robo-infra, custom services) to expose Python objects as AI tools.
This guide shows integration patterns and best practices for various domains.

---

## Pattern 1: Financial Services

Expose portfolio managers, trading systems, or analytics to LLM agents.

### Portfolio Analyzer

```python
from decimal import Decimal
from datetime import date

from ai_infra import Agent, tools_from_object


class PortfolioAnalyzer:
    """Analyze investment portfolios with LLM assistance."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    def get_total_value(self) -> Decimal:
        """Get the current total portfolio value in USD."""
        # In production: call fin-infra portfolio service
        return Decimal("125432.50")

    def get_allocation(self) -> dict[str, float]:
        """Get current asset allocation percentages by asset class."""
        return {
            "stocks": 60.0,
            "bonds": 25.0,
            "cash": 10.0,
            "crypto": 5.0,
        }

    def get_returns(self, period: str = "ytd") -> float:
        """Get portfolio returns for a time period.

        Args:
            period: Time period - 'ytd', 'mtd', '1y', '3y', or '5y'.

        Returns:
            Return percentage (e.g., 12.5 for 12.5% return).
        """
        returns = {"ytd": 8.5, "mtd": 1.2, "1y": 15.3, "3y": 42.1, "5y": 87.2}
        return returns.get(period, 0.0)

    def get_top_holdings(self, count: int = 5) -> list[dict]:
        """Get the top holdings by current value.

        Args:
            count: Number of holdings to return.

        Returns:
            List of holdings with symbol, name, value, and percentage.
        """
        return [
            {"symbol": "AAPL", "name": "Apple Inc.", "value": 15000, "pct": 12.0},
            {"symbol": "MSFT", "name": "Microsoft", "value": 12000, "pct": 9.6},
            {"symbol": "GOOGL", "name": "Alphabet", "value": 10000, "pct": 8.0},
        ][:count]


# Create tools from the analyzer
analyzer = PortfolioAnalyzer(user_id="user_123")
tools = tools_from_object(analyzer, prefix="portfolio")

# Use with Agent
agent = Agent(
    tools=tools,
    system_prompt="You are a financial advisor assistant.",
)

# Now the agent can answer questions like:
# - "What is my total portfolio value?"
# - "Show me my asset allocation"
# - "What are my top 3 holdings?"
# - "What is my year-to-date return?"
result = agent.run("What's my portfolio allocation and YTD return?")
```

### Generated Tools

The above creates these tools for the LLM:

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `portfolio_get_total_value` | Get the current total portfolio value in USD | — |
| `portfolio_get_allocation` | Get current asset allocation percentages | — |
| `portfolio_get_returns` | Get portfolio returns for a time period | `period: str` |
| `portfolio_get_top_holdings` | Get the top holdings by current value | `count: int` |

---

## Pattern 2: Multi-Object Composition

Combine multiple service objects into a unified agent.

```python
from ai_infra import Agent, tools_from_object


class AccountService:
    """Manage user accounts."""

    def get_balance(self, account_id: str) -> float:
        """Get the current balance for an account."""
        return 5432.10

    def list_transactions(self, account_id: str, limit: int = 10) -> list[dict]:
        """List recent transactions for an account."""
        return [{"date": "2025-01-15", "amount": -50.00, "description": "Coffee"}]


class BudgetService:
    """Manage budgets and spending."""

    def get_monthly_budget(self, category: str) -> float:
        """Get the monthly budget for a spending category."""
        budgets = {"food": 500, "transport": 200, "entertainment": 150}
        return budgets.get(category, 0)

    def get_spending_summary(self) -> dict[str, float]:
        """Get current month spending by category."""
        return {"food": 320.50, "transport": 85.00, "entertainment": 45.00}


# Create separate tool sets
account_tools = tools_from_object(AccountService(), prefix="accounts")
budget_tools = tools_from_object(BudgetService(), prefix="budgets")

# Combine into one agent
agent = Agent(
    tools=account_tools + budget_tools,
    system_prompt="You are a personal finance assistant.",
)

# Agent can now query both services
result = agent.run("What's my account balance and how much have I spent on food?")
```

---

## Pattern 3: Domain-Specific Wrappers

Create domain-specific wrapper functions that add business logic.

```python
from ai_infra import Agent
from ai_infra.tools import tools_from_object, tool_exclude


class TradingService:
    """Execute trades with safety checks."""

    def __init__(self, user_id: str, max_trade_size: float = 10000):
        self.user_id = user_id
        self.max_trade_size = max_trade_size

    def get_quote(self, symbol: str) -> dict:
        """Get current quote for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            Quote with price, bid, ask, and volume.
        """
        return {"symbol": symbol, "price": 150.25, "bid": 150.20, "ask": 150.30}

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
    ) -> dict:
        """Place a trade order with safety limits.

        Args:
            symbol: Stock ticker symbol.
            quantity: Number of shares.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.

        Returns:
            Order confirmation with order ID and status.
        """
        # Safety check (always enforced)
        quote = self.get_quote(symbol)
        estimated_value = quote["price"] * quantity

        if estimated_value > self.max_trade_size:
            return {
                "status": "rejected",
                "reason": f"Order exceeds max size ${self.max_trade_size}",
            }

        return {
            "order_id": "ORD-123456",
            "status": "submitted",
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
        }

    @tool_exclude
    def _execute_trade(self, order_id: str) -> None:
        """Internal method - not exposed to LLM."""
        pass


def trading_service_to_tools(
    user_id: str,
    max_trade_size: float = 10000,
) -> list:
    """Create AI tools for trading with configured safety limits.

    This is a domain-specific wrapper that enforces business rules.

    Args:
        user_id: User identifier for the trading session.
        max_trade_size: Maximum allowed trade value in USD.

    Returns:
        List of tools for Agent use.
    """
    service = TradingService(user_id, max_trade_size)
    return tools_from_object(service, prefix="trading")


# Usage
tools = trading_service_to_tools("user_123", max_trade_size=5000)
agent = Agent(tools=tools)

# The @tool_exclude decorator ensures _execute_trade is never exposed
# max_trade_size is enforced in place_order regardless of what LLM asks
```

---

## Pattern 4: Async Service Integration

Handle async services properly with tools_from_object.

```python
import asyncio
from ai_infra import Agent, tools_from_object


class AsyncDataService:
    """Service with async methods (e.g., database, API calls)."""

    async def fetch_user(self, user_id: str) -> dict:
        """Fetch user details from database.

        Args:
            user_id: Unique user identifier.

        Returns:
            User record with name, email, and created date.
        """
        # Simulate async database call
        await asyncio.sleep(0.1)
        return {
            "id": user_id,
            "name": "Alice Smith",
            "email": "alice@example.com",
        }

    async def search_products(self, query: str, limit: int = 10) -> list[dict]:
        """Search products by keyword.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching products.
        """
        await asyncio.sleep(0.1)
        return [
            {"id": "P001", "name": "Widget Pro", "price": 29.99},
            {"id": "P002", "name": "Widget Basic", "price": 19.99},
        ][:limit]


# Async methods are automatically wrapped correctly
service = AsyncDataService()
tools = tools_from_object(service, prefix="data")

# Agent handles async tools seamlessly
agent = Agent(tools=tools)
result = await agent.arun("Find the user with ID 'user_123' and search for widgets")
```

---

## Pattern 5: Selective Method Exposure

Control exactly which methods are exposed to the LLM.

```python
from ai_infra import Agent, tools_from_object
from ai_infra.tools import tool, tool_exclude


class AdminService:
    """Administrative service with mixed public/internal methods."""

    # Explicitly marked as a tool with custom name
    @tool(name="get_user_info", description="Look up user account information")
    def fetch_user_details(self, user_id: str) -> dict:
        """Internal name differs from tool name."""
        return {"id": user_id, "status": "active"}

    # Public method - included by default
    def get_system_status(self) -> str:
        """Get current system operational status."""
        return "operational"

    # Explicitly excluded - never exposed to LLM
    @tool_exclude
    def delete_user(self, user_id: str) -> bool:
        """Dangerous operation - excluded from tools."""
        return True

    # Private method - excluded by default (starts with _)
    def _internal_audit(self) -> None:
        """Internal audit logging."""
        pass


# Method 1: Default filtering (public methods, respects decorators)
tools = tools_from_object(AdminService())
# Creates: admin_service_get_user_info, admin_service_get_system_status
# Excludes: delete_user (decorator), _internal_audit (private)

# Method 2: Explicit method list
tools = tools_from_object(
    AdminService(),
    methods=["get_system_status"],  # Only this method
)

# Method 3: Exclusion list
tools = tools_from_object(
    AdminService(),
    exclude=["get_system_status"],  # Everything except this
)
```

---

## Pattern 6: Read-Only Property Access

Expose properties as getter tools using `tools_from_object_with_properties`.

```python
from ai_infra.tools import tools_from_object_with_properties


class SystemMonitor:
    """System monitoring with property-based metrics."""

    @property
    def cpu_usage(self) -> float:
        """Current CPU usage percentage."""
        return 45.2

    @property
    def memory_usage(self) -> float:
        """Current memory usage percentage."""
        return 62.8

    @property
    def disk_usage(self) -> dict[str, float]:
        """Disk usage by mount point."""
        return {"/": 75.0, "/data": 45.0}

    def get_processes(self, limit: int = 10) -> list[dict]:
        """Get top processes by CPU usage."""
        return [{"name": "python", "cpu": 12.5, "memory": 150.0}]


# Include properties as getter tools
monitor = SystemMonitor()
tools = tools_from_object_with_properties(monitor, prefix="system")

# Creates:
# - system_get_cpu_usage() -> Current CPU usage percentage
# - system_get_memory_usage() -> Current memory usage percentage  
# - system_get_disk_usage() -> Disk usage by mount point
# - system_get_processes(limit) -> Get top processes by CPU usage
```

---

## Best Practices

### 1. Write Clear Docstrings

Docstrings become tool descriptions that the LLM reads. Make them clear:

```python
# ❌ Poor: LLM doesn't understand what this does
def get_data(self, id: str) -> dict:
    """Get data."""
    ...

# ✅ Good: LLM understands purpose and parameters
def get_user_profile(self, user_id: str) -> dict:
    """Get a user's profile information including name, email, and preferences.

    Args:
        user_id: The unique identifier for the user (e.g., 'user_123').

    Returns:
        Profile with name, email, created_at, and preferences dict.
    """
    ...
```

### 2. Use Type Hints

Type hints help the LLM understand parameter types:

```python
# ❌ No type hints - LLM may guess wrong types
def search(self, query, limit):
    ...

# ✅ Clear type hints - LLM knows exact types
def search(self, query: str, limit: int = 10) -> list[dict]:
    ...
```

### 3. Return Serializable Data

Return types that serialize to JSON cleanly:

```python
# ❌ Returns complex object - may not serialize
def get_user(self, user_id: str) -> User:
    return self.db.get(user_id)  # SQLAlchemy model

# ✅ Returns dict - clean JSON serialization
def get_user(self, user_id: str) -> dict:
    user = self.db.get(user_id)
    return {"id": user.id, "name": user.name, "email": user.email}
```

### 4. Implement Safety Guards

Don't rely on the LLM to enforce limits:

```python
class TransferService:
    def __init__(self, max_amount: float = 10000):
        self.max_amount = max_amount

    def transfer(self, amount: float, to_account: str) -> dict:
        """Transfer funds to another account.

        Args:
            amount: Amount to transfer (max $10,000).
            to_account: Destination account ID.
        """
        # ALWAYS enforce in code, never trust LLM
        if amount > self.max_amount:
            return {"error": f"Amount exceeds limit ${self.max_amount}"}

        return {"status": "completed", "amount": amount}
```

### 5. Use Prefixes Meaningfully

Prefixes help the LLM understand tool context:

```python
# Multiple services with clear prefixes
account_tools = tools_from_object(AccountService(), prefix="account")
payment_tools = tools_from_object(PaymentService(), prefix="payment")
support_tools = tools_from_object(SupportService(), prefix="support")

# LLM sees: account_get_balance, payment_process, support_create_ticket
```

---

## See Also

- [Object Tools Reference](tools/object-tools.md) — Complete API documentation
- [Agent Documentation](getting-started.md) — Using tools with Agent
- [svc-infra Integration Patterns](../../svc-infra/docs/integration-patterns.md) — REST API patterns
