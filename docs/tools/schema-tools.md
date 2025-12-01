# Schema Tools

> Auto-generate CRUD tools from Pydantic models.

## Quick Start

```python
from pydantic import BaseModel
from ai_infra import Agent, tools_from_models

class User(BaseModel):
    id: int
    name: str
    email: str

tools = tools_from_models([User])
agent = Agent(tools=tools)

result = agent.run("Create a user named John with email john@example.com")
```

---

## Overview

`tools_from_models()` automatically generates Create, Read, Update, Delete (CRUD) tools from your Pydantic models. This enables agents to interact with your data models without manual tool definition.

---

## Basic Usage

### From Pydantic Models

```python
from pydantic import BaseModel, Field
from ai_infra import tools_from_models

class Product(BaseModel):
    id: int
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = True

# Generate CRUD tools
tools = tools_from_models([Product])

# Tools created:
# - create_product(name, price, in_stock) -> Product
# - get_product(id) -> Product
# - update_product(id, name?, price?, in_stock?) -> Product
# - delete_product(id) -> bool
# - list_products() -> list[Product]
```

### With Agent

```python
from ai_infra import Agent

agent = Agent(tools=tools)

# Create
result = agent.run("Create a product called 'Widget' priced at $29.99")

# Read
result = agent.run("Get the product with ID 1")

# Update
result = agent.run("Update product 1 to be out of stock")

# Delete
result = agent.run("Delete product 1")

# List
result = agent.run("List all products")
```

---

## Custom Handlers

Provide your own implementation for CRUD operations:

```python
# In-memory storage
products_db = {}

def create_product(name: str, price: float, in_stock: bool = True) -> Product:
    product_id = len(products_db) + 1
    product = Product(id=product_id, name=name, price=price, in_stock=in_stock)
    products_db[product_id] = product
    return product

def get_product(id: int) -> Product:
    return products_db.get(id)

def update_product(id: int, **updates) -> Product:
    product = products_db[id]
    for key, value in updates.items():
        if value is not None:
            setattr(product, key, value)
    return product

def delete_product(id: int) -> bool:
    return products_db.pop(id, None) is not None

def list_products() -> list[Product]:
    return list(products_db.values())

tools = tools_from_models(
    [Product],
    handlers={
        "create": create_product,
        "get": get_product,
        "update": update_product,
        "delete": delete_product,
        "list": list_products,
    }
)
```

---

## SQLAlchemy Integration

Use `tools_from_models_sql()` for database-backed models:

```python
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base
from ai_infra import tools_from_models_sql

Base = declarative_base()

class ProductModel(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Float)

# Generate tools with automatic DB operations
tools = tools_from_models_sql(
    [ProductModel],
    session_factory=SessionLocal,  # SQLAlchemy session factory
)

agent = Agent(tools=tools)
result = agent.run("Create a product named 'Gadget' for $49.99")
# Automatically saves to database!
```

---

## Multiple Models

```python
class User(BaseModel):
    id: int
    name: str
    email: str

class Order(BaseModel):
    id: int
    user_id: int
    total: float
    status: str

class Product(BaseModel):
    id: int
    name: str
    price: float

# Generate tools for all models
tools = tools_from_models([User, Order, Product])

agent = Agent(tools=tools)
result = agent.run("""
    1. Create a user named Alice
    2. Create a product called Widget for $10
    3. Create an order for user 1 with total $30
""")
```

---

## Customization

### Select Operations

```python
# Only create and read operations
tools = tools_from_models(
    [Product],
    operations=["create", "get", "list"]  # No update/delete
)
```

### Custom Tool Names

```python
tools = tools_from_models(
    [Product],
    prefix="inventory_",  # inventory_create_product, etc.
)
```

### Add Validation

```python
from pydantic import BaseModel, Field, validator

class Product(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)

    @validator("name")
    def name_must_be_capitalized(cls, v):
        return v.title()

# Validation runs automatically on create/update
tools = tools_from_models([Product])
```

---

## Use Cases

### Instant Admin API

```python
# Define your models
class Customer(BaseModel):
    id: int
    name: str
    email: str
    plan: str = "free"

class Invoice(BaseModel):
    id: int
    customer_id: int
    amount: float
    status: str = "pending"

# Create admin agent
tools = tools_from_models([Customer, Invoice])
admin_agent = Agent(
    tools=tools,
    system="You are an admin assistant for managing customers and invoices."
)

# Natural language admin
admin_agent.run("Create a customer named John on the premium plan")
admin_agent.run("Create an invoice for customer 1 for $99")
admin_agent.run("Mark invoice 1 as paid")
```

### Data Entry Bot

```python
class Contact(BaseModel):
    id: int
    name: str
    phone: str
    email: str
    notes: str = ""

tools = tools_from_models([Contact])
agent = Agent(tools=tools)

# Process natural language data entry
agent.run("""
    Add these contacts:
    - John Smith, 555-1234, john@example.com
    - Jane Doe, 555-5678, jane@example.com
""")
```

---

## See Also

- [Agent](../core/agents.md) - Using tools with agents
- [Progress](progress.md) - Progress streaming from tools
