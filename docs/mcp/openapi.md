# OpenAPI to MCP

> Convert OpenAPI specifications to MCP tools automatically.

## Quick Start

```python
from ai_infra import MCPServer

server = MCPServer()
server.add_openapi(
    "https://api.example.com/openapi.json",
    auth={"Authorization": "Bearer token123"}
)
server.run()
```

---

## Overview

ai-infra can automatically convert any OpenAPI 3.x specification into MCP tools. This means:
- Any REST API becomes usable by AI agents
- No manual tool definitions needed
- Authentication is handled automatically

---

## Adding OpenAPI Specs

### From URL

```python
from ai_infra import MCPServer

server = MCPServer()
server.add_openapi("https://api.example.com/openapi.json")
```

### From File

```python
server.add_openapi("./specs/api.yaml")
server.add_openapi("./specs/api.json")
```

### From Dict

```python
spec = {
    "openapi": "3.0.0",
    "info": {"title": "My API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "List users",
                "operationId": "listUsers"
            }
        }
    }
}

server.add_openapi(spec)
```

---

## Authentication

### Bearer Token

```python
server.add_openapi(
    "https://api.example.com/openapi.json",
    auth={"Authorization": "Bearer sk-xxx"}
)
```

### API Key

```python
server.add_openapi(
    "https://api.example.com/openapi.json",
    auth={"X-API-Key": "your-api-key"}
)
```

### Basic Auth

```python
import base64

credentials = base64.b64encode(b"user:password").decode()
server.add_openapi(
    url,
    auth={"Authorization": f"Basic {credentials}"}
)
```

### Custom Headers

```python
server.add_openapi(
    url,
    auth={
        "Authorization": "Bearer xxx",
        "X-Custom-Header": "value"
    }
)
```

---

## Operation Mapping

OpenAPI operations become MCP tools:

| OpenAPI | MCP Tool |
|---------|----------|
| `GET /users` | `get_users()` |
| `POST /users` | `create_user(body)` |
| `GET /users/{id}` | `get_user(id)` |
| `PUT /users/{id}` | `update_user(id, body)` |
| `DELETE /users/{id}` | `delete_user(id)` |

### Operation IDs

If `operationId` is defined, it becomes the tool name:

```yaml
paths:
  /users:
    get:
      operationId: listUsers  # Tool name: listUsers
```

---

## Filtering Operations

### Include Only Specific Operations

```python
server.add_openapi(
    url,
    include=["listUsers", "getUser", "createUser"]
)
```

### Exclude Operations

```python
server.add_openapi(
    url,
    exclude=["deleteUser", "updateUser"]
)
```

### By Tag

```python
server.add_openapi(
    url,
    tags=["users", "products"]  # Only include these tags
)
```

---

## With Agent

```python
from ai_infra import Agent, MCPClient

async def main():
    # Start server with OpenAPI tools
    server = MCPServer()
    server.add_openapi(
        "https://api.example.com/openapi.json",
        auth={"Authorization": "Bearer xxx"}
    )

    # In another process/thread:
    async with MCPClient("http://localhost:8080") as client:
        tools = await client.list_tools()

        agent = Agent(tools=tools)
        result = await agent.arun("List all users and find John")
```

---

## Base URL Override

```python
# Use different base URL than in spec
server.add_openapi(
    "./local-spec.json",
    base_url="https://production.api.com"
)
```

---

## Multiple APIs

```python
server = MCPServer()

# Add multiple APIs
server.add_openapi(
    "https://users-api.com/openapi.json",
    auth={"Authorization": "Bearer user-token"},
    prefix="users_"  # Tool prefix: users_get, users_create, etc.
)

server.add_openapi(
    "https://products-api.com/openapi.json",
    auth={"Authorization": "Bearer product-token"},
    prefix="products_"
)

server.run()
```

---

## Full Example

```python
from ai_infra import MCPServer

# Create server
server = MCPServer(
    name="api-gateway",
    version="1.0.0",
    description="MCP gateway for external APIs"
)

# Add GitHub API
server.add_openapi(
    "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
    auth={"Authorization": "Bearer ghp_xxx"},
    include=["repos/list-for-authenticated-user", "repos/get"],
    prefix="github_"
)

# Add OpenAI API
server.add_openapi(
    "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml",
    auth={"Authorization": "Bearer sk-xxx"},
    tags=["Chat"],
    prefix="openai_"
)

# Run
server.run(port=8080)
```

---

## Error Handling

```python
from ai_infra import MCPServer
from ai_infra.errors import OpenAPIError

try:
    server = MCPServer()
    server.add_openapi("invalid-url")
except OpenAPIError as e:
    print(f"Failed to load OpenAPI spec: {e}")
```

---

## See Also

- [MCP Server](server.md) - Create MCP servers
- [MCP Client](client.md) - Connect to MCP servers
- [Agent](../core/agents.md) - Use tools with agents
