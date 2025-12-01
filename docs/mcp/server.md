# MCP Server

> Expose tools as a Model Context Protocol (MCP) server.

## Quick Start

```python
from ai_infra import MCPServer, mcp_from_functions

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F"

server = MCPServer()
server.add_tools(mcp_from_functions([get_weather]))
server.run()
```

---

## Overview

MCPServer lets you expose Python functions as MCP-compliant tools that can be consumed by:
- ai-infra's MCPClient
- Claude Desktop
- Any MCP-compatible client

---

## Creating a Server

### Basic Server

```python
from ai_infra import MCPServer

server = MCPServer(
    name="my-server",
    version="1.0.0"
)
```

### With Description

```python
server = MCPServer(
    name="weather-server",
    version="1.0.0",
    description="Provides weather information"
)
```

---

## Adding Tools

### From Functions

```python
from ai_infra import MCPServer, mcp_from_functions

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

server = MCPServer()
server.add_tools(mcp_from_functions([add, multiply]))
```

### With Type Hints

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")

def search(params: SearchParams) -> list[str]:
    """Search for items."""
    return [f"Result for: {params.query}"]

server.add_tools(mcp_from_functions([search]))
```

### Manual Registration

```python
from ai_infra.mcp.server import Tool

server.add_tool(Tool(
    name="custom_tool",
    description="A custom tool",
    parameters={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input value"}
        },
        "required": ["input"]
    },
    handler=lambda input: f"Processed: {input}"
))
```

---

## Adding Resources

Expose files or data as resources:

```python
server.add_resource(
    uri="file://docs/readme.md",
    name="README",
    description="Project documentation",
    mime_type="text/markdown",
    content=open("README.md").read()
)
```

### Dynamic Resources

```python
def get_config():
    return json.dumps(load_config())

server.add_resource(
    uri="config://app",
    name="App Config",
    mime_type="application/json",
    handler=get_config
)
```

---

## Adding Prompts

Provide prompt templates:

```python
server.add_prompt(
    name="summarize",
    description="Summarize text",
    arguments=[
        {"name": "text", "description": "Text to summarize", "required": True}
    ],
    handler=lambda text: [
        {"role": "user", "content": f"Please summarize: {text}"}
    ]
)
```

---

## Running the Server

### HTTP/SSE (Default)

```python
server = MCPServer()
# ... add tools ...

# Run on default port 8080
server.run()

# Or specify port
server.run(port=9000)
```

### Stdio Transport

For use with Claude Desktop or stdio-based clients:

```python
server.run(transport="stdio")
```

---

## Full Example

```python
from ai_infra import MCPServer, mcp_from_functions
from pydantic import BaseModel, Field

# Define tools
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In reality, call a weather API
    return f"Weather in {city}: 72°F, sunny"

def search_web(query: str, limit: int = 5) -> list[str]:
    """Search the web for information."""
    # In reality, call a search API
    return [f"Result {i} for: {query}" for i in range(limit)]

class CalculateParams(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

def calculate(params: CalculateParams) -> float:
    """Evaluate a mathematical expression."""
    return eval(params.expression)

# Create server
server = MCPServer(
    name="assistant-tools",
    version="1.0.0",
    description="Tools for AI assistants"
)

# Add tools
server.add_tools(mcp_from_functions([
    get_weather,
    search_web,
    calculate,
]))

# Add a resource
server.add_resource(
    uri="info://about",
    name="About",
    mime_type="text/plain",
    content="This server provides weather, search, and calculation tools."
)

# Run
if __name__ == "__main__":
    server.run(port=8080)
```

---

## Claude Desktop Integration

To use with Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["/path/to/my_mcp_server.py"]
    }
  }
}
```

Run your server with stdio transport:

```python
server.run(transport="stdio")
```

---

## Error Handling

```python
def risky_tool(data: str) -> str:
    """A tool that might fail."""
    try:
        result = process(data)
        return result
    except Exception as e:
        # Errors are returned to the client
        raise ValueError(f"Processing failed: {e}")

server.add_tools(mcp_from_functions([risky_tool]))
```

---

## See Also

- [MCP Client](client.md) - Connect to MCP servers
- [OpenAPI to MCP](openapi.md) - Convert APIs to MCP
- [Agent](../core/agents.md) - Use MCP tools with agents
