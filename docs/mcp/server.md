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

## Security

**Security is automatic!** ai-infra auto-detects your deployment environment and configures appropriate security settings. Works with Railway, Render, Fly.io, Heroku, Vercel, and more.

### Auto-detection (Recommended)

```python
from ai_infra import mcp_from_functions

# Security auto-configured - works in dev and production
mcp = mcp_from_functions(name="my-mcp", functions=[my_tool])
```

### Disable Security (Dev Only)

```python
from ai_infra import MCPSecuritySettings, mcp_from_functions

mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(enable_security=False)
)
```

### Custom Domains

```python
from ai_infra import MCPSecuritySettings, mcp_from_functions

mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(domains=["api.example.com", "example.com"])
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

## Shell Tool via MCP

Expose the `run_shell` tool through MCP for remote command execution:

### Basic Shell Server

```python
from ai_infra import MCPServer
from ai_infra.llm.shell import run_shell, create_shell_tool

server = MCPServer(
    name="shell-server",
    version="1.0.0",
    description="Execute shell commands remotely"
)

# Add the shell tool
server.add_tool(run_shell)

if __name__ == "__main__":
    server.run(port=8080)
```

### Restricted Shell Server

For security, limit which commands can be executed:

```python
from ai_infra import MCPServer
from ai_infra.llm.shell import create_shell_tool

# Create restricted shell tool
safe_shell = create_shell_tool(
    allowed_commands=("ls", "cat", "grep", "find", "wc"),
    default_cwd="/safe/workspace",
    default_timeout=30.0,
)

server = MCPServer(name="safe-shell-server")
server.add_tool(safe_shell)
server.run()
```

### Shell with Session Persistence

For stateful shell sessions:

```python
from ai_infra import MCPServer
from ai_infra.llm.shell import ShellSession, create_shell_tool

# Create shared session
session = ShellSession(workspace_root="/project")

# Create tool bound to session
shell_tool = create_shell_tool(session=session)

server = MCPServer(name="stateful-shell-server")
server.add_tool(shell_tool)

# Session persists across client requests
server.run()
```

### Security Considerations

When exposing shell tools via MCP:

1. **Always restrict commands** — Use `allowed_commands` to whitelist safe commands
2. **Set workspace boundaries** — Use `default_cwd` to constrain operations
3. **Enable authentication** — Use MCP security settings for production
4. **Set timeouts** — Prevent long-running commands with `default_timeout`
5. **Monitor usage** — Enable logging to track command execution

```python
from ai_infra import MCPServer, MCPSecuritySettings
from ai_infra.llm.shell import create_shell_tool
import logging

# Enable shell logging
logging.getLogger("ai_infra.llm.shell").setLevel(logging.INFO)

# Create highly restricted shell
shell = create_shell_tool(
    allowed_commands=("pytest", "make test"),
    default_cwd="/project",
    default_timeout=120.0,
    dangerous_pattern_check=True,
)

server = MCPServer(
    name="ci-shell-server",
    security=MCPSecuritySettings(
        enable_security=True,
        require_auth=True,
    )
)
server.add_tool(shell)
server.run()
```

### Claude Desktop with Shell

To use shell tools with Claude Desktop:

```json
{
  "mcpServers": {
    "shell-tools": {
      "command": "python",
      "args": ["/path/to/shell_mcp_server.py"]
    }
  }
}
```

```python
# shell_mcp_server.py
from ai_infra import MCPServer
from ai_infra.llm.shell import create_shell_tool

# Safe shell for local use
shell = create_shell_tool(
    allowed_commands=("ls", "cat", "grep", "find", "tree"),
    default_cwd="~",
)

server = MCPServer(name="local-shell")
server.add_tool(shell)
server.run(transport="stdio")
```

---

## See Also

- [MCP Client](client.md) - Connect to MCP servers
- [OpenAPI to MCP](openapi.md) - Convert APIs to MCP
- [Agent](../core/agents.md) - Use MCP tools with agents
