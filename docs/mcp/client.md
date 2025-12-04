# MCP Client

> Connect to Model Context Protocol (MCP) servers for tool discovery and execution.

## Quick Start

```python
from ai_infra import MCPClient

async with MCPClient("http://localhost:8080") as client:
    tools = await client.list_tools()
    result = await client.call_tool("get_weather", {"city": "NYC"})
```

---

## Overview

MCPClient connects to MCP-compliant servers to:
- Discover available tools
- Execute tool calls
- Integrate external capabilities into your agents

---

## Connection

### HTTP/SSE Transport

```python
from ai_infra import MCPClient

# Connect to HTTP server
async with MCPClient("http://localhost:8080") as client:
    tools = await client.list_tools()
```

### Stdio Transport

```python
# Connect to stdio-based MCP server
async with MCPClient(
    transport="stdio",
    command=["python", "mcp_server.py"]
) as client:
    tools = await client.list_tools()
```

### With Authentication

```python
async with MCPClient(
    "http://localhost:8080",
    headers={"Authorization": "Bearer token123"}
) as client:
    tools = await client.list_tools()
```

---

## Tool Discovery

### List All Tools

```python
async with MCPClient(url) as client:
    tools = await client.list_tools()

    for tool in tools:
        print(f"Name: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Parameters: {tool.parameters}")
```

### Get Tool Details

```python
tool = await client.get_tool("search")
print(tool.parameters.schema)
```

---

## Tool Execution

### Call a Tool

```python
async with MCPClient(url) as client:
    result = await client.call_tool(
        "get_weather",
        {"city": "San Francisco", "units": "fahrenheit"}
    )
    print(result)
```

### With Timeout

```python
result = await client.call_tool(
    "slow_operation",
    {"data": "..."},
    timeout=30.0
)
```

---

## With Agent

Use MCP tools with ai-infra agents:

```python
from ai_infra import Agent, MCPClient

async def main():
    async with MCPClient("http://localhost:8080") as client:
        # Get tools from MCP server
        mcp_tools = await client.list_tools()

        # Create agent with MCP tools
        agent = Agent(tools=mcp_tools)

        result = await agent.arun("Use the MCP tools to help me")
        print(result)
```

### Multiple MCP Servers

```python
async with MCPClient("http://server1:8080") as client1, \
           MCPClient("http://server2:8080") as client2:

    tools1 = await client1.list_tools()
    tools2 = await client2.list_tools()

    agent = Agent(tools=[*tools1, *tools2])
    result = await agent.arun("Use all available tools")
```

---

## Resources

MCP servers can also expose resources (documents, data):

### List Resources

```python
async with MCPClient(url) as client:
    resources = await client.list_resources()

    for resource in resources:
        print(f"URI: {resource.uri}")
        print(f"Name: {resource.name}")
        print(f"Type: {resource.mime_type}")
```

### Read Resource

```python
content = await client.read_resource("file://docs/readme.md")
print(content)
```

---

## Prompts

MCP servers can provide prompt templates:

### List Prompts

```python
async with MCPClient(url) as client:
    prompts = await client.list_prompts()

    for prompt in prompts:
        print(f"Name: {prompt.name}")
        print(f"Description: {prompt.description}")
```

### Get Prompt

```python
prompt = await client.get_prompt(
    "summarize",
    {"text": "Long text to summarize..."}
)
print(prompt.messages)
```

---

## Callbacks

Track progress and log tool calls using callbacks:

### Progress Callbacks

```python
from ai_infra.mcp import MCPClient, Callbacks, CallbackContext

def on_tool_start(ctx: CallbackContext):
    print(f"Starting: {ctx.tool_name}")

def on_tool_end(ctx: CallbackContext):
    print(f"Completed: {ctx.tool_name} in {ctx.duration_ms}ms")

callbacks = Callbacks(
    on_tool_start=on_tool_start,
    on_tool_end=on_tool_end,
)

async with MCPClient(config, callbacks=callbacks) as client:
    await client.call_tool("server", "my_tool", {"arg": "value"})
```

### Built-in Logging Callback

```python
from ai_infra.mcp import LoggingCallback

callbacks = Callbacks(on_tool_start=LoggingCallback())
```

### Progress Callback

```python
from ai_infra.mcp import ProgressCallback

# Get progress updates during long-running tools
callbacks = Callbacks(on_progress=ProgressCallback())
```

---

## Interceptors

Intercept and modify tool calls for caching, retries, rate limiting:

### Caching Interceptor

```python
from ai_infra.mcp import MCPClient, CachingInterceptor

# Cache tool results for 5 minutes
caching = CachingInterceptor(ttl_seconds=300)

async with MCPClient(config, interceptors=[caching]) as client:
    # First call hits the server
    result1 = await client.call_tool("server", "expensive_tool", {"id": 1})

    # Second call returns cached result
    result2 = await client.call_tool("server", "expensive_tool", {"id": 1})
```

### Retry Interceptor

```python
from ai_infra.mcp import RetryInterceptor

# Retry failed calls up to 3 times with exponential backoff
retry = RetryInterceptor(max_retries=3, backoff_factor=2.0)

async with MCPClient(config, interceptors=[retry]) as client:
    result = await client.call_tool("server", "flaky_tool", {})
```

### Rate Limit Interceptor

```python
from ai_infra.mcp import RateLimitInterceptor

# Limit to 10 calls per second
rate_limit = RateLimitInterceptor(max_calls=10, period_seconds=1.0)

async with MCPClient(config, interceptors=[rate_limit]) as client:
    # Calls will be throttled to respect rate limit
    for i in range(100):
        await client.call_tool("server", "tool", {"i": i})
```

### Custom Interceptor

```python
from ai_infra.mcp import ToolCallInterceptor, MCPToolCallRequest

class LoggingInterceptor(ToolCallInterceptor):
    async def intercept(self, request: MCPToolCallRequest, call_next):
        print(f"Calling: {request.tool_name}")
        result = await call_next(request)
        print(f"Result: {result}")
        return result

async with MCPClient(config, interceptors=[LoggingInterceptor()]) as client:
    await client.call_tool("server", "tool", {})
```

### Chaining Interceptors

```python
# Interceptors are applied in order
interceptors = [
    RateLimitInterceptor(max_calls=10, period_seconds=1.0),
    RetryInterceptor(max_retries=3),
    CachingInterceptor(ttl_seconds=300),
]

async with MCPClient(config, interceptors=interceptors) as client:
    # Rate limit → Retry → Cache → actual call
    result = await client.call_tool("server", "tool", {})
```

---

## Error Handling

```python
from ai_infra import MCPClient
from ai_infra.errors import MCPError

try:
    async with MCPClient(url) as client:
        result = await client.call_tool("unknown_tool", {})
except MCPError as e:
    print(f"MCP error: {e}")
```

---

## Configuration

```python
client = MCPClient(
    url="http://localhost:8080",
    headers={"Authorization": "Bearer xxx"},
    timeout=30.0,
    max_retries=3,
)
```

---

## See Also

- [MCP Server](server.md) - Create MCP servers
- [OpenAPI to MCP](openapi.md) - Convert OpenAPI to MCP
- [Agent](../core/agents.md) - Use MCP tools with agents
