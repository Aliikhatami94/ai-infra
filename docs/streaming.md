# Streaming Guide

Typed streaming for agents using `Agent.astream()`.

## StreamEvent reference

`StreamEvent` fields (None when not applicable):
- `type`: thinking | token | tool_start | tool_end | done | error
- `content`: token text
- `tool` / `tool_id`: tool name and call ID
- `arguments`: tool args (visibility detailed+)
- `preview`: tool result preview (visibility=debug)
- `latency_ms`: tool latency
- `model`: model name (thinking)
- `tools_called`: total tools (done)
- `error`: error message
- `timestamp`: event timestamp

Serialize with `event.to_dict()`.

## StreamConfig reference

`StreamConfig` controls visibility and tool handling:
- `visibility`: minimal | standard | detailed | debug (default: standard)
- `include_thinking`: emit initial thinking event
- `include_tool_events`: emit tool_start/tool_end
- `tool_result_preview_length`: max preview length (debug)
- `deduplicate_tool_starts`: avoid duplicate starts per tool call

Pass via `agent.astream(..., stream_config=StreamConfig(visibility="detailed"))`.

## BYOK helper (temporary keys)

Use user-provided API keys for a single request:

```python
from ai_infra import Agent, atemporary_api_key

async with atemporary_api_key("openai", user_key):
    async for event in agent.astream(prompt):
        yield event.to_dict()
```

## MCP tool loader (cached)

Load and cache MCP tools once, with optional force refresh:

```python
from ai_infra import Agent, load_mcp_tools_cached

tools = await load_mcp_tools_cached("http://localhost:8000/mcp")
agent = Agent(tools=tools)
```

## Examples

### Basic streaming

```python
async for event in agent.astream("What is the refund policy?"):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Visibility control

```python
async for event in agent.astream("Search docs", visibility="detailed"):
    if event.type == "tool_start":
        print(event.arguments)
```

### With LangGraph config

```python
config = {
    "configurable": {"thread_id": "user-123"},
    "tags": ["production"],
}
async for event in agent.astream("Continue our conversation", config=config):
    ...
```

### FastAPI SSE endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
agent = Agent(tools=[search_docs])

@app.post("/chat")
async def chat(message: str, provider: str, api_key: str):
    async def generate():
        async with atemporary_api_key(provider, api_key):
            async for event in agent.astream(message):
                yield f"data: {event.to_dict()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### WebSocket endpoint

```python
from fastapi import WebSocket

@router.websocket("/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    async for event in agent.astream("Hello"):
        await ws.send_json(event.to_dict())
```

### Cached MCP tools + BYOK

```python
MCP_URL = "http://localhost:8000/mcp"

async def stream_with_mcp(message: str, api_key: str):
    tools = await load_mcp_tools_cached(MCP_URL)
    agent = Agent(tools=tools)
    async with atemporary_api_key("openai", api_key):
        async for event in agent.astream(message, visibility="debug"):
            yield event
```
