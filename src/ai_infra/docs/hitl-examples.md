# HITL Examples

Human-in-the-Loop (HITL) integration examples for web applications.

## Overview

ai-infra supports two patterns for HITL in web apps:

1. **WebSocket** - Real-time bidirectional communication
2. **REST Polling** - Simple polling-based approach

## WebSocket Pattern

Best for:
- Real-time interaction
- Desktop and web applications
- Low latency requirements

### Example: FastAPI WebSocket

```python
from fastapi import FastAPI, WebSocket
from ai_infra.llm import Agent, ApprovalRequest, ApprovalResponse
from ai_infra.llm.session import memory

app = FastAPI()

# Store pending approvals
pending_approvals: Dict[str, asyncio.Future] = {}

async def websocket_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
    """Handle approval via WebSocket."""
    session_id = request.metadata.get("session_id")
    websocket = active_connections.get(session_id)

    # Send request to frontend
    await websocket.send_json({
        "type": "approval_required",
        "tool_name": request.tool_name,
        "tool_args": request.args,
    })

    # Wait for response
    future = asyncio.Future()
    pending_approvals[request.id] = future

    try:
        return await asyncio.wait_for(future, timeout=request.timeout)
    except asyncio.TimeoutError:
        return ApprovalResponse(approved=False, reason="Timeout")

# Create agent with WebSocket handler
agent = Agent(
    tools=[dangerous_tool],
    require_approval=True,
    approval_handler=websocket_approval_handler,
)
```

See [`11_fastapi_websocket_hitl.py`](../llm/examples/11_fastapi_websocket_hitl.py) for the complete example.

## REST Polling Pattern

Best for:
- Simple integration
- Mobile apps
- Environments without WebSocket support

### Example: FastAPI REST

```python
from fastapi import FastAPI, BackgroundTasks
from ai_infra.llm import Agent
from ai_infra.llm.session import memory

app = FastAPI()

# Create agent with pause/resume
agent = Agent(
    tools=[dangerous_tool],
    session=memory(),
    pause_before=["dangerous_tool"],  # Pause before these tools
)

@app.post("/api/tasks")
async def create_task(request: CreateTaskRequest, bg: BackgroundTasks):
    """Create a task - runs in background."""
    task_id = str(uuid.uuid4())
    bg.add_task(run_agent_task, task_id, request.message)
    return {"task_id": task_id}

@app.get("/api/approvals")
async def get_pending_approvals():
    """Poll for pending approvals."""
    return [a for a in approvals.values() if a.status == "pending"]

@app.post("/api/approvals/{approval_id}")
async def submit_approval(approval_id: str, decision: ApprovalDecision):
    """Submit approval decision."""
    # Resume the paused agent
    await agent.aresume(
        session_id=approval_id,
        approved=decision.approved,
    )
```

See [`12_rest_api_polling_hitl.py`](../llm/examples/12_rest_api_polling_hitl.py) for the complete example.

## Choosing a Pattern

| Aspect | WebSocket | REST Polling |
|--------|-----------|--------------|
| Latency | Low | Higher |
| Complexity | Higher | Lower |
| Scalability | Good | Better |
| Infrastructure | WebSocket support needed | Standard HTTP |
| Mobile Support | Limited | Full |
| Offline Support | No | Partial |

## Console HITL

For scripts and CLIs, use the built-in console handler:

```python
from ai_infra.llm import Agent

agent = Agent(
    tools=[dangerous_tool],
    require_approval=True,  # Uses console prompts
)

# Will prompt: "Approve dangerous_tool(args)? [y/n]"
result = agent.run("Do something dangerous")
```

## Custom Approval Rules

```python
from ai_infra.llm import Agent, ApprovalRule

# Only require approval for specific tools
agent = Agent(
    tools=[safe_tool, dangerous_tool],
    require_approval=["dangerous_tool"],  # Only this one
)

# Or use dynamic rules
def should_approve(tool_name: str, args: dict) -> bool:
    """Require approval for high-value payments."""
    if tool_name == "make_payment":
        return args.get("amount", 0) > 100
    return False

agent = Agent(
    tools=[make_payment],
    require_approval=should_approve,
)
```

## Session Persistence

For production, use PostgreSQL for session storage:

```python
from ai_infra.llm import Agent
from ai_infra.llm.session import postgres

agent = Agent(
    tools=[...],
    session=postgres("postgresql://localhost/mydb"),
    pause_before=["dangerous_tool"],
)

# Sessions persist across restarts
result = await agent.arun("Delete file", session_id="task-123")

# Later (even after restart)
if result.paused:
    await agent.aresume(session_id="task-123", approved=True)
```
