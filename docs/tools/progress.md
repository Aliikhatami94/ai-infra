# Progress Streaming

> Stream progress updates from long-running tools.

## Quick Start

```python
from ai_infra import Agent, progress

@progress
def process_files(files: list[str]) -> str:
    """Process multiple files with progress updates."""
    for i, file in enumerate(files):
        yield {"progress": (i + 1) / len(files), "status": f"Processing {file}"}
        do_processing(file)
    return "All files processed"

agent = Agent(tools=[process_files])
```

---

## Overview

The `@progress` decorator enables tools to stream progress updates back to the caller. This is useful for:

- Long-running operations (file processing, API calls)
- Batch operations (processing lists of items)
- Multi-step workflows
- Providing real-time feedback to users

---

## Basic Usage

### Simple Progress

```python
from ai_infra import progress

@progress
def long_task() -> str:
    """A task that takes a while."""
    for i in range(10):
        yield {"progress": (i + 1) / 10}
        time.sleep(0.5)
    return "Done!"
```

### With Status Messages

```python
@progress
def process_data(data: list) -> dict:
    """Process data with status updates."""
    results = []

    for i, item in enumerate(data):
        yield {
            "progress": (i + 1) / len(data),
            "status": f"Processing item {i + 1} of {len(data)}",
            "current_item": item,
        }
        result = process_item(item)
        results.append(result)

    return {"processed": len(results), "results": results}
```

---

## Progress Events

The `@progress` decorator expects yielded dictionaries with progress info:

```python
@progress
def my_task() -> str:
    # Basic progress (0.0 to 1.0)
    yield {"progress": 0.5}

    # With status message
    yield {"progress": 0.7, "status": "Almost done..."}

    # With custom data
    yield {
        "progress": 0.9,
        "status": "Finalizing",
        "items_processed": 100,
        "errors": 0,
    }

    return "Complete"
```

---

## Consuming Progress

### With Agent

```python
from ai_infra import Agent, ProgressStream

@progress
def slow_search(query: str) -> str:
    for i in range(5):
        yield {"progress": (i + 1) / 5, "status": f"Searching page {i + 1}"}
        time.sleep(1)
    return "Search complete"

agent = Agent(tools=[slow_search])

# Run with progress callback
def on_progress(event):
    print(f"{event.progress * 100:.0f}% - {event.status}")

result = agent.run(
    "Search for information",
    on_progress=on_progress
)
```

### Direct Streaming

```python
from ai_infra import ProgressStream

@progress
def batch_process(items: list) -> list:
    for i, item in enumerate(items):
        yield {"progress": (i + 1) / len(items)}
        process(item)
    return items

# Get stream
stream = ProgressStream(batch_process, items=my_items)

# Iterate over progress events
for event in stream:
    print(f"Progress: {event.progress * 100:.0f}%")

# Get final result
result = stream.result
```

---

## ProgressEvent

```python
from ai_infra import ProgressEvent

# Event properties
event.progress    # float: 0.0 to 1.0
event.status      # str | None: Status message
event.data        # dict: Any additional data
event.timestamp   # datetime: When event was emitted
```

---

## Web Integration

### FastAPI + SSE

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ai_infra import Agent, progress
import json

app = FastAPI()

@progress
def analyze_data(data: str) -> dict:
    for i in range(10):
        yield {"progress": (i + 1) / 10, "step": f"Step {i + 1}"}
        time.sleep(0.5)
    return {"result": "Analysis complete"}

@app.post("/analyze")
async def analyze(data: str):
    agent = Agent(tools=[analyze_data])

    async def event_stream():
        async for event in agent.arun_stream(f"Analyze: {data}"):
            if event.type == "progress":
                yield f"data: {json.dumps(event.dict())}\n\n"
            elif event.type == "result":
                yield f"data: {json.dumps({'done': True, 'result': event.data})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### WebSocket

```python
from fastapi import WebSocket
from ai_infra import Agent, progress

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()

    agent = Agent(tools=[process_files])

    async for event in agent.arun_stream("Process the files"):
        await websocket.send_json({
            "progress": event.progress,
            "status": event.status,
        })

    await websocket.close()
```

---

## Nested Progress

```python
@progress
def outer_task(items: list) -> str:
    for i, item in enumerate(items):
        base_progress = i / len(items)

        # Sub-task progress within outer progress
        for j in range(3):
            sub_progress = (j + 1) / 3
            total_progress = base_progress + sub_progress / len(items)

            yield {
                "progress": total_progress,
                "status": f"Item {i + 1}, step {j + 1}",
            }

    return "All done"
```

---

## Error Handling

```python
@progress
def risky_task(data: list) -> dict:
    processed = []
    errors = []

    for i, item in enumerate(data):
        yield {"progress": (i + 1) / len(data)}

        try:
            result = process(item)
            processed.append(result)
        except Exception as e:
            errors.append({"item": item, "error": str(e)})

    return {"processed": len(processed), "errors": errors}
```

---

## Best Practices

1. **Meaningful progress** - Make progress reflect actual work done
2. **Informative status** - Tell users what's happening
3. **Reasonable frequency** - Don't flood with updates
4. **Handle errors** - Continue gracefully or report clearly
5. **Final result** - Always return something meaningful

---

## See Also

- [Agent](../core/agents.md) - Using tools with agents
- [Schema Tools](schema-tools.md) - Auto-generated CRUD tools
