# Quickstart

> From zero to AI chat in 5 minutes.

## 1. Install

```bash
pip install ai-infra
```

## 2. Set Your API Key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

## 3. Your First Chat

```python
from ai_infra import LLM

llm = LLM()
response = llm.chat("What is 2 + 2?")
print(response)  # "4"
```

Done. You have a working AI chat.

---

## 4. Add a Tool

```python
from ai_infra import Agent

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real weather API
    return f"Weather in {city}: Sunny, 72F"

agent = Agent(tools=[get_weather])
result = agent.run("What's the weather in San Francisco?")
print(result)
```

The agent automatically calls `get_weather("San Francisco")` and responds.

---

## 5. Stream Responses

```python
from ai_infra import LLM

llm = LLM()
for token in llm.stream_tokens("Tell me a joke"):
    print(token, end="", flush=True)
```

---

## 6. Get Structured Output

```python
from pydantic import BaseModel
from ai_infra import LLM

class Answer(BaseModel):
    value: int
    explanation: str

llm = LLM()
result = llm.chat("What is 2 + 2?", response_model=Answer)
print(result.value)        # 4
print(result.explanation)  # "Two plus two equals four"
```

---

## Next Steps

- [Getting Started](getting-started.md) - Full guide with all features
- [Agents](tools/agents.md) - Build complex tool-using agents
- [Streaming](streaming.md) - Real-time token streaming
- [MCP Servers](mcp/server.md) - Expose tools as MCP servers
