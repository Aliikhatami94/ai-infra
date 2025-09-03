# ai-infra

Infrastructure for efficient and scalable AI applications: clean LLM interfaces, composable graphs, and MCP client/server utilities. Batteries-included quickstarts help you ship fast.

- LLM: simple chat, agents with tools, streaming, retries, structured output, HITL hooks
- Graph: small-to-large workflows using LangGraph with typed state and tracing
- MCP: multi-server client, tool discovery, OpenMCP (OpenAPI-like) doc generation


## Install

- Python: 3.11 – 3.13
- Package manager: Poetry (recommended) or pip

Using Poetry (dev):

```bash
poetry install
poetry shell
```

Using pip (library use):

```bash
pip install ai-infra
```


## Configure providers (env)

Create a .env (or export in your shell) with any providers you plan to use.

```bash
# OpenAI
export OPENAI_API_KEY=...
# Anthropic
export ANTHROPIC_API_KEY=...
# Google Generative AI
export GOOGLE_API_KEY=...
# xAI
export XAI_API_KEY=...
```

Optional: MCP HTTP headers for servers you call through the client.

```bash
export MCP_AUTH_TOKEN=...
```


## Quickstarts

Below are tiny copy/paste snippets and how to run included examples.

### LLM: chat (sync)

```python
from ai_infra.llm import CoreLLM, Providers, Models

llm = CoreLLM()
resp = llm.chat(
    user_msg="One fun fact about the moon?",
    system="You are concise.",
    provider=Providers.openai,
    model_name=Models.openai.gpt_4o.value,
)
print(resp)
```

Run the included example (calls a main() function):

```bash
python -c "from ai_infra.llm.examples.02_llm_chat_basic import main; main()"
```

### LLM: agent (tools, sync)

```python
from ai_infra.llm import CoreAgent, Providers, Models

agent = CoreAgent()
resp = agent.run_agent(
    messages=[{"role": "user", "content": "Introduce yourself in one sentence."}],
    provider=Providers.openai,
    model_name=Models.openai.gpt_4o.value,
    model_kwargs={"temperature": 0.7},
)
print(getattr(resp, "content", resp))
```

Run the included example:

```bash
python -c "from ai_infra.llm.examples.01_agent_basic import main; main()"
```

### LLM: token streaming (async)

```python
import asyncio
from ai_infra.llm import CoreLLM, Providers, Models

async def demo():
    llm = CoreLLM()
    async for token, meta in llm.stream_tokens(
        "Stream one short paragraph about Mars.",
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    ):
        print(token, end="", flush=True)

asyncio.run(demo())
```

See more examples in src/ai_infra/llm/examples:
- 03_structured_output.py, 04_agent_stream.py, 05_tool_controls.py, 06_hitl.py, 07_retry.py, 08_agent_stream_tokens.py, 09_chat_stream.py


### Graph: minimal state machine

```python
from typing_extensions import TypedDict
from langgraph.graph import END
from ai_infra.graph.core import CoreGraph
from ai_infra.graph.models import Edge, ConditionalEdge

class MyState(TypedDict):
    value: int

def inc(s: MyState) -> MyState:
    s["value"] += 1
    return s

def mul(s: MyState) -> MyState:
    s["value"] *= 2
    return s

graph = CoreGraph(
    state_type=MyState,
    node_definitions=[inc, mul],
    edges=[
        Edge(start="inc", end="mul"),
        ConditionalEdge(
            start="mul", router_fn=lambda s: "inc" if s["value"] < 40 else END, targets=["inc", END]
        ),
    ],
)

print(graph.run({"value": 1}))
```

Run the included example:

```bash
python -c "from ai_infra.graph.examples.01_graph_basic import main; main()"
```

See also: 02_graph_stream_values.py


### MCP: multi-server client

```python
import asyncio
from ai_infra.mcp.client.core import CoreMCPClient

async def main():
    client = CoreMCPClient([
        {"transport": "streamable_http", "url": "http://127.0.0.1:8000/api/mcp", "headers": {"Authorization": "Bearer $MCP_AUTH_TOKEN"}},
        # {"transport": "stdio", "command": "./your-mcp-server", "args": []},
        # {"transport": "sse", "url": "http://127.0.0.1:8001/sse"},
    ])

    await client.discover()
    tools = await client.list_tools()
    print("Discovered tools:", tools)

    docs = await client.get_openmcp()  # or client.get_openmcp("your_server_name")
    print("OpenMCP doc keys:", list(docs.keys()))

asyncio.run(main())
```

Run the included example:

```bash
python -m ai_infra.mcp.examples.01_mcps
```


## Running all quickstarts

If you prefer a single runner command, add a tiny script like this locally:

```python
# quickstart.py
import sys

M = {
    "llm_agent_basic": "ai_infra.llm.examples.01_agent_basic:main",
    "llm_chat_basic": "ai_infra.llm.examples.02_llm_chat_basic:main",
    "graph_basic": "ai_infra.graph.examples.01_graph_basic:main",
    "mcp_discover": "ai_infra.mcp.examples.01_mcps:__main__",
}

if __name__ == "__main__":
    key = sys.argv[1]
    mod, _, func = M[key].partition(":")
    if func == "__main__":
        import runpy; runpy.run_module(mod, run_name="__main__")
    else:
        mod = __import__(mod, fromlist=[func])
        getattr(mod, func)()
```

Run:

```bash
python quickstart.py llm_chat_basic
python quickstart.py graph_basic
python quickstart.py llm_agent_basic
python quickstart.py mcp_discover
```

## MCP server config examples

Add entries like these to your Copilot MCP config (e.g., ~/.config/github-copilot/intellij/mcp.json):

```json
{
  "servers": {
    "stdio-publisher-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "--package=github:Aliikhatami94/ai-infra",
        "stdio-publisher-mcp"
      ]
    }
  }
}
```

Tip:
- If you want to pin a specific ref (branch, tag, commit), set AI_INFRA_REF in your environment before launching the IDE.

## Testing and quality

- Unit tests: pytest
  - `pytest -q`
- Lint: ruff
  - `ruff check src tests`
- Types: mypy
  - `mypy src`

Tip: add a test_examples.py that imports and runs the example main() functions to smoke test provider wiring without hitting network (use mocks).


## Project layout

- src/ai_infra/llm: core LLM and Agent APIs, providers, tools, and utils
- src/ai_infra/graph: CoreGraph wrapper, typed models, and utilities
- src/ai_infra/mcp: MCP client, examples, and server stubs
- tests: add your unit/integration tests here


## Notes and roadmap

- Providers: OpenAI, Anthropic, Google GenAI, xAI (via langchain providers)
- Features include structured output, retries, fallbacks, streaming, and tool call controls
- MCP doc generation (OpenMCP) is available via CoreMCPClient.get_openmcp()
- Nice-to-haves: add a simple example runner module; more test coverage around examples and MCP flows


## License

MIT
