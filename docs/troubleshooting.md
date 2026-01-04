# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with ai-infra.

## Quick Diagnostics

Before diving into specific issues, run this diagnostic:

```python
from ai_infra import LLM
from ai_infra.llm import PROVIDER_ENV_VARS

# Check which providers have API keys configured
for provider, env_var in PROVIDER_ENV_VARS.items():
    import os
    has_key = bool(os.getenv(env_var))
    print(f"{provider}: {'configured' if has_key else 'missing'} ({env_var})")
```

---

## API Key Not Found Errors

### Symptoms

```
AuthenticationError: No API key found for provider 'openai'
ProviderNotFoundError: Could not initialize provider - missing credentials
```

### Causes

1. Environment variable not set
2. Wrong environment variable name
3. Environment not loaded in your shell

### Solutions

**1. Set the correct environment variable:**

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Groq | `GROQ_API_KEY` |
| Together | `TOGETHER_API_KEY` |

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc)
export OPENAI_API_KEY="sk-..."
```

**2. Use a .env file with python-dotenv:**

```python
# At the top of your script
from dotenv import load_dotenv
load_dotenv()

from ai_infra import LLM
llm = LLM()  # Now picks up keys from .env
```

**3. Pass the API key directly (not recommended for production):**

```python
from ai_infra import LLM

llm = LLM(provider="openai", api_key="sk-...")
```

**4. Use temporary_api_key context manager for testing:**

```python
from ai_infra.llm import temporary_api_key

with temporary_api_key("openai", "sk-test-key"):
    llm = LLM(provider="openai")
    response = llm.chat("Hello")
```

---

## Rate Limiting and Retry Behavior

### Symptoms

```
RateLimitError: Rate limit exceeded. Retry after 60 seconds.
429 Too Many Requests
```

### Causes

1. Exceeding provider's requests-per-minute (RPM) limit
2. Exceeding tokens-per-minute (TPM) limit
3. Concurrent requests overwhelming the API

### Solutions

**1. Enable automatic retries (default behavior):**

```python
from ai_infra import LLM

# ai-infra automatically retries with exponential backoff
llm = LLM(
    max_retries=3,           # Number of retries (default: 3)
    retry_delay=1.0,         # Initial delay in seconds
    retry_multiplier=2.0,    # Exponential backoff multiplier
)
```

**2. Implement request batching:**

```python
import asyncio
from ai_infra import LLM

llm = LLM()

async def process_with_rate_limit(prompts: list[str], rpm_limit: int = 60):
    """Process prompts with rate limiting."""
    delay = 60.0 / rpm_limit  # Seconds between requests

    results = []
    for prompt in prompts:
        result = await llm.achat(prompt)
        results.append(result)
        await asyncio.sleep(delay)

    return results
```

**3. Use a semaphore for concurrent requests:**

```python
import asyncio
from ai_infra import LLM

llm = LLM()
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

async def rate_limited_chat(prompt: str) -> str:
    async with semaphore:
        return await llm.achat(prompt)
```

**4. Check your tier limits:**

| Provider | Free Tier | Paid Tier |
|----------|-----------|-----------|
| OpenAI | 3 RPM, 40K TPM | 500+ RPM |
| Anthropic | 5 RPM | 1000+ RPM |
| Google | 60 RPM | Higher |

---

## Streaming Connection Issues

### Symptoms

```
StreamingError: Connection closed unexpectedly
SSEError: Failed to parse server-sent event
asyncio.TimeoutError: Streaming response timed out
```

### Causes

1. Network instability
2. Server-side timeout
3. Proxy/firewall interference
4. Very long responses

### Solutions

**1. Add timeout handling:**

```python
import asyncio
from ai_infra import LLM

llm = LLM()

async def stream_with_timeout(prompt: str, timeout: float = 120.0):
    """Stream response with timeout protection."""
    try:
        async with asyncio.timeout(timeout):
            async for chunk in llm.astream(prompt):
                print(chunk, end="", flush=True)
    except asyncio.TimeoutError:
        print("\n[Streaming timed out - partial response above]")
```

**2. Implement reconnection logic:**

```python
async def resilient_stream(prompt: str, max_retries: int = 3):
    """Stream with automatic reconnection."""
    collected = []

    for attempt in range(max_retries):
        try:
            async for chunk in llm.astream(prompt):
                collected.append(chunk)
                yield chunk
            return  # Success
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Log and retry
            print(f"Stream interrupted, retrying... ({attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
```

**3. Fall back to non-streaming:**

```python
async def chat_with_fallback(prompt: str) -> str:
    """Try streaming first, fall back to regular chat."""
    try:
        chunks = []
        async for chunk in llm.astream(prompt):
            chunks.append(chunk)
        return "".join(chunks)
    except StreamingError:
        # Fall back to non-streaming
        return await llm.achat(prompt)
```

---

## MCP Server Connection Failures

### Symptoms

```
MCPConnectionError: Failed to connect to MCP server at stdio://...
MCPTimeoutError: MCP server did not respond within 30 seconds
MCPError: Server process exited unexpectedly
```

### Causes

1. MCP server binary not found or not executable
2. Server crashed during startup
3. Incompatible protocol version
4. Missing dependencies in MCP server

### Solutions

**1. Verify the server command exists:**

```python
import shutil
import subprocess

# Check if command exists
command = "uvx"  # or your MCP server command
if shutil.which(command) is None:
    print(f"Command '{command}' not found in PATH")
else:
    # Test running it
    result = subprocess.run([command, "--version"], capture_output=True)
    print(f"Version: {result.stdout.decode()}")
```

**2. Test MCP server independently:**

```bash
# Run the server directly to see error output
uvx mcp-server-filesystem --directory /tmp

# Or with npx
npx -y @anthropic/mcp-server-filesystem /tmp
```

**3. Add proper error handling for MCP:**

```python
from ai_infra import Agent
from ai_infra.mcp import MCPServer
from ai_infra.errors import MCPConnectionError, MCPTimeoutError

async def create_agent_with_mcp():
    """Create agent with MCP, handling failures gracefully."""
    try:
        async with MCPServer(
            name="filesystem",
            command="uvx",
            args=["mcp-server-filesystem", "--directory", "/tmp"],
            timeout=30.0,  # Connection timeout
        ) as server:
            agent = Agent(
                name="file_assistant",
                mcp_servers=[server],
            )
            return agent
    except MCPConnectionError as e:
        print(f"Could not connect to MCP server: {e}")
        # Return agent without MCP tools
        return Agent(name="file_assistant")
    except MCPTimeoutError:
        print("MCP server took too long to start")
        return Agent(name="file_assistant")
```

**4. Check MCP server logs:**

```python
from ai_infra.mcp import MCPServer

# Enable debug logging for MCP
import logging
logging.getLogger("ai_infra.mcp").setLevel(logging.DEBUG)

async with MCPServer(
    name="debug_server",
    command="uvx",
    args=["mcp-server-filesystem", "/tmp"],
) as server:
    # Logs will show all MCP protocol messages
    tools = await server.list_tools()
```

---

## Token Limit Exceeded Errors

### Symptoms

```
ContextLengthExceededError: This model's maximum context length is 128000 tokens.
You requested 145000 tokens.
TokenLimitError: Response would exceed max_tokens limit
```

### Causes

1. Conversation history grew too long
2. System prompt + user message too large
3. Requested output too long

### Solutions

**1. Use fit_context to automatically trim:**

```python
from ai_infra.llm import fit_context, LLM

llm = LLM()

# Automatically trim messages to fit
result = fit_context(
    messages=conversation_history,
    max_tokens=100000,  # Leave room for response
    strategy="sliding_window",  # Keep recent messages
)

response = await llm.achat(result.messages)
```

**2. Summarize old messages:**

```python
from ai_infra import LLM

llm = LLM()

async def summarize_and_continue(messages: list[dict], new_prompt: str):
    """Summarize old context when it gets too long."""
    from ai_infra.llm import count_tokens_approximate

    total_tokens = count_tokens_approximate(str(messages))

    if total_tokens > 50000:
        # Summarize older messages
        old_messages = messages[:-10]  # Keep last 10
        summary = await llm.achat(
            f"Summarize this conversation concisely:\n{old_messages}"
        )

        messages = [
            {"role": "system", "content": f"Previous context: {summary}"},
            *messages[-10:],
        ]

    messages.append({"role": "user", "content": new_prompt})
    return await llm.achat(messages)
```

**3. Use a model with larger context:**

```python
from ai_infra import LLM

# Models with large context windows
llm = LLM(model="gpt-4o")           # 128K context
llm = LLM(model="claude-3-5-sonnet-20241022")  # 200K context
llm = LLM(model="gemini-1.5-pro")   # 2M context (!)
```

**4. Implement chunking for long documents:**

```python
async def process_long_document(document: str, chunk_size: int = 50000):
    """Process a long document in chunks."""
    from ai_infra.llm import count_tokens_approximate

    # Split into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in document.split("\n\n"):
        para_tokens = count_tokens_approximate(paragraph)
        if current_tokens + para_tokens > chunk_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_tokens = para_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        result = await llm.achat(f"Analyze this section ({i+1}/{len(chunks)}):\n{chunk}")
        results.append(result)

    return results
```

---

## Common Environment Issues

### Issue: "No module named 'ai_infra'"

```bash
# Make sure you installed it
pip install ai-infra

# Or in the correct virtual environment
poetry add ai-infra
```

### Issue: SSL Certificate Errors

```python
# If behind a corporate proxy
import os
os.environ["REQUESTS_CA_BUNDLE"] = "/path/to/corporate-ca.crt"

# Or disable verification (NOT recommended for production)
import httpx
client = httpx.Client(verify=False)
```

### Issue: Async Event Loop Errors

```python
# "RuntimeError: This event loop is already running"
# Use nest_asyncio in Jupyter notebooks
import nest_asyncio
nest_asyncio.apply()

# Then use async functions normally
result = await llm.achat("Hello")
```

---

## Getting Help

If you're still stuck:

1. **Check the logs**: Enable debug logging with `logging.getLogger("ai_infra").setLevel(logging.DEBUG)`
2. **Search issues**: Check [GitHub Issues](https://github.com/nfraxlab/ai-infra/issues)
3. **Ask for help**: Open a new issue with:
   - ai-infra version (`pip show ai-infra`)
   - Python version
   - Full error traceback
   - Minimal reproduction code
