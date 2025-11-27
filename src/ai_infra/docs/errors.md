# Error Handling

ai-infra provides a unified error hierarchy for consistent error handling across all modules.

## Error Hierarchy

```
AIInfraError (base)
├── LLMError
│   ├── ProviderError
│   │   ├── RateLimitError
│   │   ├── AuthenticationError
│   │   ├── ModelNotFoundError
│   │   ├── ContextLengthError
│   │   └── ContentFilterError
│   └── OutputValidationError
├── MCPError
│   ├── MCPServerError
│   ├── MCPToolError
│   ├── MCPConnectionError
│   └── MCPTimeoutError
├── OpenAPIError
│   ├── OpenAPIParseError
│   ├── OpenAPINetworkError
│   └── OpenAPIValidationError
├── GraphError
│   ├── GraphExecutionError
│   └── GraphValidationError
├── ToolError
│   ├── ToolExecutionError
│   ├── ToolTimeoutError
│   └── ToolValidationError
├── ValidationError
└── ConfigurationError
```

## Usage

### Catching All ai-infra Errors

```python
from ai_infra import AIInfraError

try:
    result = await llm.achat("Hello")
except AIInfraError as e:
    print(f"Error: {e.message}")
    if e.hint:
        print(f"Hint: {e.hint}")
    if e.docs_url:
        print(f"Docs: {e.docs_url}")
```

### Catching Specific Errors

```python
from ai_infra import RateLimitError, AuthenticationError

try:
    result = await llm.achat("Hello")
except RateLimitError as e:
    if e.retry_after:
        await asyncio.sleep(e.retry_after)
        # Retry...
except AuthenticationError as e:
    print(f"Check your API key: {e.hint}")
```

## Error Attributes

All errors have these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | str | Human-readable error description |
| `details` | dict | Additional context (provider, model, etc.) |
| `hint` | str | Suggested fix or action |
| `docs_url` | str | Link to relevant documentation |

### ProviderError-specific

| Attribute | Type | Description |
|-----------|------|-------------|
| `provider` | str | Provider name (e.g., "openai") |
| `model` | str | Model name (e.g., "gpt-4o") |
| `status_code` | int | HTTP status code |
| `error_type` | str | Error type from provider |

### RateLimitError-specific

| Attribute | Type | Description |
|-----------|------|-------------|
| `retry_after` | float | Seconds to wait before retry |

### ContextLengthError-specific

| Attribute | Type | Description |
|-----------|------|-------------|
| `max_tokens` | int | Maximum allowed tokens |
| `requested_tokens` | int | Tokens requested |

## Error Translation

ai-infra automatically translates provider SDK errors into our error classes:

```python
from ai_infra.llm import LLM

llm = LLM()

try:
    # This will translate OpenAI errors to ai-infra errors
    result = llm.chat("Hello", provider="openai")
except RateLimitError as e:
    # Clear message with retry info
    print(e.message)  # "Rate limit exceeded..."
    print(e.hint)     # "Retry after 30 seconds"
```

### Manual Translation

```python
from ai_infra.llm import translate_provider_error

try:
    response = openai_client.chat.completions.create(...)
except Exception as e:
    # Translate to ai-infra error
    raise translate_provider_error(e, provider="openai", model="gpt-4o")
```

## Provider-Specific Kwargs

Check which kwargs are supported:

```python
from ai_infra.llm import get_supported_kwargs, validate_kwargs

# Get all supported kwargs for a provider
openai_kwargs = get_supported_kwargs("openai")
print(openai_kwargs["frequency_penalty"])

# Validate kwargs
warnings = validate_kwargs("anthropic", {"frequency_penalty": 0.5})
# Returns: ["'frequency_penalty' is not supported by anthropic..."]
```
