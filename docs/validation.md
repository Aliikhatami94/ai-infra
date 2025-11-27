# Input/Output Validation

ai-infra provides runtime validation for LLM parameters and outputs.

## Input Validation

### Validate Provider

```python
from ai_infra import validate_provider

validate_provider("openai")  # OK
validate_provider("invalid")  # Raises ValidationError
```

### Validate Temperature

```python
from ai_infra import validate_temperature

validate_temperature(0.7)  # OK
validate_temperature(5.0)  # Raises ValidationError

# Provider-specific validation
validate_temperature(1.5, provider="anthropic")  # Raises (max is 1.0)
```

### Validate All LLM Parameters

```python
from ai_infra import validate_llm_params

validate_llm_params(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)
```

## Output Validation

### Validate Against Pydantic Schema

```python
from pydantic import BaseModel
from ai_infra import validate_output

class Answer(BaseModel):
    value: int
    explanation: str

# Validate LLM response
result = validate_output(response, Answer)
```

### Validate JSON Output

```python
from ai_infra import validate_json_output

# Parse and validate JSON from LLM response
data = validate_json_output(response, expected_keys=["name", "age"])
```

## Decorators

### Validate Function Inputs

```python
from ai_infra import validated_inputs

@validated_inputs
def call_llm(
    provider: str,
    temperature: float = 0.7,
) -> str:
    # Provider and temperature are automatically validated
    ...
```

### Validate Function Outputs

```python
from ai_infra import validated_output

@validated_output(Answer)
def get_answer(question: str) -> Answer:
    # Output is validated against Answer schema
    ...
```

## Supported Providers

```python
from ai_infra.validation import SUPPORTED_PROVIDERS

print(SUPPORTED_PROVIDERS)
# ['openai', 'anthropic', 'google_genai', 'xai', 'ollama',
#  'azure_openai', 'bedrock', 'together', 'groq', 'deepseek']
```

## Temperature Ranges

Provider-specific temperature ranges:

| Provider | Min | Max |
|----------|-----|-----|
| openai | 0.0 | 2.0 |
| anthropic | 0.0 | 1.0 |
| google_genai | 0.0 | 2.0 |
| xai | 0.0 | 2.0 |
| bedrock | 0.0 | 1.0 |

## Error Handling

```python
from ai_infra import ValidationError

try:
    validate_provider("unknown")
except ValidationError as e:
    print(e.field)      # "provider"
    print(e.expected)   # "one of ['openai', 'anthropic', ...]"
    print(e.details)    # {"provider": "unknown", "supported": [...]}
```
