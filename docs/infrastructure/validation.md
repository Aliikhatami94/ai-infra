# Input Validation

> Validate inputs and outputs for AI operations.

## Quick Start

```python
from ai_infra import validate_prompt, validate_response

# Validate before sending
validated_prompt = validate_prompt(
    prompt=user_input,
    max_length=4000,
    block_patterns=["password", "api_key"],
)

# Validate response
validated_response = validate_response(
    response=llm_response,
    expected_format="json",
)
```

---

## Overview

Validation protects your AI applications from:
- Prompt injection attacks
- PII leakage
- Malformed outputs
- Cost overruns from long inputs
- Content policy violations

---

## Prompt Validation

### Length Validation

```python
from ai_infra import validate_prompt, PromptTooLongError

try:
    validated = validate_prompt(
        prompt=user_input,
        max_length=4000,  # Characters
        max_tokens=1000,  # Tokens (if tokenizer available)
    )
except PromptTooLongError as e:
    print(f"Prompt too long: {e.length} chars (max: {e.max_length})")
```

### Pattern Blocking

```python
from ai_infra import validate_prompt, BlockedPatternError

try:
    validated = validate_prompt(
        prompt=user_input,
        block_patterns=[
            r"password\s*[:=]",
            r"api[_-]?key",
            r"secret",
            r"ssn:\s*\d{3}-\d{2}-\d{4}",
        ],
        block_regex=True,
    )
except BlockedPatternError as e:
    print(f"Blocked pattern found: {e.pattern}")
```

### PII Detection

```python
from ai_infra import validate_prompt, PIIDetectedError

try:
    validated = validate_prompt(
        prompt=user_input,
        detect_pii=True,
        pii_types=["email", "phone", "ssn", "credit_card"],
    )
except PIIDetectedError as e:
    print(f"PII detected: {e.pii_type} at position {e.position}")
```

### Prompt Injection Detection

```python
from ai_infra import validate_prompt, PromptInjectionError

try:
    validated = validate_prompt(
        prompt=user_input,
        detect_injection=True,
    )
except PromptInjectionError as e:
    print(f"Potential prompt injection: {e.reason}")
```

---

## Response Validation

### Format Validation

```python
from ai_infra import validate_response

# Validate JSON response
validated = validate_response(
    response=llm_response,
    expected_format="json",
)

# Access parsed JSON
data = validated.parsed
```

### Schema Validation

```python
from pydantic import BaseModel
from ai_infra import validate_response

class User(BaseModel):
    name: str
    age: int
    email: str

validated = validate_response(
    response=llm_response,
    schema=User,
)

user = validated.parsed  # User instance
```

### Content Validation

```python
from ai_infra import validate_response

validated = validate_response(
    response=llm_response,
    max_length=10000,
    block_patterns=["internal use only", "confidential"],
)
```

---

## Validation Rules

### Built-in Rules

```python
from ai_infra import ValidationRules

rules = ValidationRules(
    max_prompt_length=4000,
    max_response_length=10000,
    block_patterns=["password", "secret"],
    detect_pii=True,
    detect_injection=True,
    allowed_languages=["en", "es", "fr"],
)

# Apply to LLM
llm = LLM(provider="openai", validation=rules)
```

### Custom Rules

```python
from ai_infra import ValidationRule

def no_code_blocks(text: str) -> bool:
    """Block responses with code blocks."""
    return "```" not in text

custom_rule = ValidationRule(
    name="no_code_blocks",
    validator=no_code_blocks,
    error_message="Response must not contain code blocks",
)

rules = ValidationRules(custom_rules=[custom_rule])
```

---

## Sanitization

### Input Sanitization

```python
from ai_infra import sanitize_prompt

sanitized = sanitize_prompt(
    prompt=user_input,
    strip_html=True,
    normalize_whitespace=True,
    remove_control_chars=True,
    max_length=4000,
)
```

### Output Sanitization

```python
from ai_infra import sanitize_response

sanitized = sanitize_response(
    response=llm_response,
    redact_patterns=[
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ],
    redact_with="[REDACTED]",
)
```

---

## Token Counting

### Count Tokens

```python
from ai_infra import count_tokens

count = count_tokens(
    text="Hello, world!",
    model="gpt-4o",
)

print(f"Token count: {count}")
```

### Estimate Cost

```python
from ai_infra import estimate_cost

cost = estimate_cost(
    prompt_tokens=1000,
    completion_tokens=500,
    model="gpt-4o",
)

print(f"Estimated cost: ${cost:.4f}")
```

### Truncate to Fit

```python
from ai_infra import truncate_to_tokens

truncated = truncate_to_tokens(
    text=long_text,
    max_tokens=4000,
    model="gpt-4o",
    truncate_from="end",  # or "start", "middle"
)
```

---

## Integration with LLM

### Automatic Validation

```python
from ai_infra import LLM, ValidationRules

rules = ValidationRules(
    max_prompt_length=4000,
    detect_pii=True,
    detect_injection=True,
)

llm = LLM(
    provider="openai",
    validation=rules,
)

# Validation happens automatically
response = await llm.generate(user_input)
```

### Per-Request Validation

```python
response = await llm.generate(
    user_input,
    validate_prompt=True,
    validate_response=True,
    response_schema=User,
)
```

---

## Error Handling

```python
from ai_infra import (
    ValidationError,
    PromptTooLongError,
    BlockedPatternError,
    PIIDetectedError,
    PromptInjectionError,
    InvalidFormatError,
    SchemaValidationError,
)

try:
    response = await llm.generate(user_input)
except PromptTooLongError:
    return "Your message is too long. Please shorten it."
except PIIDetectedError:
    return "Please don't include personal information."
except PromptInjectionError:
    return "Invalid input detected."
except ValidationError as e:
    return f"Validation failed: {e.message}"
```

---

## Guardrails

### Content Guardrails

```python
from ai_infra import ContentGuardrails

guardrails = ContentGuardrails(
    block_topics=["violence", "illegal_activity"],
    require_topics=["helpful", "professional"],
    max_toxicity=0.5,
)

llm = LLM(provider="openai", guardrails=guardrails)
```

### Output Guardrails

```python
guardrails = ContentGuardrails(
    output_must_contain=["disclaimer"],
    output_must_not_contain=["guaranteed"],
    max_confidence_claims=True,
)
```

---

## Batch Validation

```python
from ai_infra import validate_batch

results = validate_batch(
    items=[input1, input2, input3],
    rules=rules,
)

for result in results:
    if result.valid:
        print(f"Valid: {result.item}")
    else:
        print(f"Invalid: {result.errors}")
```

---

## Configuration

### From Environment

```python
from ai_infra import ValidationRules

# Reads from AI_INFRA_VALIDATION_* env vars
rules = ValidationRules.from_env()
```

### From Config File

```python
rules = ValidationRules.from_file("validation.yaml")
```

### YAML Example

```yaml
# validation.yaml
max_prompt_length: 4000
max_response_length: 10000

detect_pii: true
pii_types:
  - email
  - phone
  - ssn

detect_injection: true

block_patterns:
  - password
  - api_key
  - secret

allowed_languages:
  - en
  - es
```

---

## See Also

- [Errors](errors.md) - Error handling
- [Callbacks](callbacks.md) - Execution hooks
- [LLM](../core/llm.md) - LLM usage
