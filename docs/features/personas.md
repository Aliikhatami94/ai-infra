# Agent Personas

> Configure agent behavior with YAML-driven personas.

## Quick Start

```python
from ai_infra import Agent, Persona

persona = Persona.from_yaml("analyst.yaml")
agent = Agent(persona=persona)
result = agent.run("Analyze this data")
```

---

## Overview

Personas define an agent's behavior, capabilities, and constraints through configuration files. This enables:
- Reusable agent configurations
- Non-code behavior changes
- Role-based agent customization
- Easy A/B testing of agent behaviors

---

## Persona YAML Format

```yaml
# analyst.yaml
name: data_analyst
display_name: Data Analyst
description: Expert in data analysis and visualization

system_prompt: |
  You are an expert data analyst. You excel at:
  - Statistical analysis
  - Data visualization recommendations
  - Finding patterns and insights

  Always explain your reasoning clearly.

# Default model settings
provider: openai
model: gpt-4o
temperature: 0.3

# Allowed tools
tools:
  - search
  - calculate
  - create_chart

# Constraints
max_iterations: 10
require_approval:
  - delete_data
  - modify_database
```

---

## Loading Personas

### From YAML File

```python
from ai_infra import Persona

persona = Persona.from_yaml("personas/analyst.yaml")
```

### From YAML String

```python
yaml_content = """
name: helper
system_prompt: You are a helpful assistant.
provider: openai
model: gpt-4o-mini
"""

persona = Persona.from_yaml_string(yaml_content)
```

### From Dict

```python
persona = Persona(
    name="helper",
    system_prompt="You are a helpful assistant.",
    provider="openai",
    model="gpt-4o-mini",
)
```

---

## Using with Agent

```python
from ai_infra import Agent, Persona

# Load persona
persona = Persona.from_yaml("support.yaml")

# Create agent with persona
agent = Agent(persona=persona)

# Persona settings are applied automatically
result = agent.run("Help me with my issue")
```

### Override Persona Settings

```python
agent = Agent(persona=persona)

# Override specific settings
result = agent.run(
    "Quick task",
    temperature=0.9,  # Override persona's temperature
    model_name="gpt-4o-mini"  # Override persona's model
)
```

---

## Persona Properties

### Basic Properties

```yaml
name: unique_identifier
display_name: Human Readable Name
description: What this persona does
```

### System Prompt

```yaml
system_prompt: |
  You are a specialized assistant.

  Your responsibilities:
  - Task 1
  - Task 2

  Guidelines:
  - Be concise
  - Be accurate
```

### Model Settings

```yaml
provider: openai
model: gpt-4o
temperature: 0.7
max_tokens: 4096
```

### Tool Configuration

```yaml
# Allowed tools
tools:
  - search
  - calculate
  - send_email

# Tools requiring approval
require_approval:
  - send_email
  - delete_file
```

### Constraints

```yaml
max_iterations: 10
timeout: 30
max_tokens: 4096
```

---

## Example Personas

### Technical Support

```yaml
# support.yaml
name: support_agent
display_name: Technical Support Agent
description: Helps users resolve technical issues

system_prompt: |
  You are a technical support specialist. Your goal is to help users
  resolve their technical issues efficiently and professionally.

  Guidelines:
  - Ask clarifying questions before suggesting solutions
  - Provide step-by-step instructions
  - Escalate complex issues when needed
  - Always be patient and empathetic

provider: openai
model: gpt-4o
temperature: 0.5

tools:
  - search_knowledge_base
  - create_ticket
  - check_status

require_approval:
  - escalate_ticket
  - refund_request
```

### Code Reviewer

```yaml
# reviewer.yaml
name: code_reviewer
display_name: Code Reviewer
description: Reviews code for quality and best practices

system_prompt: |
  You are an expert code reviewer. Analyze code for:
  - Bugs and potential issues
  - Performance problems
  - Security vulnerabilities
  - Code style and best practices

  Provide constructive feedback with specific suggestions.

provider: anthropic
model: claude-sonnet-4-20250514
temperature: 0.2

tools:
  - read_file
  - search_codebase
  - run_linter
```

### Creative Writer

```yaml
# writer.yaml
name: creative_writer
display_name: Creative Writer
description: Generates creative content

system_prompt: |
  You are a creative writer with expertise in:
  - Storytelling
  - Marketing copy
  - Blog posts
  - Social media content

  Be creative, engaging, and adapt your tone to the audience.

provider: openai
model: gpt-4o
temperature: 0.9

tools:
  - search_web
  - generate_image
```

---

## Persona Registry

Manage multiple personas:

```python
from ai_infra import Persona

# Load all personas from directory
personas = Persona.load_directory("./personas/")

# Get specific persona
analyst = personas["data_analyst"]
support = personas["support_agent"]

# List available personas
print(personas.keys())
```

---

## Dynamic Personas

Create personas programmatically:

```python
def create_persona_for_domain(domain: str) -> Persona:
    return Persona(
        name=f"{domain}_expert",
        system_prompt=f"You are an expert in {domain}. Help users with {domain}-related questions.",
        provider="openai",
        model="gpt-4o",
        temperature=0.5,
    )

finance_persona = create_persona_for_domain("finance")
health_persona = create_persona_for_domain("healthcare")
```

---

## See Also

- [Agent](../core/agents.md) - Using agents
- [Deep Agent](deep-agent.md) - Autonomous agents
- [Replay](replay.md) - Debug agent workflows
