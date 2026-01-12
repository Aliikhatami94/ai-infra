"""Few-shot prompt templates for the Executor.

Phase 1.5: Improve LLM output quality by including examples in every prompt.

Weaker models (gpt-4.1-mini, gpt-4.1-nano) benefit significantly from few-shot
examples. Strong models are not hurt by them. These templates are used by the
context builder when generating prompts for task execution.

Templates are organized by task type:
- PYTHON_CODE_TEMPLATE: For creating/modifying Python source files
- TEST_CODE_TEMPLATE: For creating/modifying pytest test files
- CONFIG_TEMPLATE: For configuration files (YAML, TOML, JSON, INI)
- SCRIPT_TEMPLATE: For shell/CLI scripts
- DOCUMENTATION_TEMPLATE: For markdown documentation
- GENERIC_TEMPLATE: Fallback for unrecognized task types
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class TemplateType(Enum):
    """Template types for task classification."""

    PYTHON_CODE = "python_code"
    TEST_CODE = "test_code"
    CONFIG = "config"
    SCRIPT = "script"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"


# =============================================================================
# Python Code Template
# =============================================================================

PYTHON_CODE_TEMPLATE = '''You are implementing a Python module.

## Example Task
Task: Create a function that calculates the factorial of a number.

```python
def factorial(n: int) -> int:
    """Calculate the factorial of n.

    Args:
        n: Non-negative integer.

    Returns:
        The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

## Your Task
Task: {task_description}

## Instructions
1. Follow the example's style (docstrings, type hints, error handling)
2. Use descriptive variable names
3. Include proper error handling with specific exception types
4. Add type hints for all function parameters and return values
5. Write Google-style docstrings for all public functions
6. Do NOT include markdown code fences in your response
7. Return ONLY the Python code, nothing else

## Your Implementation
'''

# =============================================================================
# Test Code Template
# =============================================================================

TEST_CODE_TEMPLATE = '''You are writing pytest tests for a Python module.

## Example Task
Task: Write tests for a factorial function.

```python
import pytest

from src.math_utils import factorial


class TestFactorial:
    """Tests for the factorial function."""

    def test_zero_returns_one(self) -> None:
        """Test: factorial(0) returns 1."""
        assert factorial(0) == 1

    def test_positive_integer(self) -> None:
        """Test: factorial(5) returns 120."""
        assert factorial(5) == 120

    def test_one_returns_one(self) -> None:
        """Test: factorial(1) returns 1."""
        assert factorial(1) == 1

    def test_negative_raises_value_error(self) -> None:
        """Test: negative input raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            factorial(-1)
```

## Your Task
Task: {task_description}

## Instructions
1. Follow the example's style (one test per case, descriptive names)
2. Use test classes to group related tests
3. Test happy path, edge cases, and error cases
4. Use pytest fixtures if appropriate for setup/teardown
5. Add type hints to test methods (return None)
6. Use descriptive docstrings for each test method
7. Do NOT include markdown code fences in your response
8. Return ONLY the Python code, nothing else

## Your Implementation
'''

# =============================================================================
# Configuration Template
# =============================================================================

CONFIG_TEMPLATE = """You are creating or modifying a configuration file.

## Example Task
Task: Create a pyproject.toml with project metadata and dependencies.

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "A sample Python project"
authors = [
    {{name = "Developer", email = "dev@example.com"}}
]
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py311"
```

## Your Task
Task: {task_description}

## Instructions
1. Follow the example's structure and formatting
2. Use proper syntax for the configuration format (TOML, YAML, JSON, INI)
3. Include helpful comments where appropriate
4. Use sensible defaults for optional values
5. Do NOT include markdown code fences in your response
6. Return ONLY the configuration content, nothing else

## Your Configuration
"""

# =============================================================================
# Script Template
# =============================================================================

SCRIPT_TEMPLATE = """You are creating a shell script or CLI utility.

## Example Task
Task: Create a script to run tests with coverage.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run tests with coverage
# Usage: ./scripts/test.sh [pytest-args...]

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running tests with coverage..."
pytest --cov=src --cov-report=term-missing "$@"

echo "Done."
```

## Your Task
Task: {task_description}

## Instructions
1. Follow the example's style (set -euo pipefail, comments, quoting)
2. Include a usage comment at the top
3. Use proper variable quoting ("$VAR" not $VAR)
4. Handle errors gracefully
5. Make the script portable (use env for shebang)
6. Do NOT include markdown code fences in your response
7. Return ONLY the script content, nothing else

## Your Script
"""

# =============================================================================
# Documentation Template
# =============================================================================

DOCUMENTATION_TEMPLATE = """You are writing documentation in Markdown.

## Example Task
Task: Create a README for a Python utility library.

```markdown
# my-project

A Python utility library for common operations.

## Installation

```bash
pip install my-project
```

## Quick Start

```python
from my_project import hello

result = hello("World")
print(result)  # Hello, World!
```

## Features

- Simple API
- Type-safe with full type hints
- Comprehensive test coverage

## License

MIT
```

## Your Task
Task: {task_description}

## Instructions
1. Follow the example's structure (title, sections, code blocks)
2. Use proper Markdown formatting
3. Include code examples where appropriate
4. Keep it concise but complete
5. Do NOT include triple backticks around the entire response
6. Return ONLY the Markdown content

## Your Documentation
"""

# =============================================================================
# Generic Template (Fallback)
# =============================================================================

GENERIC_TEMPLATE = """You are an autonomous development agent executing a task.

## Your Task
Task: {task_description}

## Instructions
1. Analyze the task requirements carefully
2. Implement exactly what is requested
3. Follow project conventions and best practices
4. Include appropriate error handling
5. Add documentation where helpful
6. Do NOT include unnecessary code fences
7. Return ONLY the requested content

## Your Implementation
"""


# =============================================================================
# Template Selection
# =============================================================================

# Keywords for template type inference
_TEMPLATE_KEYWORDS: dict[TemplateType, list[str]] = {
    TemplateType.TEST_CODE: [
        "test",
        "tests",
        "testing",
        "pytest",
        "unittest",
        "test_",
        "_test",
        "spec",
    ],
    TemplateType.CONFIG: [
        "config",
        "configuration",
        "settings",
        "pyproject",
        "setup.cfg",
        "setup.py",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
        ".env",
    ],
    TemplateType.SCRIPT: [
        "script",
        "bash",
        "shell",
        ".sh",
        "makefile",
        "dockerfile",
        "cli",
        "command",
    ],
    TemplateType.DOCUMENTATION: [
        "readme",
        "docs",
        "documentation",
        "document",
        ".md",
        "markdown",
        "changelog",
        "contributing",
    ],
    TemplateType.PYTHON_CODE: [
        ".py",
        "python",
        "module",
        "class",
        "function",
        "implement",
        "create",
        "add",
        "feature",
        "refactor",
        "fix",
    ],
}

# Template mapping
TEMPLATES: dict[TemplateType, str] = {
    TemplateType.PYTHON_CODE: PYTHON_CODE_TEMPLATE,
    TemplateType.TEST_CODE: TEST_CODE_TEMPLATE,
    TemplateType.CONFIG: CONFIG_TEMPLATE,
    TemplateType.SCRIPT: SCRIPT_TEMPLATE,
    TemplateType.DOCUMENTATION: DOCUMENTATION_TEMPLATE,
    TemplateType.GENERIC: GENERIC_TEMPLATE,
}


def infer_template_type(
    task_title: str,
    task_description: str = "",
    file_hints: list[str] | None = None,
) -> TemplateType:
    """Infer the appropriate template type from task metadata.

    Uses heuristics based on keywords in the task title, description,
    and file hints to determine which few-shot template to use.

    Args:
        task_title: Title of the task.
        task_description: Optional description of the task.
        file_hints: Optional list of file paths involved.

    Returns:
        The inferred TemplateType.

    Examples:
        >>> infer_template_type("Add tests for user module")
        <TemplateType.TEST_CODE: 'test_code'>

        >>> infer_template_type("Create pyproject.toml")
        <TemplateType.CONFIG: 'config'>
    """
    # Combine all text for keyword matching
    combined = f"{task_title} {task_description}".lower()
    if file_hints:
        combined += " " + " ".join(file_hints).lower()

    # Score each template type
    scores: dict[TemplateType, int] = dict.fromkeys(TemplateType, 0)

    for template_type, keywords in _TEMPLATE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                scores[template_type] += 1
                # Bonus for exact filename match
                if keyword.startswith(".") and file_hints:
                    for hint in file_hints:
                        if hint.endswith(keyword):
                            scores[template_type] += 2

    # Find highest scoring type
    max_score = max(scores.values())
    if max_score == 0:
        return TemplateType.GENERIC

    # Test code takes priority if detected (specific > general)
    if scores[TemplateType.TEST_CODE] > 0:
        return TemplateType.TEST_CODE

    # Config and script take priority over generic Python
    for priority_type in [TemplateType.CONFIG, TemplateType.SCRIPT, TemplateType.DOCUMENTATION]:
        if scores[priority_type] == max_score:
            return priority_type

    # Default to Python code for remaining cases
    if scores[TemplateType.PYTHON_CODE] > 0:
        return TemplateType.PYTHON_CODE

    return TemplateType.GENERIC


def get_template(template_type: TemplateType) -> str:
    """Get the template string for a given type.

    Args:
        template_type: The type of template to retrieve.

    Returns:
        The template string with {task_description} placeholder.
    """
    return TEMPLATES.get(template_type, GENERIC_TEMPLATE)


def format_template(
    task_title: str,
    task_description: str = "",
    file_hints: list[str] | None = None,
    template_type: TemplateType | None = None,
) -> str:
    """Format a template with the task description.

    If template_type is not provided, it will be inferred from the task metadata.

    Args:
        task_title: Title of the task.
        task_description: Description of the task.
        file_hints: Optional list of file paths involved.
        template_type: Optional explicit template type.

    Returns:
        Formatted template with task description filled in.
    """
    if template_type is None:
        template_type = infer_template_type(task_title, task_description, file_hints)

    template = get_template(template_type)

    # Combine title and description for the placeholder
    full_description = task_title
    if task_description:
        full_description = f"{task_title}\n\n{task_description}"

    return template.format(task_description=full_description)


def get_few_shot_section(
    task: Any,
    template_type: TemplateType | None = None,
) -> str:
    """Get a few-shot example section for a task.

    This function extracts task metadata and returns a formatted
    few-shot template section suitable for inclusion in prompts.

    Args:
        task: Task object with title, description, and file_hints attributes.
        template_type: Optional explicit template type override.

    Returns:
        Formatted few-shot section string.
    """
    title = getattr(task, "title", "")
    description = getattr(task, "description", "")
    file_hints = getattr(task, "file_hints", None)

    return format_template(
        task_title=title,
        task_description=description,
        file_hints=file_hints,
        template_type=template_type,
    )
