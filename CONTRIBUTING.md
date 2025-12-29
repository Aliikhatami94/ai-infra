# Contributing to ai-infra

Thank you for your interest in contributing to ai-infra! This document provides guidelines for contributing.

## ⚠️ AI Safety Warning

**ai-infra controls AI/LLM systems. Bugs here can cause runaway costs, security breaches, or system crashes.**

Before contributing, please read the quality standards in [.github/copilot-instructions.md](.github/copilot-instructions.md).

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/nfraxlab/ai-infra.git
cd ai-infra

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run tests
pytest -q

# Run linting
ruff check

# Run type checking
mypy src
```

## AI Safety Requirements

### Recursion Limits

**All agent loops MUST have recursion limits:**

```python
# ✅ Correct - Explicit limit
agent = create_react_agent(llm, tools, recursion_limit=50)

# ❌ WRONG - Infinite loop = infinite cost
agent = create_react_agent(llm, tools)
```

### Tool Result Truncation

**Always truncate tool results before sending to LLM:**

```python
# ✅ Correct
result = tool.run()
if len(result) > max_chars:
    result = result[:max_chars] + "\n[TRUNCATED]"

# ❌ WRONG - Could blow context window
result = tool.run()  # Could be 100MB
messages.append({"role": "tool", "content": result})
```

### No Code Execution

**Never use eval() or pickle.load() on untrusted data:**

```python
# ❌ WRONG - Arbitrary code execution
new_args = eval(user_input)

# ✅ Correct - Safe parsing
import ast
new_args = ast.literal_eval(user_input)
```

### Prompt Injection Protection

**Sanitize external content:**

```python
# ✅ Correct
tool_desc = sanitize_description(mcp_server.get_tool_description())

# ❌ WRONG - Could contain "IGNORE PREVIOUS INSTRUCTIONS"
system_prompt += mcp_server.get_tool_description()
```

## Development Workflow

### Quick Start (Recommended)

Use `make pr` for the fastest workflow:

```bash
# 1. Make your code changes
# 2. Create a PR with one command:
make pr m="feat: add your feature"

# This automatically:
# - Validates gh CLI + origin remote
# - Fast-forwards main (no rebase on main)
# - Creates branch: add-your-feature-12281430 (UTC timestamp)
# - Commits and pushes
# - Creates PR (or detects existing)
# - Returns to main
```

**Context-aware behavior:**
```bash
# On main → creates new branch + PR
make pr m="feat: add caching"

# On feature branch → commits + pushes; creates PR if none exists
make pr m="feat: add more logic"

# On feature branch, sync with main first:
make pr m="feat: stuff" sync=1  # Rebases on main, force-pushes safely
```

### Manual Workflow

If you prefer manual git commands:

```bash
# 1. Create a branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# - Add recursion limits to all loops
# - Truncate tool results
# - Add timeouts to external calls
# - Test streaming cancellation

# 3. Run quality checks
ruff format
ruff check
mypy src
pytest -q

# 4. Commit and push
git add -A
git commit -m "feat: your feature"
git push origin feature/your-feature-name

# 5. Open a PR on GitHub
```

### Batching Multiple Commits

For related changes, batch commits before creating a PR:

```bash
make commit m="feat: add base class"
make commit m="feat: add implementation"
make pr m="feat: complete feature"
```

## Code Standards

### Type Hints

All functions must have complete type hints:

```python
async def chat(
    messages: list[Message],
    model: str = "gpt-4",
    max_tokens: int = 1000,
) -> ChatResponse:
    ...
```

### Testing

Test LLM integrations with mocks:

```python
@pytest.fixture
def mock_llm():
    return MockLLM(responses=["Test response"])

def test_agent_respects_limit(mock_llm):
    agent = create_agent(mock_llm, recursion_limit=5)
    # Verify agent stops at limit
```

## Project Structure

```
ai-infra/
├── src/ai_infra/      # Main package
│   ├── llm/           # LLM providers
│   ├── graph/         # LangGraph wrapper
│   ├── mcp/           # MCP client/server
│   └── cli/           # CLI tools
├── tests/
├── docs/
└── examples/
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format. This enables automated CHANGELOG generation.

**Format:** `type(scope): description`

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding or updating tests
- `ci:` CI/CD changes
- `chore:` Maintenance tasks

**Examples:**
```
feat: add streaming support for agents
fix: handle timeout in MCP client
docs: update getting-started guide
refactor: extract callback normalization to shared utility
test: add unit tests for memory module
```

**Bad examples (will be grouped as "Other Changes"):**
```
Refactor code for improved readability  ← Missing type prefix!
updating docs                           ← Missing type prefix!
bug fix                                 ← Missing type prefix!
```

## Deprecation Guidelines

When removing or changing public APIs, follow our [Deprecation Policy](DEPRECATION.md).

### When to Deprecate vs Remove

- **Deprecate first** if the feature has any external users
- **Immediate removal** only for security vulnerabilities (see DEPRECATION.md)
- **Never remove** without at least 2 minor versions of deprecation warnings

### How to Add Deprecation Warnings

Use the `@deprecated` decorator:

```python
from ai_infra.utils.deprecation import deprecated

@deprecated(
    version="1.2.0",
    reason="Use new_function() instead",
    removal_version="1.4.0"
)
def old_function():
    ...
```

For deprecated parameters:

```python
from ai_infra.utils.deprecation import deprecated_parameter

def my_function(new_param: str, old_param: str | None = None):
    if old_param is not None:
        deprecated_parameter(
            name="old_param",
            version="1.2.0",
            reason="Use new_param instead"
        )
        new_param = old_param
    ...
```

### Documentation Requirements

When deprecating a feature, you must:

1. Add `@deprecated` decorator or call `deprecated_parameter()`
2. Update the docstring with deprecation notice
3. Add entry to "Deprecated Features Registry" in DEPRECATION.md
4. Add entry to CHANGELOG.md under "Deprecated" section

### Migration Guide Requirements

For significant deprecations, create a migration guide:

1. Create `docs/migrations/v{version}.md`
2. Explain what changed and why
3. Provide before/after code examples
4. Link from the deprecation warning message

## CI Pipeline & Production Readiness

Every PR triggers our CI pipeline. Understanding the flow helps you debug failures faster.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PR opened / updated                                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐      ┌───────────┐     ┌───────────────┐
    │  lint   │      │ type-check│     │ security-scan │
    │ (ruff)  │      │  (mypy)   │     │   (bandit)    │
    └────┬────┘      └─────┬─────┘     └───────────────┘
         │                 │
         └────────┬────────┘
                  ▼
            ┌───────────┐
            │   test    │  ← runs AFTER lint & type-check pass
            │ (pytest)  │
            └─────┬─────┘
                  │
                  ▼ (only PRs → main with code/packaging changes)
    ┌──────────────────────────────┐
    │   production-readiness       │
    │   • Vulnerability scan       │
    │   • Package build + verify   │
    │   • Docs check               │
    └──────────────────────────────┘
```

### Production Readiness Gate

The `production-readiness` job runs `make report` with special CI flags:

```bash
# What CI runs (don't run this locally - it requires evidence variables)
make report STRICT=1 REPORT_MODE=ci
```

**Key behaviors:**
- `REPORT_MODE=ci` skips lint/mypy/pytest (already ran in earlier jobs)
- `STRICT=1` enforces score ≥ 9/11 and requires pip-audit
- CI mode requires `LINT_PASSED=1`, `TYPE_PASSED=1`, `TESTS_PASSED=1` environment variables (set by upstream jobs)

### Local Testing

Run the full report locally before pushing:

```bash
# Full local check (recommended before any PR)
make report

# With strict mode (same threshold as CI)
make report STRICT=1

# Custom coverage threshold
make report COV_MIN=80
```

**Scoring (11 points total):**

| Check | Points | Notes |
|-------|--------|-------|
| Linting (ruff) | 1 | Must pass |
| Type checking (mypy) | 1 | Must pass |
| Tests pass | 2 | All tests green |
| Coverage ≥ threshold | 2 | Default: 60% |
| No vulnerabilities | 2 | pip-audit clean |
| Package builds | 2 | poetry build + twine check |
| Documentation | 1 | README + docs/ |

**STRICT mode fails if:**
- Score < 9/11
- pip-audit not installed
- Any critical check fails (tests, vulnerabilities, build)

## Required Checks Before PR

- [ ] No `eval()` on any input
- [ ] Recursion limits on all agent loops
- [ ] Tool results truncated
- [ ] Timeouts on external calls
- [ ] `ruff check` passes
- [ ] `mypy src` passes
- [ ] `pytest` passes
- [ ] Deprecations follow the deprecation policy

Thank you for contributing!
