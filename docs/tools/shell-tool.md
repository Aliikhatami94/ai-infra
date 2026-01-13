# Shell Tool

> Execute shell commands from AI agents with security, redaction, and persistence.

## Quick Start

```python
from ai_infra import Agent
from ai_infra.llm.shell import run_shell

agent = Agent(tools=[run_shell])
result = agent.run("List all Python files in the current directory")
```

---

## Overview

The `run_shell` tool enables AI agents to execute shell commands on the host system. It provides:

- **Session persistence** — Environment variables and working directory persist across calls
- **Output redaction** — Sensitive data (API keys, passwords) is automatically redacted
- **Security validation** — Dangerous commands are blocked by default
- **Structured results** — Exit codes, stdout, stderr in a consistent format

---

## Function Signature

```python
@tool
async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a shell command and return the output.

    Args:
        command: Shell command to execute. Supports pipes, redirects, etc.
        cwd: Working directory for the command. If not specified, uses the
            current session's working directory or the current directory.
        timeout: Maximum seconds to wait for the command to complete.
            Default is 120 seconds.

    Returns:
        Dictionary with execution result:
        - success: Whether the command succeeded (exit code 0)
        - exit_code: The command's exit code
        - stdout: Standard output from the command
        - stderr: Standard error from the command
        - command: The command that was executed
        - timed_out: Whether the command timed out
    """
```

---

## Parameters

### `command` (required)

The shell command to execute. Supports full shell syntax including pipes, redirects, and command chaining.

```python
# Simple command
result = await run_shell.ainvoke({"command": "ls -la"})

# Pipes
result = await run_shell.ainvoke({"command": "cat file.txt | grep 'error'"})

# Command chaining
result = await run_shell.ainvoke({"command": "cd /tmp && ls"})

# Redirects
result = await run_shell.ainvoke({"command": "echo 'hello' > output.txt"})
```

### `cwd`

Working directory for the command. If not specified, uses the session's current directory.

```python
# Run in specific directory
result = await run_shell.ainvoke({
    "command": "npm install",
    "cwd": "/path/to/project"
})
```

### `timeout`

Maximum seconds to wait for the command to complete. Default is 120 seconds.

```python
# Long-running command with extended timeout
result = await run_shell.ainvoke({
    "command": "npm run build",
    "timeout": 300  # 5 minutes
})
```

---

## Return Value

The tool returns a dictionary with the execution result:

```python
{
    "success": True,           # exit_code == 0
    "exit_code": 0,            # Process exit code
    "stdout": "output...",     # Standard output (redacted)
    "stderr": "",              # Standard error (redacted)
    "command": "ls -la",       # The command that was executed
    "timed_out": False         # Whether timeout occurred
}
```

### Handling Results

```python
result = await run_shell.ainvoke({"command": "pytest -v"})

if result["success"]:
    print("Tests passed!")
    print(result["stdout"])
else:
    print(f"Tests failed with exit code {result['exit_code']}")
    print(result["stderr"])
```

---

## Usage Modes

### Standalone Mode (Stateless)

When used without a session, each command runs in isolation:

```python
from ai_infra.llm.shell import run_shell

# Each command starts fresh
result1 = await run_shell.ainvoke({"command": "export FOO=bar"})
result2 = await run_shell.ainvoke({"command": "echo $FOO"})
# $FOO is empty - no persistence between calls
```

### Session Mode (Persistent)

When used with `ShellMiddleware` or a `ShellSession`, state persists:

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/my/project")]
)

# Agent can run multiple commands with persistent state
result = agent.run("""
    1. Set up a virtual environment
    2. Install dependencies
    3. Run the tests
""")
# Working directory and environment persist across commands
```

---

## Custom Tool Configuration

Use `create_shell_tool()` for custom configurations:

```python
from ai_infra.llm.shell import create_shell_tool

# Custom timeout and working directory
shell_tool = create_shell_tool(
    default_timeout=300.0,
    default_cwd="/my/project",
)

# Allowlist for restricted commands (Phase 2.4)
safe_shell = create_shell_tool(
    allowed_commands=("pytest", "npm", "make", "poetry"),
)
# Only commands starting with these prefixes are allowed

agent = Agent(tools=[shell_tool])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `ShellSession` | `None` | Bind to specific session |
| `dangerous_pattern_check` | `bool` | `True` | Check for dangerous commands |
| `custom_dangerous_patterns` | `tuple[Pattern]` | `None` | Additional patterns to block |
| `default_timeout` | `float` | `120.0` | Default command timeout |
| `default_cwd` | `Path \| str` | `None` | Default working directory |
| `allowed_commands` | `tuple[str]` | `None` | Allowlist of command prefixes |

---

## Output Redaction

Sensitive data is automatically redacted from command output:

```python
result = await run_shell.ainvoke({"command": "cat .env"})
# Output: OPENAI_API_KEY=[REDACTED:openai_api_key]
#         DATABASE_URL=[REDACTED:connection_string]
```

### Default Redaction Rules

| Pattern | Description |
|---------|-------------|
| `sk-[a-zA-Z0-9]{32,}` | OpenAI API keys |
| `ghp_[a-zA-Z0-9]{36}` | GitHub tokens |
| `xoxb-[a-zA-Z0-9-]+` | Slack tokens |
| `password[=:]\s*\S+` | Password patterns |
| `secret[=:]\s*\S+` | Secret patterns |
| `token[=:]\s*\S+` | Token patterns |
| Connection strings | Database URLs |

### Custom Redaction Rules

```python
from ai_infra.llm.shell import RedactionRule, ShellMiddleware

custom_rules = (
    RedactionRule(
        name="internal_api_key",
        pattern=r"INTERNAL_[A-Z0-9]{24}",
        replacement="[REDACTED:internal_key]"
    ),
)

middleware = ShellMiddleware(
    redaction_rules=custom_rules,
)
```

---

## Security

### Dangerous Command Detection

By default, the following patterns are blocked:

```python
# Blocked patterns (examples)
"rm -rf /"          # Destructive filesystem commands
"rm -rf ~"          # Home directory deletion
"mkfs"              # Filesystem formatting
"dd of=/dev/"       # Direct device writes
"curl | bash"       # Remote code execution
"wget | bash"       # Remote code execution
"chmod 777 /"       # Insecure permissions
"> /etc/passwd"     # System file overwrites
```

### Security Policies

For fine-grained control, use security policies:

```python
from ai_infra.llm.shell import SecurityPolicy, create_strict_policy

# Strict policy - only allow specific commands
policy = create_strict_policy()

# Or customize
policy = SecurityPolicy(
    allowed_patterns=[r"pytest.*", r"npm (test|install).*"],
    denied_patterns=[r"sudo.*", r"rm -rf.*"],
    allow_network=False,  # Block curl, wget, etc.
)
```

---

## Error Handling

### Timeout

```python
result = await run_shell.ainvoke({
    "command": "sleep 1000",
    "timeout": 5
})

if result["timed_out"]:
    print("Command timed out")
    # exit_code is -1 for timeouts
```

### Command Failure

```python
result = await run_shell.ainvoke({"command": "false"})

if not result["success"]:
    print(f"Command failed: exit code {result['exit_code']}")
    print(f"Error: {result['stderr']}")
```

### Invalid Working Directory

```python
result = await run_shell.ainvoke({
    "command": "ls",
    "cwd": "/nonexistent/path"
})

if not result["success"]:
    # stderr contains: "Working directory does not exist: ..."
    print(result["stderr"])
```

---

## Integration with Executor

The shell tool is automatically available in the Executor:

```python
from ai_infra import Executor

executor = Executor(
    roadmap_path="/path/to/ROADMAP.md",
    enable_shell=True,           # Default: True
    shell_timeout=120.0,         # Default timeout
    shell_workspace="/project",  # Working directory
)

await executor.arun()
# Agent can use run_shell to build, test, and verify tasks
```

### CLI Options

```bash
# Enable shell (default)
ai-infra executor run --roadmap ROADMAP.md --enable-shell

# Disable shell for safety
ai-infra executor run --roadmap ROADMAP.md --no-shell

# Custom timeout
ai-infra executor run --roadmap ROADMAP.md --shell-timeout 300

# Restrict to specific commands
ai-infra executor run --roadmap ROADMAP.md --shell-allowed-commands "pytest,npm,make"
```

---

## See Also

- [ShellMiddleware](shell-middleware.md) — Persistent sessions for agents
- [Using Shell Tool in Agents](../guides/shell-tool-guide.md) — Usage guide
- [Security Best Practices](../guides/shell-security.md) — Security guidelines
