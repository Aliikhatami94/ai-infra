# Workspace

> Unified file operations for agent file access.

## Quick Start

```python
from ai_infra import Workspace

workspace = Workspace("./project")

# Read files
content = workspace.read("src/main.py")

# Write files
workspace.write("output/result.txt", "Hello World")

# Search files
matches = workspace.search("TODO")
```

---

## Overview

Workspace provides a unified interface for agents to interact with files:
- Sandboxed file access with configurable boundaries
- Consistent API across local and remote storage
- Built-in safety controls and access patterns
- Integration with tools and agents

---

## Creating a Workspace

### Local Workspace

```python
from ai_infra import Workspace

# From directory path
workspace = Workspace("./my-project")

# With explicit boundaries
workspace = Workspace(
    root="./my-project",
    allowed_paths=["src/", "docs/", "tests/"],
    denied_paths=["secrets/", ".env"],
)
```

### With Configuration

```python
workspace = Workspace(
    root="./project",
    read_only=False,
    max_file_size_mb=10,
    allowed_extensions=[".py", ".md", ".json"],
)
```

---

## File Operations

### Reading Files

```python
# Read full file
content = workspace.read("src/main.py")

# Read with encoding
content = workspace.read("data/file.txt", encoding="utf-8")

# Read binary
data = workspace.read_bytes("assets/image.png")

# Check if file exists
if workspace.exists("config.json"):
    config = workspace.read("config.json")
```

### Writing Files

```python
# Write text
workspace.write("output/result.txt", "Hello World")

# Write with encoding
workspace.write("data/output.txt", content, encoding="utf-8")

# Write binary
workspace.write_bytes("output/image.png", image_data)

# Create directories automatically
workspace.write("new/nested/path/file.txt", content)
```

### File Info

```python
info = workspace.stat("src/main.py")

print(f"Size: {info.size} bytes")
print(f"Modified: {info.modified_at}")
print(f"Type: {info.file_type}")
```

---

## Directory Operations

### Listing Files

```python
# List directory
files = workspace.list_dir("src/")

# Recursive listing
all_files = workspace.list_dir("src/", recursive=True)

# Filter by pattern
py_files = workspace.glob("**/*.py")
md_files = workspace.glob("docs/**/*.md")
```

### Creating/Removing

```python
# Create directory
workspace.mkdir("new_directory")

# Remove file
workspace.remove("temp/old_file.txt")

# Remove directory
workspace.rmdir("temp/", recursive=True)
```

---

## Searching

### Text Search

```python
# Search for pattern
results = workspace.search("TODO")

for result in results:
    print(f"{result.file}:{result.line}: {result.text}")
```

### Filtered Search

```python
# Search in specific files
results = workspace.search(
    pattern="import",
    include=["**/*.py"],
    exclude=["**/test_*.py"],
)

# Case-insensitive search
results = workspace.search("error", case_sensitive=False)

# Regex search
results = workspace.search(r"def \w+\(.*\):", regex=True)
```

---

## Safety Controls

### Path Boundaries

```python
workspace = Workspace(
    root="./project",
    allowed_paths=["src/", "tests/"],
    denied_paths=["secrets/", ".git/"],
)

# Allowed
content = workspace.read("src/main.py")

# Denied - raises WorkspaceAccessError
workspace.read("secrets/api_key.txt")
workspace.read("../outside_project/file.txt")
```

### Read-Only Mode

```python
workspace = Workspace("./project", read_only=True)

# Allowed
content = workspace.read("file.txt")

# Denied - raises WorkspaceReadOnlyError
workspace.write("file.txt", "new content")
```

### File Size Limits

```python
workspace = Workspace(
    root="./project",
    max_file_size_mb=10,
)

# Raises error if file exceeds limit
workspace.read("huge_file.txt")
```

### Extension Filtering

```python
workspace = Workspace(
    root="./project",
    allowed_extensions=[".py", ".md", ".json", ".txt"],
)

# Denied - extension not allowed
workspace.read("script.sh")
```

---

## Agent Integration

### As Tool

```python
from ai_infra import Agent, workspace_tools

workspace = Workspace("./project")
agent = Agent(
    persona=persona,
    tools=workspace_tools(workspace),
)

# Agent can now read/write/search files
result = agent.run("Read the main.py file and summarize it")
```

### Custom Tool Configuration

```python
from ai_infra import workspace_tools

# Read-only tools
tools = workspace_tools(workspace, write=False)

# Full access tools
tools = workspace_tools(workspace, write=True, search=True)
```

---

## Remote Workspaces

### S3 Workspace

```python
from ai_infra import S3Workspace

workspace = S3Workspace(
    bucket="my-bucket",
    prefix="project/",
    region="us-east-1",
)

# Same API as local workspace
content = workspace.read("src/main.py")
```

### GCS Workspace

```python
from ai_infra import GCSWorkspace

workspace = GCSWorkspace(
    bucket="my-bucket",
    prefix="project/",
)
```

---

## Context Manager

```python
from ai_infra import Workspace

with Workspace("./project") as ws:
    content = ws.read("file.txt")
    ws.write("output.txt", processed_content)
# Cleanup happens automatically
```

---

## Error Handling

```python
from ai_infra import (
    Workspace,
    WorkspaceError,
    WorkspaceAccessError,
    WorkspaceNotFoundError,
    WorkspaceReadOnlyError,
)

workspace = Workspace("./project")

try:
    content = workspace.read("missing.txt")
except WorkspaceNotFoundError:
    print("File not found")
except WorkspaceAccessError:
    print("Access denied")
except WorkspaceError as e:
    print(f"Workspace error: {e}")
```

---

## Snapshots

Create point-in-time snapshots:

```python
# Create snapshot
snapshot = workspace.create_snapshot()

# Make changes
workspace.write("file.txt", "new content")

# Restore from snapshot
workspace.restore_snapshot(snapshot)
```

---

## See Also

- [Agent](../core/agents.md) - Using agents with workspace
- [Tools](../tools/schema-tools.md) - Creating custom tools
- [Deep Agent](deep-agent.md) - Autonomous file operations
