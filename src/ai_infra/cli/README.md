# ai-infra CLI

Command-line interface for ai-infra SDK.

## Quick Start

```bash
# Start interactive chat
ai-infra chat

# Send a one-shot message
ai-infra chat -m "What is Python?"

# Resume last session
ai-infra chat                   # Auto-resumes last session

# Start fresh session
ai-infra chat --new

# List available providers
ai-infra providers
```

## Installation

The CLI is installed automatically with ai-infra:

```bash
pip install ai-infra
```

Or with Poetry:

```bash
poetry install
poetry run ai-infra --help
```

## Commands

### Chat

Interactive chat REPL with session persistence:

```bash
# Interactive mode (auto-resumes last session)
ai-infra chat

# Start a new session (don't auto-resume)
ai-infra chat --new

# Resume or create a named session
ai-infra chat --session my-project

# With specific provider/model
ai-infra chat --provider openai --model gpt-4o

# One-shot message (no persistence)
ai-infra chat -m "Explain Docker"

# With system prompt
ai-infra chat -m "Hello" -s "You are helpful"

# JSON output (for scripting)
ai-infra chat -m "Hello" --json

# Disable persistence
ai-infra chat --no-persist
```

#### Chat Session Commands

Within the interactive chat, use these commands:

```
/help              Show all commands
/clear             Clear conversation history
/system <prompt>   Set or update system prompt
/history           Show conversation history

# Session management
/sessions          List all saved sessions
/save [name]       Save current session
/load <name>       Load a saved session
/new               Start a new session
/delete <name>     Delete a saved session
/rename <name>     Rename current session

# Model settings
/model <name>      Change model
/provider <name>   Change provider
/temp <value>      Set temperature (0.0-2.0)

/quit, /exit       Save session and exit
```

Sessions are stored in `~/.ai-infra/chat_sessions/` as JSON files.

### Discovery

List providers and models:

```bash
# List all providers
ai-infra providers

# Only configured providers
ai-infra providers --configured

# List models for a provider
ai-infra models --provider openai

# List all models (configured providers)
ai-infra models --all
```

### Image Generation

```bash
# List providers
ai-infra image-providers

# List models
ai-infra image-models --provider openai
```

### Multimodal (TTS/STT)

```bash
# TTS
ai-infra tts-providers
ai-infra tts-voices --provider openai
ai-infra tts-models --provider elevenlabs

# STT
ai-infra stt-providers
ai-infra stt-models --provider openai
```

## Environment Variables

Set API keys for providers:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

---

# MCP Stdio Publisher

Publish Python MCP stdio servers as npx-runner CLIs so any agent can launch them.

## Requirements

- Python 3.11+
- `uv`/`uvx` available (or provide `UVX_PATH`)
- Your project depends on the package that provides the CLI entrypoint:
  ```toml
  [tool.poetry.scripts]
  mcp-publish = "ai_infra.mcp.server.custom.publish.cli:app"

## Generate a shim

```bash
poetry run mcp-publish add \
  --tool-name <TOOL_NAME> \
  --module <PY_MODULE> \
  --repo https://github.com/<OWNER>/<REPO>.git \
  --ref <REF> \
  --python-package-root <PY_PKG> \
  --package-name <NPM_PKG_NAME>
```

* <TOOL_NAME>: CLI name published to users (e.g., auth-infra-mcp)
* <PY_MODULE>: module with main() that starts your MCP stdio server (e.g., svc_infra.auth.mcp)
* <OWNER>/<REPO>: GitHub owner/repo
* <REF>: branch/tag/sha (e.g., main)
* <PY_PKG>: your top-level Python package under src/ (e.g., svc_infra)
* <NPM_PKG_NAME>: name to write in package.json if creating it (e.g., mcp-stdio-expose)

This writes a shim at:
`src/<PY_PKG>/mcp-shim/bin/<TOOL_NAME>.js`

and updates/creates package.json with:
```json
{
  "bin": {
    "<TOOL_NAME>": "src/<PY_PKG>/mcp-shim/bin/<TOOL_NAME>.js"
  }
}
```

## Make the shim executable (and commit it)

You can run the provided Makefile target:
```bash
make chmod-shim \
  SHIM=src/<PY_PKG>/mcp-shim/bin/<TOOL_NAME>.js
```

Optionally commit the bit:
```bash
make commit-shim \
  SHIM=src/<PY_PKG>/mcp-shim/bin/<TOOL_NAME>.js \
  MSG="chore: ensure shim executable"
```

Note: npx installs a wrapper that runs node <file>.js, so the +x bit isn’t strictly required for consumers. Keeping it set is still good practice and enables direct execution.

## How consumers run it

```bash
npx --y --package=github:<OWNER>/<REPO> <TOOL_NAME> [args...]
# If uvx is not on PATH:
UVX_PATH=/abs/path/to/uvx npx --yes --package=github:<OWNER>/<REPO> <TOOL_NAME>
```

## MCP client config (example)

```bash
{
  "servers": {
    "<FriendlyName>": {
      "command": "npx",
      "args": ["--yes","--package=github:<OWNER>/<REPO>","<TOOL_NAME>"],
      "env": { "UVX_PATH": "/abs/path/to/uvx" }
    }
  }
}
```

## Remove a shim

```bash
poetry run mcp-publish remove \
  --tool-name <TOOL_NAME> \
  --python-package-root <PY_PKG> \
  --delete-file
```

## Runtime environment variables

* UVX_PATH — absolute path to uvx if not on PATH
* SVC_INFRA_REPO — override repo without regenerating (e.g., https://github.com/<OWNER>/<REPO>.git)
* SVC_INFRA_REF — override ref/branch/tag without regenerating
* UVX_REFRESH=1 — force uvx to refresh the env on next run
* The shim passes --quiet to uvx to reduce noise.
