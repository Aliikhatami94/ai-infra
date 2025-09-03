# MCP Stdio Publisher

Publish your **Python MCP stdio servers** as **npx-runner CLIs** so any agent can launch them without installing Python packages.

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