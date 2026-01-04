# Security Audit

**Last Audited**: January 3, 2026
**Reviewer**: AI Agent

This document records all Bandit security scanner skips and their justifications.

---

## Bandit Configuration

The following Bandit rules are skipped in `pyproject.toml`:

```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "venv"]
skips = [
    "B101",  # assert used (intentional in tests and contracts)
    "B104",  # hardcoded_bind_all_interfaces (intentional for dev servers in examples)
    "B301",  # pickle load (our FAISS/retriever persistence is trusted local files)
    "B311",  # random for non-crypto (intentional)
    "B608",  # hardcoded_sql_expressions (table names are validated, not user input)
]
```

---

## Skip Justifications

### B101: Use of assert

**Locations**: Throughout codebase
**Justification**: Assertions are used for programming contracts and invariants in non-production paths. Tests use assertions extensively for validation.
**Risk**: None - assertions are appropriate here.

### B104: Binding to 0.0.0.0

**Location**: [src/ai_infra/mcp/server/server.py](../src/ai_infra/mcp/server/server.py) line 582

```python
def run_uvicorn(self, host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
```

**Justification**: MCP servers are designed to run in containers where binding to all interfaces is required for external access. The default is overridable by the caller.
**Risk**: Low - users control deployment environment. Documentation recommends binding to localhost for local development.

### B301: Pickle Deserialization

**Locations**:
- [src/ai_infra/retriever/retriever.py](../src/ai_infra/retriever/retriever.py) lines 1263-1399
- [src/ai_infra/retriever/backends/faiss.py](../src/ai_infra/retriever/backends/faiss.py) lines 127-167

**Justification**: Pickle is used for Retriever persistence (saving/loading embeddings and metadata). Files are local, user-created artifacts - not untrusted input.

**Risk**: Medium - pickle can execute arbitrary code if an attacker provides a malicious file.

**Mitigations**:
1. Warning logged on load: "Loading a pickle file can execute arbitrary code"
2. Users must explicitly provide file paths
3. **Planned removal**: Phase 2.2 of security roadmap will replace pickle with JSON + numpy format

### B311: Random for Non-Crypto

**Locations**: Various utility functions
**Justification**: `random` module is used for non-security purposes (sampling, shuffling). Cryptographic operations use `secrets` module.
**Risk**: None - appropriate use of randomness.

### B608: Hardcoded SQL Expressions

**Locations**: [src/ai_infra/llm/memory/store.py](../src/ai_infra/llm/memory/store.py)

**Justification**: All SQL uses parameterized queries with `?` placeholders. Table and column names are hardcoded constants, not user input.

```python
cursor.execute(
    """
    INSERT OR REPLACE INTO memories
    (namespace, key, value, embedding, created_at, updated_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (ns_str, key, value_json, embedding_blob, created_at, updated_at, expires_at)
)
```

**Risk**: None - proper parameterization prevents SQL injection.

---

## Security Considerations for Users

1. **Retriever Persistence**: Only load `.pkl` files you created or from trusted sources. Pickle files can execute arbitrary code.

2. **API Keys**: Store LLM provider API keys in environment variables, not in code. Use `AI_INFRA_OPENAI_API_KEY`, `AI_INFRA_ANTHROPIC_API_KEY`, etc.

3. **MCP Servers**: When deploying MCP servers, use appropriate network isolation. Bind to localhost for development.

4. **Tool Execution**: Agent tool calls execute user-defined functions. Validate inputs in your tool implementations.

---

## Recommendations

1. **Migrate from Pickle** (P0): Replace pickle serialization in Retriever with JSON + numpy format to eliminate code execution risk.

2. **Input Validation**: When using tools with LLM agents, implement input validation in tool functions.

3. **Rate Limiting**: Implement rate limiting on MCP endpoints to prevent abuse.
