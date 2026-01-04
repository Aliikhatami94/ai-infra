# Performance Baselines

> **Last Updated**: January 4, 2026
>
> **Environment**: macOS, Python 3.11, Apple Silicon

This document establishes baseline performance metrics for ai-infra to track regressions and improvements.

---

## Import Time

| Metric | Value | Notes |
|--------|-------|-------|
| Full package import | ~1,200ms | `import ai_infra` |

The import time is relatively high due to:
- Lazy loading of all providers (OpenAI, Anthropic, Google, xAI)
- NumPy and embedding dependencies
- MCP protocol initialization

**Recommendation**: Use selective imports for faster startup:
```python
# Slower (full package)
from ai_infra import LLM  # ~1200ms

# Faster (direct import)
from ai_infra.llm import LLM  # ~400ms
```

---

## Component Initialization

| Component | Init Time | Notes |
|-----------|-----------|-------|
| `LLM()` | ~1,200ms | Includes httpx client setup |
| `Embeddings()` | ~580ms | OpenAI embeddings client |
| `Agent(tools=[])` | <1ms | Lightweight, no API calls |
| `Retriever()` | ~75ms | In-memory vector store |

**Note**: These times include the `import ai_infra` overhead on first call. Subsequent initializations within the same process are faster.

---

## Streaming Performance

| Metric | Target | Notes |
|--------|--------|-------|
| First token latency | Provider-dependent | Measured from request to first streamed token |
| Token throughput | Provider-dependent | Measured in tokens/second |

**Streaming test** (requires API key):
```python
import time
from ai_infra import LLM

llm = LLM()
start = time.perf_counter()
first_token_time = None

for chunk in llm.stream("Count from 1 to 10"):
    if first_token_time is None:
        first_token_time = time.perf_counter() - start
    print(chunk, end="", flush=True)

total_time = time.perf_counter() - start
print(f"\nFirst token: {first_token_time*1000:.0f}ms")
print(f"Total time: {total_time*1000:.0f}ms")
```

---

## Embedding Throughput

| Metric | Value | Notes |
|--------|-------|-------|
| Single text embedding | Provider-dependent | ~100-200ms (network latency) |
| Batch embedding (100 texts) | Provider-dependent | ~500-1000ms |
| Retriever search (1000 docs) | <1ms | In-memory cosine similarity |

**Retriever performance** scales with document count:
- 1,000 documents: <1ms search
- 10,000 documents: ~5ms search
- 100,000 documents: ~50ms search

---

## Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Base import | ~100MB | Python + dependencies |
| LLM instance | +5MB | HTTP client |
| Retriever (empty) | +1MB | Base structures |
| Retriever (1000 docs, 1536 dim) | +12MB | ~12KB per document |
| Retriever (10000 docs, 1536 dim) | +120MB | Scales linearly |

---

## Recommendations

### For Low Latency Applications

1. **Pre-initialize components** during app startup
2. **Use streaming** for responsive UIs
3. **Pre-warm connections** with a dummy request

### For High Throughput Applications

1. **Batch embedding requests** (up to 100 texts per call)
2. **Use async methods** (`achat`, `astream`, `aembed`)
3. **Enable connection pooling** (default in httpx)

### For Memory-Constrained Environments

1. **Use selective imports** instead of `import ai_infra`
2. **Limit Retriever document count** or use external vector DB
3. **Clear unused instances** to release memory

---

## Running Your Own Benchmarks

```bash
# Install benchmark dependencies
pip install pytest-benchmark

# Run benchmark suite
pytest benchmarks/ --benchmark-only

# Compare to baseline
pytest benchmarks/ --benchmark-compare
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-04 | Initial baseline measurements |
