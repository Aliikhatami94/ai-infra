# Retriever (RAG)

> Retrieval-Augmented Generation with multiple storage backends.

## Quick Start

```python
from ai_infra import Retriever

retriever = Retriever()

# Add documents
retriever.add("Python is a programming language")
retriever.add("JavaScript runs in browsers")

# Search
results = retriever.search("coding")
print(results[0].content)  # "Python is a programming language"
```

---

## Supported Backends

| Backend | Persistence | Scalability | Setup |
|---------|-------------|-------------|-------|
| Memory | [X] | Low | Zero config |
| SQLite | [OK] | Low | Local file |
| PostgreSQL | [OK] | High | pgvector extension |
| Chroma | [OK] | Medium | Local or server |
| Pinecone | [OK] | High | Managed cloud |
| Qdrant | [OK] | High | Cloud or self-hosted |
| FAISS | [X] | High | Local, fast |

---

## Basic Usage

### Add Text

```python
from ai_infra import Retriever

retriever = Retriever()

# Add text directly
retriever.add("Some important information")

# Add with metadata
retriever.add(
    "More information",
    metadata={"source": "manual", "date": "2024-01-01"}
)
```

### Search

```python
results = retriever.search("query", k=5)

for result in results:
    print(f"Content: {result.content}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

---

## File Loading

### Single File

```python
retriever = Retriever()

# Load various file types
retriever.add_file("document.pdf")
retriever.add_file("report.docx")
retriever.add_file("notes.txt")
retriever.add_file("data.csv")
retriever.add_file("config.json")
retriever.add_file("page.html")
retriever.add_file("readme.md")
```

### Directory

```python
# Load all supported files from directory
retriever.add_directory("./documents/")

# With pattern filter
retriever.add_directory("./documents/", pattern="*.pdf")

# Recursive
retriever.add_directory("./documents/", recursive=True)
```

---

## Remote Content Loading

Load content directly from GitHub repositories and URLs:

### From GitHub

```python
# Load files from a GitHub repository
await retriever.add_from_github(
    repo="owner/repo-name",
    paths=["README.md", "docs/guide.md"],  # Optional: specific files
    branch="main",  # Optional: default "main"
)

# Load entire docs folder
await retriever.add_from_github(
    repo="nfraxlab/svc-infra",
    paths=["docs/"],  # Directory path
)
```

Metadata automatically includes:
- `repo`: "owner/repo-name"
- `package`: "repo-name" (extracted)
- `path`: file path within repo
- `type`: "github"

### From URL

```python
# Load from any URL (raw GitHub, documentation sites, etc.)
await retriever.add_from_url("https://raw.githubusercontent.com/owner/repo/main/README.md")

# Multiple URLs
for url in urls:
    await retriever.add_from_url(url)
```

### Sync Wrappers

For non-async contexts:

```python
# Sync versions (creates event loop internally)
retriever.add_from_github_sync(repo="owner/repo", paths=["docs/"])
retriever.add_from_url_sync("https://example.com/doc.md")
```

> **Note:** Sync wrappers detect if called from async context and raise an error suggesting the async versions.

### Custom Loaders

Use any content loader from svc-infra:

```python
from svc_infra.loader import GitHubLoader

loader = GitHubLoader()

# Load documents then add to retriever
await retriever.add_from_loader(
    loader,
    repo="owner/repo",
    paths=["docs/"]
)
```

---

## Chunking

Control how documents are split:

```python
retriever = Retriever(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
)

# Or per-file
retriever.add_file(
    "large_document.pdf",
    chunk_size=500,
    chunk_overlap=100,
)
```

---

## Storage Backends

### Memory (Default)

```python
# No persistence, fast
retriever = Retriever(backend="memory")
```

### SQLite

```python
# Local file persistence
retriever = Retriever(
    backend="sqlite",
    db_path="./retriever.db"
)
```

### PostgreSQL + pgvector

```python
# Production-ready
retriever = Retriever(
    backend="postgres",
    connection_string="postgresql://user:pass@localhost:5432/db"
)
```

### Chroma

```python
# Easy vector DB
retriever = Retriever(
    backend="chroma",
    persist_directory="./chroma_db"
)

# Or connect to server
retriever = Retriever(
    backend="chroma",
    host="localhost",
    port=8000
)
```

### Pinecone

```python
# Managed cloud
retriever = Retriever(
    backend="pinecone",
    index_name="my-index",
    # Uses PINECONE_API_KEY env var
)
```

### Qdrant

```python
# Cloud or self-hosted
retriever = Retriever(
    backend="qdrant",
    url="http://localhost:6333",
    collection_name="my_collection"
)
```

### FAISS

```python
# High-performance local
retriever = Retriever(backend="faiss")
```

---

## Search Options

### Basic Search

```python
results = retriever.search("query")
```

### Limit Results

```python
results = retriever.search("query", k=10)
```

### Minimum Score

```python
results = retriever.search("query", min_score=0.7)
```

### Detailed Results

```python
results = retriever.search("query", detailed=True)
for r in results:
    print(f"Chunk: {r.chunk_index}")
    print(f"File: {r.metadata.get('source')}")
```

### SearchResult Object

Each search result provides:

```python
result = results[0]

# Core attributes
result.text       # Document content
result.score      # Similarity score (0.0-1.0)
result.metadata   # Dict of metadata
result.source     # Source file/URL
result.page       # Page number (if applicable)
result.chunk_index  # Chunk index in document

# Convenience properties (for remote content)
result.package      # Package name from metadata
result.repo         # Repository from metadata  
result.path         # File path from metadata
result.content_type # Content type from metadata

# Serialization
result.to_dict()  # JSON-serializable dict
```

**`to_dict()` output:**

```python
{
    "text": "Document content...",
    "score": 0.9523,  # Rounded to 4 decimals
    "source": "docs/guide.md",
    "page": None,
    "chunk_index": 0,
    "metadata": {"repo": "nfraxlab/svc-infra", ...}
}
```
```

---

## With Agent

Use retriever as an agent tool:

```python
from ai_infra import Agent, Retriever, create_retriever_tool

retriever = Retriever()
retriever.add_file("knowledge_base.pdf")

# Create tool from retriever
tool = create_retriever_tool(
    retriever,
    name="search_knowledge",
    description="Search the knowledge base for information"
)

agent = Agent(tools=[tool])
result = agent.run("What does the knowledge base say about X?")
```

### Structured Results

Get structured dictionary output instead of text:

```python
# Default: text output (formatted string)
tool = create_retriever_tool(retriever, name="search")

# Structured: dict output with results, query, count
tool = create_retriever_tool(
    retriever,
    name="search",
    structured=True,  # Returns dict instead of string
)

# Structured result format:
# {
#     "results": [
#         {"text": "...", "score": 0.95, "source": "doc.pdf", ...},
#         ...
#     ],
#     "query": "search query",
#     "count": 5
# }
```

### Async Tool

```python
from ai_infra import create_retriever_tool_async

tool = create_retriever_tool_async(retriever, name="search")
agent = Agent(tools=[tool])
result = await agent.arun("Search for...")

# Also supports structured parameter
tool = create_retriever_tool_async(retriever, name="search", structured=True)
```

---

## Async Usage

```python
import asyncio
from ai_infra import Retriever

async def main():
    retriever = Retriever()

    # Add async
    await retriever.aadd("Document content")

    # Search async
    results = await retriever.asearch("query")

asyncio.run(main())
```

---

## Configuration

```python
retriever = Retriever(
    # Embedding settings
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",

    # Chunking settings
    chunk_size=1000,
    chunk_overlap=200,

    # Backend settings
    backend="sqlite",
    db_path="./data.db",
)
```

### Auto-Configuration

When using known embedding models, the retriever can automatically determine the embedding dimension:

```python
# Auto-detect dimension from known models
retriever = Retriever(
    provider="openai",
    model="text-embedding-3-small",
    auto_configure=True,  # Default: False
)
# Automatically sets dimension=1536

# Supported models (28+ known dimensions):
# - openai: text-embedding-3-small (1536), text-embedding-3-large (3072), text-embedding-ada-002 (1536)
# - huggingface: sentence-transformers/all-MiniLM-L6-v2 (384), BAAI/bge-* (384-1024), etc.
# - cohere: embed-english-* (1024)
```

To see all known dimensions:

```python
from ai_infra.retriever import KNOWN_EMBEDDING_DIMENSIONS

print(KNOWN_EMBEDDING_DIMENSIONS)
# {"sentence-transformers/all-MiniLM-L6-v2": 384, ...}
```

---

## Persistence (Save/Load)

Save retriever state to disk and reload later without re-embedding:

```python
from ai_infra import Retriever

# Create and populate
retriever = Retriever()
retriever.add("./documents/")

# Save to disk
retriever.save("./cache/my_retriever.pkl")

# Later, load without re-embedding
retriever = Retriever.load("./cache/my_retriever.pkl")
results = retriever.search("query")  # Works immediately
```

### Auto-Persistence

Automatically save after each `add()` operation:

```python
retriever = Retriever(
    persist_path="./cache/retriever.pkl",
    auto_save=True,  # Default
)

# File created/updated automatically
retriever.add("New document")
# Saved automatically!

# On restart, loads automatically if file exists
retriever = Retriever(persist_path="./cache/retriever.pkl")
```

### Metadata Sidecar

When saving, a `.json` sidecar file is created with human-readable metadata:

```json
{
  "version": 1,
  "created_at": "2025-12-02T10:30:00Z",
  "backend": "memory",
  "embeddings_provider": "openai",
  "embeddings_model": "text-embedding-3-small",
  "similarity": "cosine",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "doc_count": 15,
  "chunk_count": 47
}
```

---

## Lazy Initialization

Defer embedding model loading until first use for faster server startup:

```python
from ai_infra import Retriever

# Fast startup - model NOT loaded yet
retriever = Retriever(
    provider="huggingface",
    lazy_init=True,
)

# ... server starts immediately ...

# Model loads on first add() or search()
retriever.add("First document")  # Model loads here
```

### Combined with Persistence

Best of both worlds - fast startup AND no re-embedding:

```python
retriever = Retriever(
    persist_path="./cache/retriever.pkl",
    lazy_init=True,
)

# If file exists: loads embeddings, model NOT loaded
# Model only loads if you call add() with new content
results = retriever.search("query")  # Uses cached embeddings
```

---

## Similarity Metrics

Choose the similarity function for search:

```python
from ai_infra import Retriever

# Cosine similarity (default) - best general choice
retriever = Retriever(similarity="cosine")

# Euclidean distance - converted to similarity score
retriever = Retriever(similarity="euclidean")

# Dot product - best for normalized embeddings
retriever = Retriever(similarity="dot_product")
```

### When to Use Each

| Metric | Best For | Score Range |
|--------|----------|-------------|
| `cosine` | General text similarity | [-1, 1] |
| `euclidean` | When magnitude matters | [0, 1] |
| `dot_product` | Normalized embeddings | Unbounded |

### Backend Support

All backends (memory, sqlite, postgres, etc.) support all three metrics.

---

## Tool Formatting Options

Control how retriever tool formats results for agents:

```python
from ai_infra import Retriever, create_retriever_tool

retriever = Retriever()
retriever.add("./docs/")

# Basic tool
tool = create_retriever_tool(retriever, name="search_docs")

# With formatting options
tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Search documentation",
    k=5,
    min_score=0.7,
    return_scores=True,    # Include similarity scores
    max_chars=2000,        # Truncate long results
    format="markdown",     # Output format
)
```

### Output Formats

**`format="text"` (default)**
```
First document content

---

Second document content
```

**`format="markdown"`**
```markdown
### Result 1
**Score:** 0.95
**Source:** doc1.txt

First document content

---

### Result 2
...
```

**`format="json"`**
```json
[
  {"text": "First document content", "score": 0.95, "source": "doc1.txt"},
  {"text": "Second document content", "score": 0.85, "source": "doc2.txt"}
]
```

---

## Production Deployment

Recommended configuration for production:

```python
from ai_infra import Retriever

retriever = Retriever(
    # Fast startup
    lazy_init=True,

    # Persist to disk (survives restarts)
    persist_path="./cache/embeddings.pkl",
    auto_save=True,

    # Embedding configuration
    provider="openai",
    model="text-embedding-3-small",

    # Similarity metric
    similarity="cosine",

    # Chunking
    chunk_size=500,
    chunk_overlap=50,
)
```

### FastAPI Integration

```python
from fastapi import FastAPI
from ai_infra import Retriever, create_retriever_tool_async

app = FastAPI()

# Initialize once at startup (fast with lazy_init)
retriever = Retriever(
    persist_path="./cache/docs.pkl",
    lazy_init=True,
)

@app.on_event("startup")
async def load_docs():
    # Only loads if cache doesn't exist
    if retriever.count == 0:
        retriever.add("./documents/")

@app.get("/search")
async def search(query: str):
    results = await retriever.asearch(query, k=5)
    return [{"text": r.text, "score": r.score} for r in results]
```

---

## Document Management

### Delete

```python
# Delete by ID
retriever.delete(doc_id)

# Delete by filter
retriever.delete_by_filter({"source": "old_file.pdf"})
```

### Clear All

```python
retriever.clear()
```

### Get Stats

```python
stats = retriever.stats()
print(f"Documents: {stats['document_count']}")
print(f"Chunks: {stats['chunk_count']}")
```

---

## Supported File Types

| Type | Extensions | Library |
|------|------------|---------|
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx |
| Text | `.txt` | Built-in |
| Markdown | `.md` | Built-in |
| CSV | `.csv` | pandas |
| JSON | `.json` | Built-in |
| HTML | `.html`, `.htm` | BeautifulSoup |

Install extras for file support:
```bash
pip install ai-infra[documents]
```

---

## Error Handling

```python
from ai_infra import Retriever
from ai_infra.errors import AIInfraError

try:
    retriever = Retriever()
    retriever.add_file("document.pdf")
    results = retriever.search("query")
except AIInfraError as e:
    print(f"Retriever error: {e}")
```

---

## Imports

Multiple import paths available:

```python
# Main retriever class
from ai_infra import Retriever

# Tool functions
from ai_infra import create_retriever_tool, create_retriever_tool_async

# Full module imports
from ai_infra.retriever import (
    Retriever,
    SearchResult,
    Chunk,
    create_retriever_tool,
    create_retriever_tool_async,
)

# Top-level aliased imports (avoid name conflicts)
from ai_infra import RetrieverSearchResult, RetrieverChunk

# These are the same classes:
assert RetrieverSearchResult is SearchResult
assert RetrieverChunk is Chunk
```
```

---

## See Also

- [Embeddings](embeddings.md) - Embedding providers
- [VectorStore](vectorstore.md) - Low-level vector storage
- [Agent](../core/agents.md) - Use retriever with agents
