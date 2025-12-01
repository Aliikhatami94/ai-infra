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
| Memory | ❌ | Low | Zero config |
| SQLite | ✅ | Low | Local file |
| PostgreSQL | ✅ | High | pgvector extension |
| Chroma | ✅ | Medium | Local or server |
| Pinecone | ✅ | High | Managed cloud |
| Qdrant | ✅ | High | Cloud or self-hosted |
| FAISS | ❌ | High | Local, fast |

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

### Async Tool

```python
from ai_infra import create_retriever_tool_async

tool = create_retriever_tool_async(retriever, name="search")
agent = Agent(tools=[tool])
result = await agent.arun("Search for...")
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

## See Also

- [Embeddings](embeddings.md) - Embedding providers
- [VectorStore](vectorstore.md) - Low-level vector storage
- [Agent](../core/agents.md) - Use retriever with agents
