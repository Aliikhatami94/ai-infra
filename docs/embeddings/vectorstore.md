# VectorStore

> Store documents with embeddings for similarity search.

## Quick Start

```python
from ai_infra import VectorStore

store = VectorStore()
store.add("doc1", "Python is a programming language")
store.add("doc2", "JavaScript runs in browsers")

results = store.search("coding languages")
print(results[0].content)  # "Python is a programming language"
```

---

## Basic Usage

### Create Store

```python
from ai_infra import VectorStore

# In-memory (default)
store = VectorStore()

# With custom embeddings provider
store = VectorStore(embedding_provider="voyage")
```

### Add Documents

```python
# Add single document
store.add("id1", "Document content here")

# Add with metadata
store.add("id2", "Another document", metadata={"source": "file.txt"})

# Add multiple
store.add_batch([
    ("id3", "Third document", {"type": "article"}),
    ("id4", "Fourth document", {"type": "blog"}),
])
```

### Search

```python
# Basic search
results = store.search("query text", k=5)

for result in results:
    print(f"ID: {result.id}")
    print(f"Content: {result.content}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

---

## Search Options

### Limit Results

```python
# Top 3 results
results = store.search("query", k=3)
```

### Minimum Score

```python
# Only results with score >= 0.7
results = store.search("query", min_score=0.7)
```

### Filter by Metadata

```python
# Only search documents with specific metadata
results = store.search(
    "query",
    filter={"type": "article"}
)
```

---

## Search Results

```python
from ai_infra import SearchResult

results = store.search("query")

for result in results:
    # Result properties
    result.id        # Document ID
    result.content   # Document text
    result.score     # Similarity score (0-1)
    result.metadata  # Document metadata dict
    result.vector    # Embedding vector (if requested)
```

### Get Detailed Results

```python
results = store.search("query", detailed=True)

for result in results:
    print(f"Vector dims: {len(result.vector)}")
```

---

## Document Management

### Get Document

```python
doc = store.get("id1")
print(doc.content)
```

### Delete Document

```python
store.delete("id1")
```

### Update Document

```python
store.update("id1", "Updated content", metadata={"updated": True})
```

### List Documents

```python
docs = store.list()
print(f"Total documents: {len(docs)}")
```

---

## Persistence

### Save to File

```python
store = VectorStore()
store.add("id1", "Document")

# Save
store.save("vectors.pkl")

# Load
store = VectorStore.load("vectors.pkl")
```

---

## Async Usage

```python
import asyncio
from ai_infra import VectorStore

async def main():
    store = VectorStore()

    # Add async
    await store.aadd("id1", "Document")

    # Search async
    results = await store.asearch("query")

asyncio.run(main())
```

---

## Configuration

```python
store = VectorStore(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

---

## Use Cases

### Document Search

```python
store = VectorStore()

# Index documents
for doc in documents:
    store.add(doc.id, doc.text, metadata={"title": doc.title})

# Search
results = store.search(user_query)
```

### FAQ Bot

```python
store = VectorStore()

# Add FAQ pairs
faqs = [
    ("faq1", "How do I reset my password?", {"answer": "Go to settings..."}),
    ("faq2", "What payment methods?", {"answer": "We accept..."}),
]
store.add_batch(faqs)

# Find matching FAQ
results = store.search(user_question, k=1)
print(results[0].metadata["answer"])
```

---

## See Also

- [Embeddings](embeddings.md) - Generate embeddings
- [Retriever](retriever.md) - Full RAG pipeline
- [Providers](../core/providers.md) - Embedding providers
