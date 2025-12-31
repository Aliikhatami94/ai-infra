# Embeddings

> Generate vector embeddings for text with multiple providers.

## Quick Start

```python
from ai_infra import Embeddings

embeddings = Embeddings()
vector = embeddings.embed("Hello, world!")
print(len(vector))  # 1536 (depends on model)
```

---

## Supported Providers

| Provider | Models | Dimensions |
|----------|--------|------------|
| OpenAI | text-embedding-3-small, text-embedding-3-large | 1536, 3072 |
| Voyage | voyage-3, voyage-3-lite | 1024 |
| Google | text-embedding-004 | 768 |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 | 1024 |

---

## Basic Usage

### Auto-Detect Provider

```python
from ai_infra import Embeddings

# Auto-detects from env vars (OpenAI -> Voyage -> Google -> Cohere)
embeddings = Embeddings()
vector = embeddings.embed("Some text")
```

### Explicit Provider

```python
embeddings = Embeddings(provider="voyage")
vector = embeddings.embed("Some text")
```

---

## Single Text Embedding

```python
from ai_infra import Embeddings

embeddings = Embeddings()

# Get embedding vector
vector = embeddings.embed("Hello, world!")
print(type(vector))  # list[float]
print(len(vector))   # 1536
```

---

## Batch Embedding

Embed multiple texts efficiently:

```python
texts = [
    "First document",
    "Second document",
    "Third document",
]

vectors = embeddings.embed_batch(texts)
print(len(vectors))     # 3
print(len(vectors[0]))  # 1536
```

---

## Provider-Specific

### OpenAI Embeddings

```python
from ai_infra import Embeddings

embeddings = Embeddings(provider="openai")

# Default model: text-embedding-3-small (1536 dims)
vector = embeddings.embed("Hello")

# Large model: text-embedding-3-large (3072 dims)
embeddings = Embeddings(provider="openai", model="text-embedding-3-large")
vector = embeddings.embed("Hello")
```

### Voyage Embeddings

```python
embeddings = Embeddings(provider="voyage")

# Default: voyage-3
vector = embeddings.embed("Hello")

# Lightweight: voyage-3-lite
embeddings = Embeddings(provider="voyage", model="voyage-3-lite")
```

### Google Embeddings

```python
embeddings = Embeddings(provider="google_genai")
vector = embeddings.embed("Hello")
```

### Cohere Embeddings

```python
embeddings = Embeddings(provider="cohere")

# English optimized
embeddings = Embeddings(provider="cohere", model="embed-english-v3.0")

# Multilingual
embeddings = Embeddings(provider="cohere", model="embed-multilingual-v3.0")
```

---

## Async Usage

```python
import asyncio
from ai_infra import Embeddings

async def main():
    embeddings = Embeddings()

    # Single embed
    vector = await embeddings.aembed("Hello")

    # Batch embed
    vectors = await embeddings.aembed_batch(["Hello", "World"])

asyncio.run(main())
```

---

## Model Discovery

```python
from ai_infra import Embeddings

# List embedding providers
providers = Embeddings.list_providers()
# ['openai', 'voyage', 'google_genai', 'cohere']

# List models for a provider
models = Embeddings.list_models("openai")
# ['text-embedding-3-small', 'text-embedding-3-large']
```

---

## Configuration

```python
embeddings = Embeddings(
    provider="openai",
    model="text-embedding-3-small",
    # Optional: reduce dimensions (OpenAI only)
    dimensions=512,
)
```

---

## Error Handling

```python
from ai_infra import Embeddings
from ai_infra.errors import AIInfraError, ProviderError

try:
    embeddings = Embeddings()
    vector = embeddings.embed("Hello")
except ProviderError as e:
    print(f"Provider error: {e}")
except AIInfraError as e:
    print(f"Embedding error: {e}")
```

---

## Use Cases

### Semantic Search

```python
from ai_infra import Embeddings
import numpy as np

embeddings = Embeddings()

# Index documents
documents = ["Python is great", "JavaScript is popular", "Rust is fast"]
doc_vectors = embeddings.embed_batch(documents)

# Search
query = "programming languages"
query_vector = embeddings.embed(query)

# Find most similar (cosine similarity)
similarities = [
    np.dot(query_vector, doc_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vec))
    for doc_vec in doc_vectors
]
best_idx = np.argmax(similarities)
print(f"Best match: {documents[best_idx]}")
```

### Clustering

```python
from sklearn.cluster import KMeans

vectors = embeddings.embed_batch(texts)
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(vectors)
```

---

## See Also

- [VectorStore](vectorstore.md) - Store and search embeddings
- [Retriever](retriever.md) - RAG with embeddings
- [Providers](../core/providers.md) - Provider configuration
