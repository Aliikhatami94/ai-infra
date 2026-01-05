# API Reference

This section provides API documentation for the ai-infra package.

For detailed class documentation with parameters and methods, see the [AI-Infra API Reference](/ai-infra/api) on nfrax.dev.

## Core Modules

### LLM

The `LLM` class provides a unified interface for interacting with language models.

See [LLM API Reference](/ai-infra/api/llm) for full documentation.

### Agent

The `Agent` class extends LLM with tool execution and agentic capabilities.

See [Agent API Reference](/ai-infra/api/agent) for full documentation.

### Graph

The `Graph` class provides workflow orchestration with nodes and edges.

See [Graph API Reference](/ai-infra/api/graph) for full documentation.

### MCP

Model Context Protocol client and server implementations.

- [MCPClient API Reference](/ai-infra/api/mcpclient) - Connect to MCP servers
- [MCPServer API Reference](/ai-infra/api/mcpserver) - Build MCP servers

### Embeddings

The `Embeddings` class provides text embedding generation and similarity search.

See [Embeddings API Reference](/ai-infra/api/embeddings) for full documentation.

### Retriever

The `Retriever` class provides RAG (Retrieval-Augmented Generation) capabilities.

See [Retriever API Reference](/ai-infra/api/retriever) for full documentation.

### ImageGen

The `ImageGen` class provides image generation capabilities.

See [ImageGen API Reference](/ai-infra/api/imagegen) for full documentation.

## Evaluation

### Evaluators

- [SemanticSimilarity](/ai-infra/api/semanticsimilarity) - Evaluate semantic similarity
- [ToolUsageEvaluator](/ai-infra/api/toolusageevaluator) - Evaluate tool usage
- [RAGFaithfulness](/ai-infra/api/ragfaithfulness) - Evaluate RAG faithfulness
