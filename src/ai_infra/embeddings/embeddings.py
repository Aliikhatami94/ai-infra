"""Provider-agnostic Embeddings class.

Uses LangChain embedding integrations internally but exposes a clean,
simple interface. Users never need to know about or import LangChain.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, cast

from ai_infra.providers import ProviderCapability, ProviderRegistry

# LangChain package/class mappings (implementation detail, not config)
_LANGCHAIN_MAPPINGS: dict[str, dict[str, str]] = {
    "openai": {"package": "langchain_openai", "class": "OpenAIEmbeddings"},
    "google_genai": {
        "package": "langchain_google_genai",
        "class": "GoogleGenerativeAIEmbeddings",
    },
    "voyage": {"package": "langchain_voyageai", "class": "VoyageAIEmbeddings"},
    "cohere": {"package": "langchain_cohere", "class": "CohereEmbeddings"},
    "huggingface": {
        "package": "langchain_huggingface",
        "class": "HuggingFaceEmbeddings",
    },
}

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES: dict[str, str] = {
    "google": "google_genai",
    "anthropic": "voyage",  # Anthropic recommends Voyage AI for embeddings
    "local": "huggingface",  # Local/free option
    "sentence-transformers": "huggingface",
}

# Default models per provider (for providers not in registry)
_DEFAULT_MODELS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}

# Provider priority for auto-detection (huggingface last as fallback)
_PROVIDER_PRIORITY = ["openai", "voyage", "google_genai", "cohere", "huggingface"]


def _get_provider_config() -> dict[str, dict[str, Any]]:
    """Build provider config from registry (for backwards compat)."""
    config = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.EMBEDDINGS):
        provider = ProviderRegistry.get(name)
        if provider and name in _LANGCHAIN_MAPPINGS:
            cap = provider.get_capability(ProviderCapability.EMBEDDINGS)
            config[name] = {
                "env_key": provider.env_var,
                "default_model": cap.default_model if cap else None,
                **_LANGCHAIN_MAPPINGS[name],
            }

    # Add huggingface (local, no API key needed)
    if "huggingface" not in config:
        config["huggingface"] = {
            "env_key": None,  # No API key needed
            "default_model": _DEFAULT_MODELS["huggingface"],
            **_LANGCHAIN_MAPPINGS["huggingface"],
        }

    # Add aliases
    for alias, target in _PROVIDER_ALIASES.items():
        if target in config:
            config[alias] = config[target].copy()
    return config


# Legacy constant for backwards compatibility
_PROVIDER_CONFIG = _get_provider_config()


class Embeddings:
    """Simple, provider-agnostic text embeddings.

    Generate embeddings from text using any supported provider (OpenAI, Google,
    Voyage, Cohere, Anthropic) or use free local embeddings with HuggingFace.

    Features:
        - Zero-config: Auto-detects provider, falls back to free local embeddings
        - Free option: Use HuggingFace/sentence-transformers (no API key needed)
        - Simple API: `embed()` for one, `embed_batch()` for many
        - Async support: `aembed()` and `aembed_batch()`
        - Similarity helper: `similarity()` for cosine similarity
        - Flexible: Override provider, model, dimensions

    Example:
        ```python
        from ai_infra import Embeddings

        # Auto-detect provider (falls back to free local if no API keys)
        embeddings = Embeddings()
        vector = embeddings.embed("Hello, world!")

        # Free local embeddings (no API key needed)
        embeddings = Embeddings(provider="huggingface")
        embeddings = Embeddings(provider="local")  # alias

        # Specific provider and model
        embeddings = Embeddings(provider="openai", model="text-embedding-3-large")

        # Batch embedding
        vectors = embeddings.embed_batch(["Hello", "World"])

        # Similarity
        score = embeddings.similarity("Hello", "Hi")
        ```

    Providers:
        - openai: OpenAI embeddings (requires OPENAI_API_KEY)
        - google/google_genai: Google embeddings (requires GOOGLE_API_KEY)
        - voyage: Voyage AI embeddings (requires VOYAGE_API_KEY)
        - cohere: Cohere embeddings (requires COHERE_API_KEY)
        - huggingface/local: Free local embeddings (no API key, runs on CPU)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize embeddings.

        Args:
            provider: Provider name (openai, google, voyage, cohere, anthropic,
                     huggingface/local). Auto-detects from environment if not
                     specified, falling back to huggingface (free, local).
            model: Model name. Uses provider default if not specified.
            dimensions: Embedding dimensions (OpenAI text-embedding-3 only).
            **kwargs: Additional provider-specific options.

        Raises:
            ValueError: If invalid provider specified.

        Example:
            ```python
            # Auto-detect (falls back to free local embeddings)
            embeddings = Embeddings()

            # Specific provider
            embeddings = Embeddings(provider="openai")

            # Free local embeddings (no API key needed)
            embeddings = Embeddings(provider="huggingface")
            embeddings = Embeddings(provider="local")  # alias

            # Custom model
            embeddings = Embeddings(
                provider="openai",
                model="text-embedding-3-large",
                dimensions=256  # Reduce dimensions for cost savings
            )
            ```
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._get_available_provider()
            if provider is None:
                # Get env vars from registry for error message
                env_vars = set()
                for name in ProviderRegistry.list_for_capability(ProviderCapability.EMBEDDINGS):
                    p = ProviderRegistry.get(name)
                    if p:
                        env_vars.add(p.env_var)
                raise ValueError(
                    "No embedding provider available. Set one of: " + ", ".join(sorted(env_vars))
                )

        provider = provider.lower()
        # Resolve aliases
        provider = _PROVIDER_ALIASES.get(provider, provider)

        if provider not in _PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {', '.join(_PROVIDER_CONFIG.keys())}"
            )

        config = _PROVIDER_CONFIG[provider]
        self._provider_name = provider
        self._model = model or config["default_model"]
        self._dimensions = dimensions

        # Initialize the underlying LangChain embeddings
        self._lc_embeddings = self._init_embeddings(config, dimensions, **kwargs)

    def _get_available_provider(self) -> str | None:
        """Find first available provider from environment.

        Returns the first configured provider, or 'huggingface' as a free fallback
        if no API keys are set.
        """
        # Use registry for provider discovery
        for provider_name in _PROVIDER_PRIORITY:
            if provider_name == "huggingface":
                # huggingface is always available (local, no API key)
                continue
            if ProviderRegistry.is_configured(provider_name):
                return provider_name

        # Fall back to huggingface (free, local)
        return "huggingface"

    def _init_embeddings(
        self,
        config: dict[str, Any],
        dimensions: int | None,
        **kwargs: Any,
    ) -> Any:
        """Initialize the LangChain embedding model."""
        import importlib

        package_name = config["package"]
        class_name = config["class"]

        try:
            module = importlib.import_module(package_name)
            embedding_cls = getattr(module, class_name)
        except ImportError as e:
            # Provide helpful install instructions
            install_cmd = package_name
            if self._provider_name == "huggingface":
                install_cmd = "langchain-huggingface sentence-transformers"
            raise ImportError(
                f"Embedding provider '{self._provider_name}' requires: pip install {install_cmd}"
            ) from e

        # Build kwargs based on provider
        if self._provider_name == "huggingface":
            # HuggingFace uses model_name, not model
            lc_kwargs: dict[str, Any] = {"model_name": self._model, **kwargs}
        else:
            lc_kwargs = {"model": self._model, **kwargs}

        # OpenAI supports custom dimensions
        if dimensions is not None and self._provider_name == "openai":
            lc_kwargs["dimensions"] = dimensions

        return embedding_cls(**lc_kwargs)

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def dimensions(self) -> int | None:
        """Get custom dimensions (if set)."""
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Example:
            ```python
            vector = embeddings.embed("Hello, world!")
            print(f"Dimensions: {len(vector)}")
            ```
        """
        return cast("list[float]", self._lc_embeddings.embed_query(text))

    async def aembed(self, text: str) -> list[float]:
        """Async embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Example:
            ```python
            vector = await embeddings.aembed("Hello, world!")
            ```
        """
        return cast("list[float]", await self._lc_embeddings.aembed_query(text))

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed multiple texts.

        Processes in batches for efficiency with large lists.

        Args:
            texts: List of texts to embed.
            batch_size: Texts per API call (default 100).

        Returns:
            List of embedding vectors.

        Example:
            ```python
            texts = ["Hello", "World", "!"]
            vectors = embeddings.embed_batch(texts)
            assert len(vectors) == len(texts)
            ```
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = cast("list[list[float]]", self._lc_embeddings.embed_documents(batch))
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def aembed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        max_concurrency: int = 5,
    ) -> list[list[float]]:
        """Async embed multiple texts with concurrency control.

        Args:
            texts: List of texts to embed.
            batch_size: Texts per API call (default 100).
            max_concurrency: Max concurrent API calls (default 5).

        Returns:
            List of embedding vectors.

        Example:
            ```python
            vectors = await embeddings.aembed_batch(texts)
            ```
        """
        if not texts:
            return []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return cast("list[list[float]]", await self._lc_embeddings.aembed_documents(batch))

        results = await asyncio.gather(*[process_batch(b) for b in batches])

        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        return all_embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between -1 and 1.
            1 = identical meaning, 0 = unrelated, -1 = opposite.

        Example:
            ```python
            score = embeddings.similarity("Hello", "Hi there")
            print(f"Similarity: {score:.2f}")  # e.g., 0.85
            ```
        """
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return self._cosine_similarity(v1, v2)

    async def asimilarity(self, text1: str, text2: str) -> float:
        """Async calculate cosine similarity between two texts."""
        v1, v2 = await asyncio.gather(
            self.aembed(text1),
            self.aembed(text2),
        )
        return self._cosine_similarity(v1, v2)

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all supported providers.

        Returns:
            List of provider names.

        Example:
            ```python
            print(Embeddings.list_providers())
            # ['openai', 'google', 'voyage', 'cohere', 'anthropic']
            ```
        """
        return list(_PROVIDER_CONFIG.keys())

    @classmethod
    def list_configured_providers(cls) -> list[str]:
        """List providers with API keys configured.

        Returns:
            List of available provider names.

        Example:
            ```python
            print(Embeddings.list_configured_providers())
            # ['openai']  # Only if OPENAI_API_KEY is set
            ```
        """
        available = []
        for provider, config in _PROVIDER_CONFIG.items():
            env_key = config["env_key"]
            # Provider is available if it has no env_key (local) or if the env_key is set
            if env_key is None or os.environ.get(env_key):
                available.append(provider)
        return available

    def __repr__(self) -> str:
        dims = f", dimensions={self._dimensions}" if self._dimensions else ""
        return f"Embeddings(provider={self._provider_name!r}, model={self._model!r}{dims})"
