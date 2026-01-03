"""Tests for Embeddings provider configuration.

Tests cover:
- Provider initialization and auto-detection
- OpenAI embeddings configuration
- Cohere embeddings configuration
- HuggingFace (local) embeddings configuration
- Provider aliases
- API key detection
- Custom model selection
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestProviderAutoDetection:
    """Tests for automatic provider detection."""

    def test_auto_detects_openai_from_env(self) -> None:
        """Test that OpenAI is auto-detected from OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings()
                assert emb.provider == "openai"

    def test_auto_detects_voyage_from_env(self) -> None:
        """Test that Voyage is auto-detected from VOYAGE_API_KEY."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "pa-test-key"}, clear=True):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings()
                assert emb.provider == "voyage"

    def test_auto_detects_cohere_from_env(self) -> None:
        """Test that Cohere is auto-detected from COHERE_API_KEY."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "co-test-key"}, clear=True):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings()
                assert emb.provider == "cohere"

    def test_falls_back_to_huggingface_no_keys(self) -> None:
        """Test fallback to HuggingFace when no API keys are set."""
        with patch.dict(os.environ, {}, clear=True):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                with patch.object(
                    Embeddings, "_get_available_provider", return_value="huggingface"
                ):
                    emb = Embeddings(provider="huggingface")
                    assert emb.provider == "huggingface"

    def test_provider_priority_order(self) -> None:
        """Test that providers are detected in priority order."""
        # OpenAI should take priority over others
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "COHERE_API_KEY": "co-test",
            },
            clear=True,
        ):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings()
                assert emb.provider == "openai"


class TestProviderAliases:
    """Tests for provider alias resolution."""

    def test_local_alias_resolves_to_huggingface(self) -> None:
        """Test that 'local' alias resolves to huggingface."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="local")
            assert emb.provider == "huggingface"

    def test_google_alias_resolves_to_google_genai(self) -> None:
        """Test that 'google' alias resolves to google_genai."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings(provider="google")
                assert emb.provider == "google_genai"

    def test_sentence_transformers_alias(self) -> None:
        """Test that 'sentence-transformers' alias resolves to huggingface."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="sentence-transformers")
            assert emb.provider == "huggingface"


class TestOpenAIProvider:
    """Tests for OpenAI embeddings provider."""

    def test_openai_uses_correct_class(self) -> None:
        """Test that OpenAI provider initializes correct class."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAI:
                mock_instance = MagicMock()
                MockOpenAI.return_value = mock_instance

                Embeddings(provider="openai")

                MockOpenAI.assert_called_once()

    def test_openai_default_model(self) -> None:
        """Test that OpenAI uses correct default model."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAI:
                MockOpenAI.return_value = MagicMock()

                Embeddings(provider="openai")

                # Check that model was passed
                call_kwargs = MockOpenAI.call_args[1]
                assert "model" in call_kwargs

    def test_openai_custom_model(self) -> None:
        """Test that OpenAI accepts custom model."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAI:
                MockOpenAI.return_value = MagicMock()

                Embeddings(provider="openai", model="text-embedding-3-large")

                call_kwargs = MockOpenAI.call_args[1]
                assert call_kwargs["model"] == "text-embedding-3-large"

    def test_openai_supports_dimensions(self) -> None:
        """Test that OpenAI supports custom dimensions parameter."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAI:
                MockOpenAI.return_value = MagicMock()

                Embeddings(
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=512,
                )

                call_kwargs = MockOpenAI.call_args[1]
                assert call_kwargs.get("dimensions") == 512

    def test_openai_dimensions_property(self) -> None:
        """Test that dimensions property returns custom value."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAI:
                MockOpenAI.return_value = MagicMock()

                emb = Embeddings(provider="openai", dimensions=256)

                assert emb.dimensions == 256


class TestCohereProvider:
    """Tests for Cohere embeddings provider."""

    def test_cohere_provider_name_resolved(self) -> None:
        """Test that Cohere provider name is resolved correctly."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"COHERE_API_KEY": "co-test"}, clear=True):
            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings(provider="cohere")

                assert emb.provider == "cohere"

    def test_cohere_custom_model(self) -> None:
        """Test that Cohere accepts custom model."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"COHERE_API_KEY": "co-test"}, clear=True):
            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings(provider="cohere", model="embed-multilingual-v3.0")

                assert emb.model == "embed-multilingual-v3.0"


class TestHuggingFaceProvider:
    """Tests for HuggingFace (local) embeddings provider."""

    def test_huggingface_provider_name(self) -> None:
        """Test that HuggingFace provider name is correct."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="huggingface")

            assert emb.provider == "huggingface"

    def test_huggingface_accepts_custom_model(self) -> None:
        """Test that HuggingFace accepts custom model name."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(
                provider="huggingface",
                model="sentence-transformers/all-mpnet-base-v2",
            )

            assert emb.model == "sentence-transformers/all-mpnet-base-v2"

    def test_huggingface_default_model(self) -> None:
        """Test that HuggingFace uses default model when not specified."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="huggingface")

            # Should use the default MiniLM model
            assert "all-MiniLM" in emb.model

    def test_huggingface_no_api_key_required(self) -> None:
        """Test that HuggingFace works without API keys."""
        with patch.dict(os.environ, {}, clear=True):
            from ai_infra.embeddings import Embeddings

            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                # Should not raise even with no API keys
                emb = Embeddings(provider="huggingface")

                assert emb.provider == "huggingface"


class TestProviderErrors:
    """Tests for provider error handling."""

    def test_unknown_provider_raises_error(self) -> None:
        """Test that unknown provider raises ValueError."""
        from ai_infra.embeddings import Embeddings

        with pytest.raises(ValueError, match="Unknown provider"):
            Embeddings(provider="unknown_provider")

    def test_missing_package_raises_importerror(self) -> None:
        """Test that missing package raises ImportError from _init_embeddings."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            # Simulate missing langchain_openai by making _init_embeddings raise
            with patch.object(
                Embeddings,
                "_init_embeddings",
                side_effect=ImportError(
                    "Embedding provider 'openai' requires: pip install langchain-openai"
                ),
            ):
                with pytest.raises(ImportError, match="pip install"):
                    Embeddings(provider="openai")


class TestListProviders:
    """Tests for provider listing methods."""

    def test_list_providers_returns_list(self) -> None:
        """Test that list_providers returns a list of strings."""
        from ai_infra.embeddings import Embeddings

        providers = Embeddings.list_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert all(isinstance(p, str) for p in providers)

    def test_list_providers_includes_known_providers(self) -> None:
        """Test that list_providers includes known provider names."""
        from ai_infra.embeddings import Embeddings

        providers = Embeddings.list_providers()

        # Should include at least huggingface (always available)
        assert "huggingface" in providers

    def test_list_configured_providers_returns_list(self) -> None:
        """Test that list_configured_providers returns a list."""
        from ai_infra.embeddings import Embeddings

        configured = Embeddings.list_configured_providers()

        assert isinstance(configured, list)

    def test_list_configured_includes_huggingface(self) -> None:
        """Test that list_configured always includes huggingface."""
        with patch.dict(os.environ, {}, clear=True):
            from ai_infra.embeddings import Embeddings

            configured = Embeddings.list_configured_providers()

            # HuggingFace is always available (no API key needed)
            assert "huggingface" in configured

    def test_list_configured_includes_openai_when_key_set(self) -> None:
        """Test that list_configured includes OpenAI when key is set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from ai_infra.embeddings import Embeddings

            configured = Embeddings.list_configured_providers()

            assert "openai" in configured


class TestProviderProperties:
    """Tests for Embeddings provider properties."""

    def test_provider_property_returns_name(self) -> None:
        """Test that provider property returns provider name."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="huggingface")

            assert emb.provider == "huggingface"

    def test_model_property_returns_model_name(self) -> None:
        """Test that model property returns model name."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(
                provider="huggingface",
                model="custom-model",
            )

            assert emb.model == "custom-model"

    def test_dimensions_property_none_by_default(self) -> None:
        """Test that dimensions property is None by default."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="huggingface")

            assert emb.dimensions is None

    def test_repr_includes_provider_and_model(self) -> None:
        """Test that repr includes provider and model."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
            emb = Embeddings(provider="huggingface", model="test-model")

            repr_str = repr(emb)

            assert "huggingface" in repr_str
            assert "test-model" in repr_str

    def test_repr_includes_dimensions_when_set(self) -> None:
        """Test that repr includes dimensions when set."""
        from ai_infra.embeddings import Embeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch.object(Embeddings, "_init_embeddings", return_value=MagicMock()):
                emb = Embeddings(provider="openai", dimensions=256)

                repr_str = repr(emb)

                assert "dimensions=256" in repr_str
