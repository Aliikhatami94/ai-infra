"""Tests for validation module."""

import pytest
from pydantic import BaseModel

from ai_infra.errors import ConfigurationError, ValidationError
from ai_infra.validation import (
    SUPPORTED_PROVIDERS,
    validate_config,
    validate_env_var,
    validate_json_output,
    validate_llm_params,
    validate_max_tokens,
    validate_messages,
    validate_output,
    validate_provider,
    validate_temperature,
)

# =============================================================================
# Test Models
# =============================================================================


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestValidateProvider:
    """Tests for validate_provider."""

    def test_valid_providers(self):
        """Test all supported providers are valid."""
        for provider in SUPPORTED_PROVIDERS:
            validate_provider(provider)  # Should not raise

    def test_invalid_provider(self):
        """Test invalid provider raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_provider("invalid_provider")

        assert "Unknown provider" in str(exc_info.value)
        assert exc_info.value.field == "provider"

    def test_error_includes_supported_providers(self):
        """Test error message includes supported providers."""
        with pytest.raises(ValidationError) as exc_info:
            validate_provider("not_a_provider")

        assert "openai" in str(exc_info.value.details.get("supported", []))


class TestValidateTemperature:
    """Tests for validate_temperature."""

    def test_valid_temperature(self):
        """Test valid temperatures."""
        validate_temperature(0.0)
        validate_temperature(0.7)
        validate_temperature(1.0)
        validate_temperature(2.0)

    def test_temperature_too_high(self):
        """Test temperature above max raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_temperature(5.0)

        assert "out of range" in str(exc_info.value)
        assert exc_info.value.field == "temperature"

    def test_temperature_too_low(self):
        """Test negative temperature raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_temperature(-0.5)

        assert "out of range" in str(exc_info.value)

    def test_provider_specific_range_anthropic(self):
        """Test Anthropic has 0-1 range."""
        # Valid for Anthropic (0-1)
        validate_temperature(0.5, provider="anthropic")

        # Invalid for Anthropic (max is 1.0)
        with pytest.raises(ValidationError):
            validate_temperature(1.5, provider="anthropic")

    def test_provider_specific_range_openai(self):
        """Test OpenAI allows 0-2."""
        validate_temperature(1.5, provider="openai")  # OK for OpenAI


class TestValidateMaxTokens:
    """Tests for validate_max_tokens."""

    def test_valid_max_tokens(self):
        """Test valid max_tokens values."""
        validate_max_tokens(1)
        validate_max_tokens(100)
        validate_max_tokens(4096)

    def test_zero_max_tokens(self):
        """Test zero raises error."""
        with pytest.raises(ValidationError):
            validate_max_tokens(0)

    def test_negative_max_tokens(self):
        """Test negative raises error."""
        with pytest.raises(ValidationError):
            validate_max_tokens(-10)


class TestValidateMessages:
    """Tests for validate_messages."""

    def test_valid_messages(self):
        """Test valid message list."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        validate_messages(messages)  # Should not raise

    def test_empty_messages(self):
        """Test empty list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([])

        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_message_type(self):
        """Test non-dict message raises error."""
        with pytest.raises(ValidationError):
            validate_messages(["not a dict"])  # type: ignore

    def test_missing_role(self):
        """Test message without role raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([{"content": "Hello"}])

        assert "missing 'role'" in str(exc_info.value)

    def test_invalid_role(self):
        """Test invalid role raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([{"role": "invalid", "content": "Hello"}])

        assert "Invalid role" in str(exc_info.value)

    def test_tool_call_message(self):
        """Test message with tool_calls is valid."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {}}]},
        ]
        validate_messages(messages)  # Should not raise


class TestValidateLLMParams:
    """Tests for validate_llm_params."""

    def test_all_valid(self):
        """Test all valid parameters."""
        validate_llm_params(
            provider="openai",
            temperature=0.7,
            max_tokens=1000,
        )

    def test_multiple_errors(self):
        """Test multiple validation errors are combined."""
        with pytest.raises(ValidationError) as exc_info:
            validate_llm_params(
                provider="invalid",
                temperature=5.0,
                max_tokens=-1,
            )

        # Should contain multiple errors
        errors = exc_info.value.details.get("errors", [])
        assert len(errors) >= 2


# =============================================================================
# Output Validation Tests
# =============================================================================


class TestValidateOutput:
    """Tests for validate_output."""

    def test_dict_input(self):
        """Test validating dict input."""
        result = validate_output({"name": "Alice", "age": 30}, Person)

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_already_correct_type(self):
        """Test already correct type passes through."""
        person = Person(name="Bob", age=25)
        result = validate_output(person, Person)

        assert result is person

    def test_json_string_input(self):
        """Test validating JSON string."""
        json_str = '{"name": "Charlie", "age": 35}'
        result = validate_output(json_str, Person)

        assert isinstance(result, Person)
        assert result.name == "Charlie"

    def test_invalid_data(self):
        """Test invalid data raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_output({"name": "Alice"}, Person)  # Missing age

        assert "Output validation failed" in str(exc_info.value)
        assert "Person" in str(exc_info.value)

    def test_strict_mode_false(self):
        """Test non-strict mode returns None on failure."""
        result = validate_output({"name": "Alice"}, Person, strict=False)
        assert result is None

    def test_wrong_types(self):
        """Test wrong types in data raises error."""
        with pytest.raises(ValidationError):
            validate_output({"name": 123, "age": "thirty"}, Person)


class TestValidateJsonOutput:
    """Tests for validate_json_output."""

    def test_valid_json(self):
        """Test valid JSON string."""
        json_str = '{"name": "Alice", "age": 30}'
        result = validate_json_output(json_str, Person)

        assert result.name == "Alice"
        assert result.age == 30

    def test_invalid_json_syntax(self):
        """Test invalid JSON syntax raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_json_output("{not valid json}", Person)

        assert "Invalid JSON" in str(exc_info.value)

    def test_valid_json_wrong_schema(self):
        """Test valid JSON but wrong schema raises error."""
        json_str = '{"street": "123 Main St"}'  # Missing city and zip

        with pytest.raises(ValidationError):
            validate_json_output(json_str, Address)


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config."""

    def test_all_required_present(self):
        """Test config with all required keys."""
        config = {"api_key": "sk-123", "model": "gpt-4"}
        validate_config(config, required=["api_key", "model"])  # OK

    def test_missing_required(self):
        """Test missing required key raises error."""
        config = {"api_key": "sk-123"}

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config, required=["api_key", "model"])

        assert "model" in str(exc_info.value)

    def test_extra_keys_allowed(self):
        """Test extra keys don't cause errors."""
        config = {"api_key": "sk-123", "extra": "value"}
        validate_config(config, required=["api_key"])  # OK


class TestValidateEnvVar:
    """Tests for validate_env_var."""

    def test_existing_env_var(self, monkeypatch):
        """Test existing env var returns value."""
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")

        result = validate_env_var("TEST_API_KEY")
        assert result == "sk-test-123"

    def test_missing_required_env_var(self, monkeypatch):
        """Test missing required env var raises error."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_env_var("NONEXISTENT_VAR", required=True)

        assert "NONEXISTENT_VAR" in str(exc_info.value)

    def test_missing_optional_env_var(self, monkeypatch):
        """Test missing optional env var returns None."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)

        result = validate_env_var("OPTIONAL_VAR", required=False)
        assert result is None
