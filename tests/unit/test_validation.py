"""Tests for validation module."""

import pytest
from pydantic import BaseModel

from ai_infra.errors import ConfigurationError, ValidationError
from ai_infra.validation import (
    SUPPORTED_PROVIDERS,
    validate_config,
    validate_env_var,
    validate_inputs,
    validate_json_output,
    validate_llm_params,
    validate_max_tokens,
    validate_messages,
    validate_output,
    validate_provider,
    validate_return,
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


# =============================================================================
# Decorator Tests
# =============================================================================


class TestValidateInputsDecorator:
    """Tests for @validate_inputs decorator."""

    def test_validates_dict_to_pydantic(self):
        """Test decorator converts dict to Pydantic model."""

        @validate_inputs
        def process_person(person: Person) -> str:
            return person.name

        result = process_person({"name": "Alice", "age": 30})
        assert result == "Alice"

    def test_passes_already_pydantic(self):
        """Test decorator passes through existing Pydantic model."""

        @validate_inputs
        def process_person(person: Person) -> str:
            return person.name

        alice = Person(name="Alice", age=30)
        result = process_person(alice)
        assert result == "Alice"

    def test_validates_keyword_args(self):
        """Test decorator validates keyword arguments."""

        @validate_inputs
        def process_person(person: Person) -> str:
            return person.name

        result = process_person(person={"name": "Bob", "age": 25})
        assert result == "Bob"

    def test_raises_on_invalid_input(self):
        """Test decorator raises ValidationError on invalid input."""

        @validate_inputs
        def process_person(person: Person) -> str:
            return person.name

        with pytest.raises(ValidationError) as exc_info:
            process_person({"name": "Alice"})  # Missing age

        assert exc_info.value.field == "person"

    def test_multiple_pydantic_args(self):
        """Test decorator with multiple Pydantic arguments."""

        @validate_inputs
        def greet(person: Person, address: Address) -> str:
            return f"{person.name} lives on {address.street}"

        result = greet(
            {"name": "Alice", "age": 30},
            {"street": "Main St", "city": "NYC", "zip_code": "10001"},
        )
        assert result == "Alice lives on Main St"

    def test_mixed_pydantic_and_regular_args(self):
        """Test decorator with mixed Pydantic and regular args."""

        @validate_inputs
        def greet(greeting: str, person: Person) -> str:
            return f"{greeting}, {person.name}!"

        result = greet("Hello", {"name": "Alice", "age": 30})
        assert result == "Hello, Alice!"

    def test_no_validation_needed(self):
        """Test decorator with no Pydantic hints is pass-through."""

        @validate_inputs
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3


class TestValidateReturnDecorator:
    """Tests for @validate_return decorator."""

    def test_validates_dict_return(self):
        """Test decorator validates dict return as Pydantic model."""

        @validate_return(Person)
        def get_person() -> Person:
            return {"name": "Alice", "age": 30}  # type: ignore

        result = get_person()
        assert isinstance(result, Person)
        assert result.name == "Alice"

    def test_passes_pydantic_return(self):
        """Test decorator passes through Pydantic model return."""

        @validate_return(Person)
        def get_person() -> Person:
            return Person(name="Bob", age=25)

        result = get_person()
        assert isinstance(result, Person)
        assert result.name == "Bob"

    def test_raises_on_invalid_return(self):
        """Test decorator raises ValidationError on invalid return."""

        @validate_return(Person)
        def get_person() -> Person:
            return {"name": "Alice"}  # type: ignore  # Missing age

        with pytest.raises(ValidationError) as exc_info:
            get_person()

        assert "Output validation failed" in str(exc_info.value)

    def test_with_function_args(self):
        """Test decorator works with function arguments."""

        @validate_return(Person)
        def create_person(name: str, age: int) -> Person:
            return {"name": name, "age": age}  # type: ignore

        result = create_person("Charlie", 35)
        assert isinstance(result, Person)
        assert result.name == "Charlie"
        assert result.age == 35

    def test_chain_with_validate_inputs(self):
        """Test both decorators can be chained."""

        @validate_return(Person)
        @validate_inputs
        def transform_person(person: Person) -> Person:
            return {"name": person.name.upper(), "age": person.age + 1}  # type: ignore

        result = transform_person({"name": "alice", "age": 30})
        assert isinstance(result, Person)
        assert result.name == "ALICE"
        assert result.age == 31
