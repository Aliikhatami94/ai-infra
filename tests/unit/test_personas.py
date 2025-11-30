"""Tests for personas - Agent.from_persona() and Persona class."""

import tempfile
from pathlib import Path

import pytest

from ai_infra.llm import Agent
from ai_infra.llm.personas import Persona, build_tool_filter


class TestPersona:
    """Tests for Persona dataclass."""

    def test_create_basic_persona(self):
        persona = Persona(
            name="test",
            prompt="You are a test assistant.",
        )

        assert persona.name == "test"
        assert persona.prompt == "You are a test assistant."
        assert persona.tools is None
        assert persona.deny is None
        assert persona.approve is None

    def test_create_full_persona(self):
        persona = Persona(
            name="analyst",
            prompt="You are a data analyst.",
            tools=["query_database", "create_chart"],
            deny=["delete_record"],
            approve=["send_email"],
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            temperature=0.3,
            max_tokens=4000,
        )

        assert persona.name == "analyst"
        assert persona.tools == ["query_database", "create_chart"]
        assert persona.deny == ["delete_record"]
        assert persona.approve == ["send_email"]
        assert persona.provider == "anthropic"
        assert persona.model_name == "claude-sonnet-4-20250514"
        assert persona.temperature == 0.3
        assert persona.max_tokens == 4000

    def test_from_dict(self):
        data = {
            "name": "analyst",
            "prompt": "You are a data analyst.",
            "tools": ["query_database"],
            "temperature": 0.5,
            "custom_field": "extra_value",  # Goes to metadata
        }

        persona = Persona.from_dict(data)

        assert persona.name == "analyst"
        assert persona.tools == ["query_database"]
        assert persona.temperature == 0.5
        assert persona.metadata == {"custom_field": "extra_value"}

    def test_to_dict(self):
        persona = Persona(
            name="test",
            prompt="Test prompt",
            tools=["tool1"],
            temperature=0.7,
        )

        data = persona.to_dict()

        assert data["name"] == "test"
        assert data["prompt"] == "Test prompt"
        assert data["tools"] == ["tool1"]
        assert data["temperature"] == 0.7
        # Empty fields should not be in dict
        assert "deny" not in data
        assert "approve" not in data


class TestPersonaYAML:
    """Tests for Persona YAML loading/saving."""

    def test_from_yaml(self):
        yaml_content = """
name: analyst
prompt: |
  You are a senior data analyst.
  Always verify data accuracy.

tools:
  - query_database
  - create_chart

deny:
  - delete_record

approve:
  - send_email

temperature: 0.3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            persona = Persona.from_yaml(f.name)

            assert persona.name == "analyst"
            assert "data analyst" in persona.prompt
            assert persona.tools == ["query_database", "create_chart"]
            assert persona.deny == ["delete_record"]
            assert persona.approve == ["send_email"]
            assert persona.temperature == 0.3

    def test_from_yaml_uses_filename_as_name(self):
        yaml_content = """
prompt: You are an assistant.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="myagent_"
        ) as f:
            f.write(yaml_content)
            f.flush()

            persona = Persona.from_yaml(f.name)

            # Should use filename stem as name
            assert "myagent" in persona.name

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Persona.from_yaml("/nonexistent/path/persona.yaml")

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            with pytest.raises(ValueError, match="Empty persona file"):
                Persona.from_yaml(f.name)

    def test_save_yaml(self):
        persona = Persona(
            name="test",
            prompt="Test prompt",
            tools=["tool1", "tool2"],
            temperature=0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saved.yaml"
            persona.save_yaml(path)

            # Load back and verify
            loaded = Persona.from_yaml(path)
            assert loaded.name == "test"
            assert loaded.prompt == "Test prompt"
            assert loaded.tools == ["tool1", "tool2"]
            assert loaded.temperature == 0.5


class TestBuildToolFilter:
    """Tests for build_tool_filter function."""

    def test_no_filter(self):
        filter_fn = build_tool_filter(None, None)
        assert filter_fn is None

    def test_whitelist_only(self):
        filter_fn = build_tool_filter(["allowed1", "allowed2"], None)

        assert filter_fn("allowed1") is True
        assert filter_fn("allowed2") is True
        assert filter_fn("other") is False

    def test_blacklist_only(self):
        filter_fn = build_tool_filter(None, ["denied1", "denied2"])

        assert filter_fn("denied1") is False
        assert filter_fn("denied2") is False
        assert filter_fn("other") is True

    def test_whitelist_and_blacklist(self):
        # Whitelist takes precedence, blacklist is additional restriction
        filter_fn = build_tool_filter(
            ["tool1", "tool2", "tool3"],  # Allowed
            ["tool2"],  # But tool2 is denied
        )

        assert filter_fn("tool1") is True
        assert filter_fn("tool2") is False  # Denied overrides allowed
        assert filter_fn("tool3") is True
        assert filter_fn("other") is False  # Not in whitelist


class TestAgentFromPersona:
    """Tests for Agent.from_persona() class method."""

    def test_from_persona_inline(self):
        agent = Agent.from_persona(
            name="analyst",
            prompt="You are a data analyst.",
            tools=["query_database"],
            deny=["delete_record"],
            approve=["send_email"],
        )

        assert agent._name == "analyst"
        assert agent._system == "You are a data analyst."
        assert agent._tool_filter is not None
        assert agent._persona is not None
        assert agent._persona.name == "analyst"

    def test_from_persona_with_provider_overrides(self):
        agent = Agent.from_persona(
            name="test",
            prompt="Test",
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            temperature=0.5,
        )

        assert agent._default_provider == "anthropic"
        assert agent._default_model_name == "claude-sonnet-4-20250514"
        assert agent._default_model_kwargs.get("temperature") == 0.5

    def test_from_persona_yaml_file(self):
        yaml_content = """
name: yaml_agent
prompt: You are from YAML.
tools:
  - yaml_tool
temperature: 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            agent = Agent.from_persona(f.name)

            assert agent._name == "yaml_agent"
            assert agent._system == "You are from YAML."
            assert agent._default_model_kwargs.get("temperature") == 0.2
            assert agent._tool_filter is not None
            assert agent._tool_filter("yaml_tool") is True
            assert agent._tool_filter("other") is False

    def test_from_persona_yaml_with_overrides(self):
        yaml_content = """
name: yaml_agent
prompt: YAML prompt
temperature: 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Override name and temperature
            agent = Agent.from_persona(
                f.name,
                name="overridden_name",
                temperature=0.8,
            )

            assert agent._name == "overridden_name"
            assert agent._default_model_kwargs.get("temperature") == 0.8

    def test_from_persona_approval_integration(self):
        agent = Agent.from_persona(
            name="test",
            prompt="Test",
            approve=["dangerous_tool"],
        )

        # Should have require_approval set
        assert agent._approval_config is not None
        assert agent._approval_config.require_approval == ["dangerous_tool"]

    def test_tool_filter_integration(self):
        agent = Agent.from_persona(
            name="test",
            prompt="Test",
            allowed_tools=["allowed_tool"],
            deny=["denied_tool"],
        )

        filter_fn = agent._tool_filter
        assert filter_fn is not None
        assert filter_fn("allowed_tool") is True
        assert filter_fn("denied_tool") is False
        assert filter_fn("other_tool") is False  # Not in whitelist

    def test_from_persona_minimal(self):
        # Just name, should work
        agent = Agent.from_persona(name="minimal")

        assert agent._name == "minimal"
        assert agent._system == ""
        assert agent._tool_filter is None  # No filtering

    def test_persona_metadata_preserved(self):
        yaml_content = """
name: meta_agent
prompt: Test
custom_key: custom_value
another_field: 123
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            agent = Agent.from_persona(f.name)

            assert agent._persona.metadata == {
                "custom_key": "custom_value",
                "another_field": 123,
            }
