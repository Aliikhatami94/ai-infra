"""Tests for Phase 16.5.3: Subagent Model Inheritance.

Tests that subagents inherit the main --model flag from ExecutorConfig
unless explicitly overridden via --subagent-model.
"""

from __future__ import annotations

from ai_infra.executor.agents.config import (
    DEFAULT_CODER_CONFIG,
    DEFAULT_TESTER_CONFIG,
    SubAgentConfig,
)
from ai_infra.executor.agents.registry import SubAgentType

# =============================================================================
# SubAgentConfig Model Inheritance Tests
# =============================================================================


class TestSubAgentConfigInheritance:
    """Tests for Phase 16.5.3: Model inheritance from base_model."""

    def test_default_config_uses_defaults(self) -> None:
        """Config with no base_model uses agent-specific defaults."""
        config = SubAgentConfig()

        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == DEFAULT_CODER_CONFIG.model

        tester_config = config.get_config(SubAgentType.TESTER)
        assert tester_config.model == DEFAULT_TESTER_CONFIG.model

    def test_base_model_inherited_by_all_agents(self) -> None:
        """base_model should be inherited by all agent types."""
        config = SubAgentConfig(base_model="gpt-5-mini")

        # All agents should use the base model
        for agent_type in SubAgentType:
            agent_config = config.get_config(agent_type)
            assert agent_config.model == "gpt-5-mini", f"{agent_type} should inherit base_model"

    def test_inherit_model_false_disables_inheritance(self) -> None:
        """inherit_model=False should use defaults even with base_model set."""
        config = SubAgentConfig(base_model="gpt-5-mini", inherit_model=False)

        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == DEFAULT_CODER_CONFIG.model

        tester_config = config.get_config(SubAgentType.TESTER)
        assert tester_config.model == DEFAULT_TESTER_CONFIG.model

    def test_base_model_preserves_max_tokens(self) -> None:
        """Inheriting base_model should preserve agent-specific max_tokens."""
        config = SubAgentConfig(base_model="gpt-5-mini")

        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == "gpt-5-mini"
        assert coder_config.max_tokens == DEFAULT_CODER_CONFIG.max_tokens  # 8192

        tester_config = config.get_config(SubAgentType.TESTER)
        assert tester_config.model == "gpt-5-mini"
        assert tester_config.max_tokens == DEFAULT_TESTER_CONFIG.max_tokens  # 4096

    def test_explicit_override_takes_priority(self) -> None:
        """Explicit --subagent-model override takes priority over base_model."""
        config = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o"},
            base_model="gpt-5-mini",
        )

        # Coder uses explicit override
        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == "gpt-4o"

        # Other agents inherit base_model
        tester_config = config.get_config(SubAgentType.TESTER)
        assert tester_config.model == "gpt-5-mini"

        reviewer_config = config.get_config(SubAgentType.REVIEWER)
        assert reviewer_config.model == "gpt-5-mini"

    def test_multiple_overrides_with_base_model(self) -> None:
        """Multiple overrides should work with base_model inheritance."""
        config = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o", "debugger": "claude-sonnet-4-20250514"},
            base_model="gpt-5-mini",
        )

        # Explicit overrides
        assert config.get_config(SubAgentType.CODER).model == "gpt-4o"
        assert config.get_config(SubAgentType.DEBUGGER).model == "claude-sonnet-4-20250514"

        # Inherited from base_model
        assert config.get_config(SubAgentType.TESTER).model == "gpt-5-mini"
        assert config.get_config(SubAgentType.REVIEWER).model == "gpt-5-mini"
        assert config.get_config(SubAgentType.RESEARCHER).model == "gpt-5-mini"


class TestSubAgentConfigSerialization:
    """Tests for SubAgentConfig serialization with inheritance fields."""

    def test_to_dict_includes_inheritance_fields(self) -> None:
        """to_dict should include inherit_model and base_model."""
        config = SubAgentConfig(base_model="gpt-5-mini", inherit_model=True)
        d = config.to_dict()

        assert d["inherit_model"] is True
        assert d["base_model"] == "gpt-5-mini"
        assert "models" in d

    def test_from_dict_restores_inheritance_fields(self) -> None:
        """from_dict should restore inherit_model and base_model."""
        data = {
            "models": {},
            "inherit_model": True,
            "base_model": "gpt-5-mini",
        }
        config = SubAgentConfig.from_dict(data)

        assert config.inherit_model is True
        assert config.base_model == "gpt-5-mini"

    def test_from_dict_legacy_format(self) -> None:
        """from_dict should support legacy format without inheritance fields."""
        # Legacy format: direct agent_type -> config mapping
        data = {
            "coder": {"model": "gpt-4o", "max_tokens": 8192, "temperature": 0.0},
            "tester": {"model": "gpt-4o-mini", "max_tokens": 4096, "temperature": 0.0},
        }
        config = SubAgentConfig.from_dict(data)

        assert config.get_config("coder").model == "gpt-4o"
        assert config.get_config("tester").model == "gpt-4o-mini"

    def test_roundtrip_serialization(self) -> None:
        """Config should survive serialization roundtrip."""
        original = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o"},
            base_model="gpt-5-mini",
        )

        data = original.to_dict()
        restored = SubAgentConfig.from_dict(data)

        assert restored.base_model == original.base_model
        assert restored.inherit_model == original.inherit_model

        # Check that configs match
        for agent_type in SubAgentType:
            original_cfg = original.get_config(agent_type)
            restored_cfg = restored.get_config(agent_type)
            assert original_cfg.model == restored_cfg.model, f"Mismatch for {agent_type}"


class TestModelPrecedence:
    """Tests for model precedence: override > base_model > defaults."""

    def test_precedence_explicit_override(self) -> None:
        """Explicit override should take highest priority."""
        config = SubAgentConfig.with_overrides(
            {"coder": "explicit-model"},
            base_model="base-model",
        )

        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == "explicit-model"

    def test_precedence_base_model(self) -> None:
        """base_model should take priority over defaults."""
        config = SubAgentConfig(base_model="base-model")

        tester_config = config.get_config(SubAgentType.TESTER)
        # Without base_model, would be claude-haiku-4-20250514
        assert tester_config.model == "base-model"

    def test_precedence_defaults(self) -> None:
        """Defaults should be used when no override or base_model."""
        config = SubAgentConfig()

        coder_config = config.get_config(SubAgentType.CODER)
        assert coder_config.model == DEFAULT_CODER_CONFIG.model

        tester_config = config.get_config(SubAgentType.TESTER)
        assert tester_config.model == DEFAULT_TESTER_CONFIG.model

    def test_full_precedence_chain(self) -> None:
        """Test full precedence chain with mixed config."""
        config = SubAgentConfig.with_overrides(
            {"coder": "explicit-coder-model"},  # Override for coder only
            base_model="inherited-model",  # Others inherit this
        )

        # Explicit override
        assert config.get_config(SubAgentType.CODER).model == "explicit-coder-model"

        # Inherited from base_model
        assert config.get_config(SubAgentType.TESTER).model == "inherited-model"
        assert config.get_config(SubAgentType.REVIEWER).model == "inherited-model"
        assert config.get_config(SubAgentType.DEBUGGER).model == "inherited-model"
        assert config.get_config(SubAgentType.RESEARCHER).model == "inherited-model"


class TestSubAgentConfigFactory:
    """Tests for SubAgentConfig.with_overrides factory method."""

    def test_with_overrides_no_base_model(self) -> None:
        """with_overrides without base_model uses defaults for non-overridden."""
        config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})

        assert config.get_config(SubAgentType.CODER).model == "gpt-4o"
        # Without base_model, tester uses its default
        assert config.get_config(SubAgentType.TESTER).model == DEFAULT_TESTER_CONFIG.model

    def test_with_overrides_with_base_model(self) -> None:
        """with_overrides with base_model inherits for non-overridden."""
        config = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o"},
            base_model="gpt-5-mini",
        )

        assert config.get_config(SubAgentType.CODER).model == "gpt-4o"
        # With base_model, tester inherits it
        assert config.get_config(SubAgentType.TESTER).model == "gpt-5-mini"

    def test_with_overrides_empty_overrides(self) -> None:
        """with_overrides with empty dict and base_model uses inheritance."""
        config = SubAgentConfig.with_overrides({}, base_model="gpt-5-mini")

        # All agents should inherit base_model
        for agent_type in SubAgentType:
            assert config.get_config(agent_type).model == "gpt-5-mini"

    def test_with_overrides_inherit_model_false(self) -> None:
        """with_overrides with inherit_model=False uses defaults."""
        config = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o"},
            base_model="gpt-5-mini",
            inherit_model=False,
        )

        # Explicit override works
        assert config.get_config(SubAgentType.CODER).model == "gpt-4o"
        # But tester falls back to default, not base_model
        assert config.get_config(SubAgentType.TESTER).model == DEFAULT_TESTER_CONFIG.model
