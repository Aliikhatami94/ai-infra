"""Subagent model configuration.

Phase 7.4.1 of EXECUTOR_3.md: Provides configurable model settings per
subagent type for cost optimization and performance tuning.

Phase 16.5.3 of EXECUTOR_5.md: Added model inheritance from ExecutorConfig
so subagents inherit the main --model flag unless explicitly overridden.

Different agent types have different requirements:
- Coder/Debugger: Need powerful reasoning, use Sonnet with higher token limits
- Tester/Researcher: Simpler tasks, can use Haiku for cost savings
- Reviewer: Needs good analysis, uses Sonnet with moderate tokens

Model precedence (highest to lowest):
1. Explicit --subagent-model override (e.g., --subagent-model coder=gpt-4o)
2. Inherited model from --model flag (when inherit_model=True)
3. Agent-specific defaults (e.g., Sonnet for Coder, Haiku for Tester)

Example:
    ```python
    from ai_infra.executor.agents.config import SubAgentConfig
    from ai_infra.executor.agents.registry import SubAgentType

    # Use defaults
    config = SubAgentConfig()
    model_config = config.get_config(SubAgentType.TESTER)
    print(f"Tester uses: {model_config.model}")  # claude-haiku-4-20250514

    # Inherit from main model (Phase 16.5.3)
    config = SubAgentConfig(base_model="gpt-5-mini")
    model_config = config.get_config(SubAgentType.CODER)
    print(f"Coder uses: {model_config.model}")  # gpt-5-mini (inherited)

    # Override specific models
    config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.agents.registry import SubAgentType

__all__ = [
    "SubAgentConfig",
    "SubAgentModelConfig",
]


@dataclass
class SubAgentModelConfig:
    """Model configuration for a subagent type.

    Attributes:
        model: The model name to use (e.g., "claude-sonnet-4-20250514").
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature (0.0 = deterministic).
    """

    model: str
    max_tokens: int = 4096
    temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


# =============================================================================
# Default Model Configurations
# =============================================================================
# Cost-optimized defaults:
# - Powerful tasks (Coder, Debugger): Use Sonnet with high token limits
# - Simpler tasks (Tester, Researcher): Use Haiku for cost savings
# - Analysis tasks (Reviewer): Use Sonnet with moderate tokens

DEFAULT_CODER_CONFIG = SubAgentModelConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=8192,
    temperature=0.0,
)

DEFAULT_REVIEWER_CONFIG = SubAgentModelConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    temperature=0.0,
)

DEFAULT_TESTER_CONFIG = SubAgentModelConfig(
    model="claude-haiku-4-20250514",  # Cheaper for test generation
    max_tokens=4096,
    temperature=0.0,
)

DEFAULT_DEBUGGER_CONFIG = SubAgentModelConfig(
    model="claude-sonnet-4-20250514",  # Needs strong reasoning
    max_tokens=8192,
    temperature=0.0,
)

DEFAULT_RESEARCHER_CONFIG = SubAgentModelConfig(
    model="claude-haiku-4-20250514",  # Simple lookups
    max_tokens=2048,
    temperature=0.0,
)


def _get_default_models() -> dict[str, SubAgentModelConfig]:
    """Get default model configurations.

    Returns dict keyed by agent type string value for easier serialization.
    """
    return {
        "coder": DEFAULT_CODER_CONFIG,
        "reviewer": DEFAULT_REVIEWER_CONFIG,
        "tester": DEFAULT_TESTER_CONFIG,
        "debugger": DEFAULT_DEBUGGER_CONFIG,
        "researcher": DEFAULT_RESEARCHER_CONFIG,
    }


@dataclass
class SubAgentConfig:
    """Configuration for all subagents.

    Phase 16.5.3: Added model inheritance support so subagents can inherit
    the main --model flag from ExecutorConfig unless explicitly overridden.

    Provides model configuration per agent type with support for
    overrides via constructor or factory method.

    Model precedence (highest to lowest):
    1. Explicit model in `overrides` dict (from --subagent-model)
    2. `base_model` inheritance (from --model flag)
    3. Agent-specific defaults (Sonnet for Coder, Haiku for Tester)

    Attributes:
        models: Dict mapping agent type to model config (defaults + any overrides).
        overrides: Set of agent types that were explicitly overridden.
        inherit_model: Whether to inherit base_model for non-overridden agents.
        base_model: Model to inherit from (typically from ExecutorConfig.model).

    Example:
        ```python
        # Use defaults
        config = SubAgentConfig()

        # Inherit from main model (Phase 16.5.3)
        config = SubAgentConfig(base_model="gpt-5-mini")
        coder_config = config.get_config(SubAgentType.CODER)
        print(coder_config.model)  # "gpt-5-mini" (inherited)

        # Override specific models (takes priority over inheritance)
        config = SubAgentConfig.with_overrides(
            {"coder": "gpt-4o"},
            base_model="gpt-5-mini"
        )
        # coder uses gpt-4o (explicit), tester uses gpt-5-mini (inherited)
        ```
    """

    models: dict[str, SubAgentModelConfig] = field(default_factory=_get_default_models)
    # Phase 16.5.3: Track which agents were explicitly overridden
    overrides: set[str] = field(default_factory=set)
    # Phase 16.5.3: Model inheritance support
    inherit_model: bool = True
    base_model: str | None = None  # From ExecutorConfig.model

    def get_config(self, agent_type: SubAgentType | str) -> SubAgentModelConfig:
        """Get model config for agent type.

        Phase 16.5.3: Updated to support model inheritance.

        Priority:
        1. Explicit override (agent_type in self.overrides)
        2. base_model (when inherit_model=True and base_model set)
        3. Default config for agent type

        Args:
            agent_type: SubAgentType enum or string value.

        Returns:
            SubAgentModelConfig for the agent type.
        """
        # Convert enum to string if needed
        if hasattr(agent_type, "value"):
            type_key = agent_type.value
        else:
            type_key = str(agent_type).lower()

        # Get default config for this agent type
        default_config = _get_default_models().get(type_key, DEFAULT_CODER_CONFIG)

        # Check for explicit override (Phase 16.5.3: use overrides set)
        if type_key in self.overrides:
            return self.models.get(type_key, default_config)

        # Phase 16.5.3: Apply base_model inheritance if enabled
        if self.inherit_model and self.base_model:
            return SubAgentModelConfig(
                model=self.base_model,
                max_tokens=default_config.max_tokens,
                temperature=default_config.temperature,
            )

        # Fall back to default
        return self.models.get(type_key, default_config)

    @classmethod
    def with_overrides(
        cls,
        overrides: dict[str, str | SubAgentModelConfig],
        base_model: str | None = None,
        inherit_model: bool = True,
    ) -> SubAgentConfig:
        """Create config with model overrides.

        Phase 16.5.3: Added base_model and inherit_model parameters.

        Args:
            overrides: Dict mapping agent type to model name or full config.
                Keys are agent type strings: "coder", "tester", "debugger", etc.
                Values can be model names (str) or SubAgentModelConfig.
            base_model: Model to inherit from (typically from --model flag).
            inherit_model: Whether to inherit base_model for non-overridden agents.

        Returns:
            SubAgentConfig with overrides applied.

        Example:
            ```python
            # Override specific models only
            config = SubAgentConfig.with_overrides({
                "coder": "gpt-4o",
            })

            # Override + inherit from main model (Phase 16.5.3)
            config = SubAgentConfig.with_overrides(
                {"coder": "gpt-4o"},
                base_model="gpt-5-mini"
            )
            # coder uses gpt-4o, other agents use gpt-5-mini
            ```
        """
        models = _get_default_models()
        override_keys: set[str] = set()  # Phase 16.5.3: Track explicit overrides

        for agent_type, override in overrides.items():
            type_key = agent_type.lower()
            if type_key not in models:
                # Unknown agent type, skip
                continue

            override_keys.add(type_key)  # Mark as explicitly overridden

            if isinstance(override, str):
                # Just override the model name, keep other settings
                existing = models[type_key]
                models[type_key] = SubAgentModelConfig(
                    model=override,
                    max_tokens=existing.max_tokens,
                    temperature=existing.temperature,
                )
            elif isinstance(override, SubAgentModelConfig):
                models[type_key] = override

        return cls(
            models=models,
            overrides=override_keys,
            base_model=base_model,
            inherit_model=inherit_model,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Phase 16.5.3: Include inherit_model, base_model, and overrides fields.
        """
        return {
            "models": {agent_type: config.to_dict() for agent_type, config in self.models.items()},
            "overrides": list(self.overrides),  # Convert set to list for JSON
            "inherit_model": self.inherit_model,
            "base_model": self.base_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubAgentConfig:
        """Create from dictionary.

        Phase 16.5.3: Support inherit_model, base_model, and overrides fields.

        Args:
            data: Dict with models, inherit_model, base_model, and overrides keys.
                  Also supports legacy format with just agent_type -> config.

        Returns:
            SubAgentConfig instance.
        """
        # Support legacy format (direct agent_type -> config mapping)
        if "models" not in data and any(
            key in data for key in ["coder", "tester", "reviewer", "debugger", "researcher"]
        ):
            models = {}
            override_keys: set[str] = set()
            for agent_type, config_data in data.items():
                if isinstance(config_data, dict):
                    models[agent_type] = SubAgentModelConfig(
                        model=config_data.get("model", "claude-sonnet-4-20250514"),
                        max_tokens=config_data.get("max_tokens", 4096),
                        temperature=config_data.get("temperature", 0.0),
                    )
                    override_keys.add(agent_type)
            return cls(models=models, overrides=override_keys)

        # New format with explicit fields
        models_data = data.get("models", {})
        models = {}
        for agent_type, config_data in models_data.items():
            models[agent_type] = SubAgentModelConfig(
                model=config_data.get("model", "claude-sonnet-4-20250514"),
                max_tokens=config_data.get("max_tokens", 4096),
                temperature=config_data.get("temperature", 0.0),
            )

        # Restore overrides set from list
        overrides_list = data.get("overrides", [])
        override_keys = set(overrides_list) if overrides_list else set()

        return cls(
            models=models if models else _get_default_models(),
            overrides=override_keys,
            inherit_model=data.get("inherit_model", True),
            base_model=data.get("base_model"),
        )
