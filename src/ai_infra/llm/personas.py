"""
Agent Personas: Config-driven agent behavior.

This module provides the Persona dataclass and utilities for creating agents
from YAML configuration files.

Example:
    ```python
    from ai_infra import Agent

    # From YAML file
    agent = Agent.from_persona("personas/analyst.yaml")

    # Or inline
    agent = Agent.from_persona(
        name="analyst",
        prompt="You are a data analyst. Be precise and data-driven.",
        tools=["query_database", "create_chart"],
        deny=["delete_record", "drop_table"],
        approve=["send_email"],
        temperature=0.3,
    )
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Persona:
    """
    Agent persona configuration.

    Defines agent behavior, allowed tools, and safety constraints
    in a declarative format.

    Attributes:
        name: Persona identifier for logging/debugging
        prompt: System prompt defining agent behavior
        tools: List of allowed tool names (whitelist)
        deny: List of blocked tool names (blacklist)
        approve: List of tools requiring HITL approval
        provider: Override default LLM provider
        model_name: Override default model
        temperature: Override temperature setting
        max_tokens: Override max tokens setting
        metadata: Additional arbitrary metadata

    Example YAML:
        ```yaml
        name: analyst
        prompt: |
          You are a senior data analyst.
          Always verify data accuracy before making claims.

        tools:
          - query_database
          - create_chart

        deny:
          - delete_record
          - drop_table

        approve:
          - send_email
          - publish_report

        temperature: 0.3
        max_tokens: 4000
        ```
    """

    name: str
    prompt: str = ""
    tools: list[str] | None = None
    deny: list[str] | None = None
    approve: list[str] | None = None
    provider: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Persona:
        """
        Load persona from a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Persona instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If required fields are missing
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Persona file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty persona file: {path}")

        if "name" not in data:
            # Use filename as name if not specified
            data["name"] = path.stem

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        """
        Create persona from dictionary.

        Args:
            data: Dictionary with persona configuration

        Returns:
            Persona instance
        """
        # Extract known fields
        known_fields = {
            "name",
            "prompt",
            "tools",
            "deny",
            "approve",
            "provider",
            "model_name",
            "temperature",
            "max_tokens",
        }

        # Separate known fields from metadata
        persona_kwargs = {}
        metadata = {}

        for key, value in data.items():
            if key in known_fields:
                persona_kwargs[key] = value
            else:
                metadata[key] = value

        persona_kwargs["metadata"] = metadata

        return cls(**persona_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert persona to dictionary (for serialization).

        Returns:
            Dictionary representation
        """
        result: dict[str, Any] = {"name": self.name}

        if self.prompt:
            result["prompt"] = self.prompt
        if self.tools:
            result["tools"] = self.tools
        if self.deny:
            result["deny"] = self.deny
        if self.approve:
            result["approve"] = self.approve
        if self.provider:
            result["provider"] = self.provider
        if self.model_name:
            result["model_name"] = self.model_name
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.metadata:
            result.update(self.metadata)

        return result

    def save_yaml(self, path: str | Path) -> None:
        """
        Save persona to YAML file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def build_tool_filter(
    allowed: list[str] | None = None,
    denied: list[str] | None = None,
) -> Callable[[str], bool] | None:
    """
    Build a tool filter function from allow/deny lists.

    Args:
        allowed: List of allowed tool names (whitelist).
            If provided, only these tools are allowed.
        denied: List of denied tool names (blacklist).
            If provided, these tools are blocked.

    Returns:
        Filter function that returns True if tool is allowed,
        or None if no filtering is needed.

    Note:
        - If both allowed and denied are provided, allowed takes precedence
        - denied is applied as additional restriction on allowed
    """
    if not allowed and not denied:
        return None

    allowed_set = set(allowed) if allowed else None
    denied_set = set(denied) if denied else set()

    def filter_fn(tool_name: str) -> bool:
        # If whitelist exists, tool must be in it
        if allowed_set is not None and tool_name not in allowed_set:
            return False
        # Blacklist always applies
        if tool_name in denied_set:
            return False
        return True

    return filter_fn
