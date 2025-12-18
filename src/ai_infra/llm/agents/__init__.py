"""Agent submodules for split functionality.

This package contains extracted modules from the main Agent class:
- deep: DeepAgents integration for autonomous multi-step tasks
- callbacks: Callback wrapping utilities for tool instrumentation
"""

from ai_infra.llm.agents.callbacks import wrap_tool_with_callbacks
from ai_infra.llm.agents.deep import (
    HAS_DEEPAGENTS,
    AgentMiddleware,
    CompiledSubAgent,
    FilesystemMiddleware,
    SubAgent,
    SubAgentMiddleware,
    build_deep_agent,
)

__all__ = [
    # Deep agents
    "HAS_DEEPAGENTS",
    "SubAgent",
    "CompiledSubAgent",
    "SubAgentMiddleware",
    "FilesystemMiddleware",
    "AgentMiddleware",
    "build_deep_agent",
    # Callbacks
    "wrap_tool_with_callbacks",
]
