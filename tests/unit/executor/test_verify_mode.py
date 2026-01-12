"""Unit tests for verify mode functionality (Phase 5.9.2).

Tests for VerifyMode enum and agent-written verification mode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor import (
    ExecutorConfig,
    VerifyMode,
)
from ai_infra.executor.loop import Executor
from ai_infra.executor.roadmap import ParsedTask

# =============================================================================
# VerifyMode Enum Tests
# =============================================================================


class TestVerifyModeEnum:
    """Tests for the VerifyMode enum."""

    def test_verify_mode_values(self) -> None:
        """Verify all expected modes exist."""
        assert VerifyMode.AUTO.value == "auto"
        assert VerifyMode.AGENT.value == "agent"
        assert VerifyMode.SKIP.value == "skip"
        assert VerifyMode.PYTEST.value == "pytest"

    def test_verify_mode_is_string_enum(self) -> None:
        """VerifyMode should be a string enum."""
        assert isinstance(VerifyMode.AUTO.value, str)
        assert str(VerifyMode.AUTO) == "VerifyMode.AUTO"


# =============================================================================
# ExecutorConfig Tests
# =============================================================================


class TestExecutorConfigVerifyMode:
    """Tests for verify_mode in ExecutorConfig."""

    def test_default_verify_mode_is_auto(self) -> None:
        """Default verify mode should be AUTO."""
        config = ExecutorConfig()
        assert config.verify_mode == VerifyMode.AUTO

    def test_verify_mode_in_to_dict(self) -> None:
        """verify_mode should be included in to_dict()."""
        config = ExecutorConfig(verify_mode=VerifyMode.AGENT)
        config_dict = config.to_dict()
        assert "verify_mode" in config_dict
        assert config_dict["verify_mode"] == "agent"

    def test_config_accepts_all_verify_modes(self) -> None:
        """Config should accept all verify mode values."""
        for mode in VerifyMode:
            config = ExecutorConfig(verify_mode=mode)
            assert config.verify_mode == mode


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestAgentVerificationPrompt:
    """Tests for agent verification mode prompt building."""

    @pytest.fixture
    def sample_roadmap_content(self) -> str:
        """Sample ROADMAP content."""
        return """\
# Test Project ROADMAP

## Phase 0: Foundation

### 0.1 Setup

- [ ] **Create utils module**
  Create a utility module with helper functions.
"""

    @pytest.fixture
    def roadmap_file(self, tmp_path: Path, sample_roadmap_content: str) -> Path:
        """Create a temporary ROADMAP.md file."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text(sample_roadmap_content)
        # Add pyproject.toml so it's detected as Python project
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        return roadmap

    @pytest.fixture
    def sample_task(self) -> ParsedTask:
        """Create a sample parsed task."""
        return ParsedTask(
            id="0.1.1",
            title="Create utils module",
            description="Create a utility module with helper functions.",
            phase_id="0",
            section_id="0.1",
            file_hints=[],
            code_context=[],
            subtasks=[],
        )

    @pytest.mark.asyncio
    async def test_agent_mode_adds_verification_requirement(
        self, roadmap_file: Path, sample_task: ParsedTask
    ) -> None:
        """Agent mode should add verification requirement to prompt."""
        config = ExecutorConfig(verify_mode=VerifyMode.AGENT)

        with patch("ai_infra.executor.loop.ProjectContext") as mock_context_cls:
            mock_context = MagicMock()
            mock_context_cls.return_value = mock_context
            mock_context.build_index = AsyncMock()

            executor = Executor(roadmap=roadmap_file, config=config, agent=None)

            # Build prompt
            prompt = await executor._build_prompt(sample_task)

            # Should contain verification requirement section
            assert "VERIFICATION REQUIREMENT" in prompt
            assert "After completing this task, you MUST verify" in prompt
            assert "pytest" in prompt or "npm test" in prompt or "cargo test" in prompt

    @pytest.mark.asyncio
    async def test_auto_mode_does_not_add_verification_requirement(
        self, roadmap_file: Path, sample_task: ParsedTask
    ) -> None:
        """Auto mode should NOT add verification requirement to prompt."""
        config = ExecutorConfig(verify_mode=VerifyMode.AUTO)

        with patch("ai_infra.executor.loop.ProjectContext") as mock_context_cls:
            mock_context = MagicMock()
            mock_context_cls.return_value = mock_context
            mock_context.build_index = AsyncMock()

            executor = Executor(roadmap=roadmap_file, config=config, agent=None)

            prompt = await executor._build_prompt(sample_task)

            assert "VERIFICATION REQUIREMENT" not in prompt

    @pytest.mark.asyncio
    async def test_skip_mode_does_not_add_verification_requirement(
        self, roadmap_file: Path, sample_task: ParsedTask
    ) -> None:
        """Skip mode should NOT add verification requirement to prompt."""
        config = ExecutorConfig(verify_mode=VerifyMode.SKIP)

        with patch("ai_infra.executor.loop.ProjectContext") as mock_context_cls:
            mock_context = MagicMock()
            mock_context_cls.return_value = mock_context
            mock_context.build_index = AsyncMock()

            executor = Executor(roadmap=roadmap_file, config=config, agent=None)

            prompt = await executor._build_prompt(sample_task)

            assert "VERIFICATION REQUIREMENT" not in prompt


# =============================================================================
# Verification Skipping Tests
# =============================================================================


class TestVerificationSkipping:
    """Tests for skipping verification in agent/skip modes."""

    def test_should_verify_auto_mode(self) -> None:
        """AUTO mode should run verification."""
        config = ExecutorConfig(verify_mode=VerifyMode.AUTO)
        should_verify = (
            not config.skip_verification
            and config.verify_mode != VerifyMode.AGENT
            and config.verify_mode != VerifyMode.SKIP
        )
        assert should_verify is True

    def test_should_not_verify_agent_mode(self) -> None:
        """AGENT mode should NOT run automated verification."""
        config = ExecutorConfig(verify_mode=VerifyMode.AGENT)
        should_verify = (
            not config.skip_verification
            and config.verify_mode != VerifyMode.AGENT
            and config.verify_mode != VerifyMode.SKIP
        )
        assert should_verify is False

    def test_should_not_verify_skip_mode(self) -> None:
        """SKIP mode should NOT run automated verification."""
        config = ExecutorConfig(verify_mode=VerifyMode.SKIP)
        should_verify = (
            not config.skip_verification
            and config.verify_mode != VerifyMode.AGENT
            and config.verify_mode != VerifyMode.SKIP
        )
        assert should_verify is False

    def test_should_verify_pytest_mode(self) -> None:
        """PYTEST mode should run verification."""
        config = ExecutorConfig(verify_mode=VerifyMode.PYTEST)
        should_verify = (
            not config.skip_verification
            and config.verify_mode != VerifyMode.AGENT
            and config.verify_mode != VerifyMode.SKIP
        )
        assert should_verify is True

    def test_skip_verification_flag_overrides_auto_mode(self) -> None:
        """skip_verification=True should override verify_mode."""
        config = ExecutorConfig(verify_mode=VerifyMode.AUTO, skip_verification=True)
        should_verify = (
            not config.skip_verification
            and config.verify_mode != VerifyMode.AGENT
            and config.verify_mode != VerifyMode.SKIP
        )
        assert should_verify is False
