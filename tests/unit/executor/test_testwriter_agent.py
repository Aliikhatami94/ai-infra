"""Tests for Phase 16.5.11.1: TestWriterAgent.

This module tests the TestWriterAgent functionality:
- TestWriterAgent instantiation and registration
- TestWriterAgent class attributes
- SubAgentType.TESTWRITER enum value
- Agent is registered in SubAgentRegistry
"""

from __future__ import annotations

import pytest

from ai_infra.executor.agents.base import (
    TEST_WRITER_SYSTEM_PROMPT,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.agents.testwriter import TEST_WRITER_PROMPT, TestWriterAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def ensure_testwriter_registered():
    """Ensure TestWriterAgent is properly registered before tests."""
    # Re-register TestWriterAgent in case registry was cleared by another test
    SubAgentRegistry._agents[SubAgentType.TESTWRITER] = TestWriterAgent
    yield


# =============================================================================
# SubAgentType Tests
# =============================================================================


class TestSubAgentTypeEnum:
    """Tests for TESTWRITER enum value in SubAgentType."""

    def test_testwriter_enum_exists(self) -> None:
        """Test TESTWRITER is defined in SubAgentType."""
        assert hasattr(SubAgentType, "TESTWRITER")
        assert SubAgentType.TESTWRITER.value == "testwriter"

    def test_testwriter_is_string_enum(self) -> None:
        """Test TESTWRITER inherits from str."""
        assert isinstance(SubAgentType.TESTWRITER, str)
        assert SubAgentType.TESTWRITER == "testwriter"

    def test_all_agent_types_exist(self) -> None:
        """Test all expected agent types are defined."""
        expected_types = {
            "coder",
            "reviewer",
            "tester",
            "debugger",
            "researcher",
            "testwriter",
        }
        actual_types = {t.value for t in SubAgentType}
        assert expected_types.issubset(actual_types)


# =============================================================================
# TestWriterAgent Class Attributes Tests
# =============================================================================


class TestTestWriterAgentAttributes:
    """Tests for TestWriterAgent class attributes."""

    def test_name_attribute(self) -> None:
        """Test TestWriterAgent has correct name."""
        assert TestWriterAgent.name == "TestWriter"

    def test_description_attribute(self) -> None:
        """Test TestWriterAgent has correct description."""
        assert TestWriterAgent.description == "Creates comprehensive test suites"

    def test_model_attribute(self) -> None:
        """Test TestWriterAgent has correct default model."""
        assert TestWriterAgent.model == "claude-sonnet-4-20250514"

    def test_system_prompt_attribute(self) -> None:
        """Test TestWriterAgent has correct system prompt."""
        assert TestWriterAgent.system_prompt == TEST_WRITER_SYSTEM_PROMPT


# =============================================================================
# TestWriterAgent Instantiation Tests
# =============================================================================


class TestTestWriterAgentInstantiation:
    """Tests for TestWriterAgent instantiation."""

    def test_default_instantiation(self) -> None:
        """Test TestWriterAgent can be instantiated with defaults."""
        agent = TestWriterAgent()
        assert agent._model == "claude-sonnet-4-20250514"
        assert agent._timeout == 300.0
        assert agent._shell_timeout == 60.0

    def test_custom_model_instantiation(self) -> None:
        """Test TestWriterAgent can be instantiated with custom model."""
        agent = TestWriterAgent(model="gpt-4o")
        assert agent._model == "gpt-4o"

    def test_custom_timeout_instantiation(self) -> None:
        """Test TestWriterAgent can be instantiated with custom timeouts."""
        agent = TestWriterAgent(timeout=600.0, shell_timeout=120.0)
        assert agent._timeout == 600.0
        assert agent._shell_timeout == 120.0

    def test_repr(self) -> None:
        """Test TestWriterAgent string representation."""
        agent = TestWriterAgent()
        repr_str = repr(agent)
        assert "TestWriterAgent" in repr_str
        assert "TestWriter" in repr_str
        assert "claude-sonnet-4-20250514" in repr_str


# =============================================================================
# SubAgentRegistry Tests
# =============================================================================


class TestTestWriterAgentRegistry:
    """Tests for TestWriterAgent registration in SubAgentRegistry."""

    def test_agent_is_registered(self) -> None:
        """Test TestWriterAgent is registered in SubAgentRegistry."""
        agent = SubAgentRegistry.get(SubAgentType.TESTWRITER)
        assert isinstance(agent, TestWriterAgent)

    def test_registry_returns_correct_type(self) -> None:
        """Test SubAgentRegistry.get returns TestWriterAgent instance."""
        agent = SubAgentRegistry.get(SubAgentType.TESTWRITER)
        assert agent.name == "TestWriter"

    def test_testwriter_in_available_types(self) -> None:
        """Test TESTWRITER appears in available types."""
        available = SubAgentRegistry.available_types()
        assert SubAgentType.TESTWRITER in available


# =============================================================================
# TEST_WRITER_SYSTEM_PROMPT Tests
# =============================================================================


class TestTestWriterSystemPrompt:
    """Tests for TEST_WRITER_SYSTEM_PROMPT content."""

    def test_prompt_is_string(self) -> None:
        """Test prompt is a non-empty string."""
        assert isinstance(TEST_WRITER_SYSTEM_PROMPT, str)
        assert len(TEST_WRITER_SYSTEM_PROMPT) > 0

    def test_prompt_contains_key_instructions(self) -> None:
        """Test prompt contains key testing instructions."""
        assert "test" in TEST_WRITER_SYSTEM_PROMPT.lower()
        assert "edge case" in TEST_WRITER_SYSTEM_PROMPT.lower()
        assert "pytest" in TEST_WRITER_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_arr_pattern(self) -> None:
        """Test prompt mentions Arrange-Act-Assert pattern."""
        assert "Arrange-Act-Assert" in TEST_WRITER_SYSTEM_PROMPT


# =============================================================================
# TEST_WRITER_PROMPT Tests
# =============================================================================


class TestTestWriterPrompt:
    """Tests for TEST_WRITER_PROMPT template."""

    def test_prompt_has_required_placeholders(self) -> None:
        """Test prompt has all required format placeholders."""
        required_placeholders = [
            "{task_title}",
            "{task_description}",
            "{workspace}",
            "{project_type}",
            "{dependencies}",
            "{files_modified}",
            "{enriched_context}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in TEST_WRITER_PROMPT, f"Missing {placeholder}"

    def test_prompt_can_be_formatted(self) -> None:
        """Test prompt can be formatted with values."""
        formatted = TEST_WRITER_PROMPT.format(
            task_title="Test User Model",
            task_description="Create tests for user.py",
            workspace="/workspace",
            project_type="python",
            dependencies="pytest",
            files_modified="- src/user.py",
            enriched_context="",
        )
        assert "Test User Model" in formatted
        assert "/workspace" in formatted


# =============================================================================
# TestWriterAgent._get_tools Tests
# =============================================================================


class TestTestWriterAgentTools:
    """Tests for TestWriterAgent._get_tools method."""

    def test_get_tools_returns_empty_list(self) -> None:
        """Test _get_tools returns empty list (tools created per-execution)."""
        agent = TestWriterAgent()
        tools = agent._get_tools()
        assert tools == []
        assert isinstance(tools, list)


# =============================================================================
# TestWriterAgent._extract_output Tests
# =============================================================================


class TestTestWriterAgentExtractOutput:
    """Tests for TestWriterAgent._extract_output method."""

    def test_extract_from_string(self) -> None:
        """Test extracting output from string result."""
        agent = TestWriterAgent()
        result = agent._extract_output("Test output")
        assert result == "Test output"

    def test_extract_from_dict_with_content(self) -> None:
        """Test extracting output from dict with content key."""
        agent = TestWriterAgent()
        result = agent._extract_output({"content": "Test content"})
        assert result == "Test content"

    def test_extract_from_dict_with_output(self) -> None:
        """Test extracting output from dict with output key."""
        agent = TestWriterAgent()
        result = agent._extract_output({"output": "Test output"})
        assert result == "Test output"

    def test_extract_from_object_with_content(self) -> None:
        """Test extracting output from object with content attribute."""
        agent = TestWriterAgent()

        class MockResult:
            content = "Mock content"

        result = agent._extract_output(MockResult())
        assert result == "Mock content"


# =============================================================================
# TestWriterAgent._extract_command_string Tests
# =============================================================================


class TestTestWriterAgentExtractCommandString:
    """Tests for TestWriterAgent._extract_command_string method."""

    def test_extract_from_string(self) -> None:
        """Test extracting from string command."""
        agent = TestWriterAgent()
        result = agent._extract_command_string("cat > test.py")
        assert result == "cat > test.py"

    def test_extract_from_shell_result(self) -> None:
        """Test extracting from ShellResult-like object."""
        agent = TestWriterAgent()

        class MockShellResult:
            command = "pytest tests/"

        result = agent._extract_command_string(MockShellResult())
        assert result == "pytest tests/"


# =============================================================================
# TestWriterAgent._analyze_file_changes Tests
# =============================================================================


class TestTestWriterAgentAnalyzeFileChanges:
    """Tests for TestWriterAgent._analyze_file_changes method."""

    def test_detect_cat_redirect_creation(self) -> None:
        """Test detecting file creation via cat >."""
        agent = TestWriterAgent()
        history = ["cat > tests/test_user.py << 'EOF'"]
        created, modified = agent._analyze_file_changes(history)
        assert "tests/test_user.py" in created

    def test_detect_touch_creation(self) -> None:
        """Test detecting file creation via touch."""
        agent = TestWriterAgent()
        history = ["touch tests/__init__.py"]
        created, modified = agent._analyze_file_changes(history)
        assert "tests/__init__.py" in created

    def test_detect_sed_modification(self) -> None:
        """Test detecting file modification via sed -i."""
        agent = TestWriterAgent()
        history = ["sed -i '' 's/old/new/g' tests/test_user.py"]
        created, modified = agent._analyze_file_changes(history)
        assert "tests/test_user.py" in modified

    def test_mkdir_not_counted_as_file(self) -> None:
        """Test mkdir commands are not counted as file changes."""
        agent = TestWriterAgent()
        history = ["mkdir -p tests/unit"]
        created, modified = agent._analyze_file_changes(history)
        assert len(created) == 0
        assert len(modified) == 0

    def test_empty_history(self) -> None:
        """Test with empty command history."""
        agent = TestWriterAgent()
        created, modified = agent._analyze_file_changes([])
        assert created == []
        assert modified == []


# =============================================================================
# SubAgentResult Tests
# =============================================================================


class TestSubAgentResultForTestWriter:
    """Tests for SubAgentResult used by TestWriterAgent."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        result = SubAgentResult(
            success=True,
            output="Created tests successfully",
            files_created=["tests/test_user.py"],
            files_modified=[],
            metrics={"duration_ms": 1500.0},
        )
        assert result.success is True
        assert len(result.files_created) == 1
        assert result.metrics["duration_ms"] == 1500.0

    def test_failure_result(self) -> None:
        """Test creating a failed result."""
        result = SubAgentResult(
            success=False,
            error="Shell command failed",
            metrics={"duration_ms": 500.0},
        )
        assert result.success is False
        assert result.error == "Shell command failed"


# =============================================================================
# Integration: Imports and Exports
# =============================================================================


class TestTestWriterAgentExports:
    """Tests for TestWriterAgent module exports."""

    def test_import_from_agents_module(self) -> None:
        """Test TestWriterAgent can be imported from agents module."""
        from ai_infra.executor.agents import TestWriterAgent as ImportedAgent

        assert ImportedAgent is TestWriterAgent

    def test_import_from_agents_all(self) -> None:
        """Test TestWriterAgent is in agents __all__."""
        from ai_infra.executor.agents import __all__

        assert "TestWriterAgent" in __all__
