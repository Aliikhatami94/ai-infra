"""Unit tests for Executor prompts and templates.

Phase 1.5: Added tests for few-shot templates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_infra.executor.prompts import (
    CONFIG_TEMPLATE,
    DOCUMENTATION_TEMPLATE,
    GENERIC_TEMPLATE,
    NORMALIZE_ROADMAP_PROMPT,
    NORMALIZE_ROADMAP_SYSTEM_PROMPT,
    PYTHON_CODE_TEMPLATE,
    SCRIPT_TEMPLATE,
    TEMPLATES,
    TEST_CODE_TEMPLATE,
    TemplateType,
    format_template,
    get_few_shot_section,
    get_template,
    infer_template_type,
)
from ai_infra.executor.prompts.normalize_roadmap import (
    NORMALIZE_ROADMAP_PROMPT_COMPACT,
)


class TestNormalizeRoadmapPrompt:
    """Tests for the ROADMAP normalization prompt templates."""

    def test_prompt_has_placeholder(self) -> None:
        """Test that the prompt has the roadmap_content placeholder."""
        assert "{roadmap_content}" in NORMALIZE_ROADMAP_PROMPT
        assert "{roadmap_content}" in NORMALIZE_ROADMAP_PROMPT_COMPACT

    def test_prompt_format_with_content(self) -> None:
        """Test that the prompt can be formatted with content."""
        roadmap = "# Test\\n- [ ] Task 1"
        formatted = NORMALIZE_ROADMAP_PROMPT.format(roadmap_content=roadmap)

        assert roadmap in formatted
        assert "{roadmap_content}" not in formatted

    def test_prompt_includes_json_structure(self) -> None:
        """Test that the prompt specifies the expected JSON structure."""
        assert '"todos"' in NORMALIZE_ROADMAP_PROMPT
        assert '"id"' in NORMALIZE_ROADMAP_PROMPT
        assert '"title"' in NORMALIZE_ROADMAP_PROMPT
        assert '"status"' in NORMALIZE_ROADMAP_PROMPT
        assert '"file_hints"' in NORMALIZE_ROADMAP_PROMPT
        assert '"dependencies"' in NORMALIZE_ROADMAP_PROMPT
        assert '"source_line"' in NORMALIZE_ROADMAP_PROMPT
        assert '"source_text"' in NORMALIZE_ROADMAP_PROMPT

    def test_prompt_includes_status_values(self) -> None:
        """Test that the prompt documents all status values."""
        assert "pending" in NORMALIZE_ROADMAP_PROMPT
        assert "completed" in NORMALIZE_ROADMAP_PROMPT
        assert "skipped" in NORMALIZE_ROADMAP_PROMPT

    def test_prompt_includes_format_examples(self) -> None:
        """Test that the prompt includes format recognition examples."""
        # Should recognize various formats
        assert "[ ]" in NORMALIZE_ROADMAP_PROMPT
        assert "[x]" in NORMALIZE_ROADMAP_PROMPT
        assert "emoji" in NORMALIZE_ROADMAP_PROMPT.lower()
        assert "bullet" in NORMALIZE_ROADMAP_PROMPT.lower()

    def test_system_prompt_exists(self) -> None:
        """Test that the system prompt is defined."""
        assert len(NORMALIZE_ROADMAP_SYSTEM_PROMPT) > 50
        assert "task extraction" in NORMALIZE_ROADMAP_SYSTEM_PROMPT.lower()
        assert "JSON" in NORMALIZE_ROADMAP_SYSTEM_PROMPT

    def test_compact_prompt_is_shorter(self) -> None:
        """Test that the compact prompt is shorter than the full prompt."""
        assert len(NORMALIZE_ROADMAP_PROMPT_COMPACT) < len(NORMALIZE_ROADMAP_PROMPT)
        # But still functional
        assert "{roadmap_content}" in NORMALIZE_ROADMAP_PROMPT_COMPACT
        assert "todos" in NORMALIZE_ROADMAP_PROMPT_COMPACT


# =============================================================================
# Phase 1.5: Few-Shot Template Tests
# =============================================================================


@dataclass
class MockTask:
    """Mock task object for testing."""

    id: str = "task-1"
    title: str = "Test task"
    description: str = ""
    file_hints: list[str] | None = None


@pytest.fixture
def python_task() -> MockTask:
    """Create a Python code task."""
    return MockTask(
        id="task-python",
        title="Create a utility function",
        description="Implement a function to parse configuration files",
        file_hints=["src/utils.py"],
    )


@pytest.fixture
def test_task() -> MockTask:
    """Create a test code task."""
    return MockTask(
        id="task-test",
        title="Add tests for user module",
        description="Write pytest tests for the user service",
        file_hints=["tests/test_user.py"],
    )


@pytest.fixture
def config_task() -> MockTask:
    """Create a configuration task."""
    return MockTask(
        id="task-config",
        title="Create pyproject.toml",
        description="Set up project configuration with dependencies",
        file_hints=["pyproject.toml"],
    )


@pytest.fixture
def script_task() -> MockTask:
    """Create a script task."""
    return MockTask(
        id="task-script",
        title="Create build script",
        description="Write a shell script to build the project",
        file_hints=["scripts/build.sh"],
    )


@pytest.fixture
def docs_task() -> MockTask:
    """Create a documentation task."""
    return MockTask(
        id="task-docs",
        title="Update README documentation",
        description="Add installation instructions to README.md",
        file_hints=["README.md"],
    )


class TestTemplateConstants:
    """Tests for template constant definitions."""

    def test_python_template_has_placeholder(self) -> None:
        """Test: Python template contains task_description placeholder."""
        assert "{task_description}" in PYTHON_CODE_TEMPLATE

    def test_test_template_has_placeholder(self) -> None:
        """Test: Test template contains task_description placeholder."""
        assert "{task_description}" in TEST_CODE_TEMPLATE

    def test_config_template_has_placeholder(self) -> None:
        """Test: Config template contains task_description placeholder."""
        assert "{task_description}" in CONFIG_TEMPLATE

    def test_script_template_has_placeholder(self) -> None:
        """Test: Script template contains task_description placeholder."""
        assert "{task_description}" in SCRIPT_TEMPLATE

    def test_documentation_template_has_placeholder(self) -> None:
        """Test: Documentation template contains task_description placeholder."""
        assert "{task_description}" in DOCUMENTATION_TEMPLATE

    def test_generic_template_has_placeholder(self) -> None:
        """Test: Generic template contains task_description placeholder."""
        assert "{task_description}" in GENERIC_TEMPLATE

    def test_templates_dict_has_all_types(self) -> None:
        """Test: TEMPLATES dict contains all TemplateType values."""
        for template_type in TemplateType:
            assert template_type in TEMPLATES

    def test_python_template_has_example(self) -> None:
        """Test: Python template includes a code example."""
        assert "def factorial" in PYTHON_CODE_TEMPLATE

    def test_test_template_has_pytest_example(self) -> None:
        """Test: Test template includes pytest example."""
        assert "pytest" in TEST_CODE_TEMPLATE
        assert "assert" in TEST_CODE_TEMPLATE

    def test_config_template_has_toml_example(self) -> None:
        """Test: Config template includes TOML example."""
        assert "[project]" in CONFIG_TEMPLATE

    def test_script_template_has_bash_example(self) -> None:
        """Test: Script template includes bash example."""
        assert "#!/usr/bin/env bash" in SCRIPT_TEMPLATE
        assert "set -euo pipefail" in SCRIPT_TEMPLATE


class TestInferTemplateType:
    """Tests for infer_template_type function."""

    def test_test_keywords_detected(self) -> None:
        """Test: Test-related keywords trigger TEST_CODE type."""
        assert infer_template_type("Add tests for module") == TemplateType.TEST_CODE
        assert infer_template_type("Write pytest tests") == TemplateType.TEST_CODE

    def test_config_keywords_detected(self) -> None:
        """Test: Config-related keywords trigger CONFIG type."""
        assert infer_template_type("Create pyproject.toml") == TemplateType.CONFIG
        assert infer_template_type("Update settings.yaml") == TemplateType.CONFIG

    def test_script_keywords_detected(self) -> None:
        """Test: Script-related keywords trigger SCRIPT type."""
        assert infer_template_type("Write bash script") == TemplateType.SCRIPT
        assert infer_template_type("Create shell utility") == TemplateType.SCRIPT

    def test_documentation_keywords_detected(self) -> None:
        """Test: Documentation-related keywords trigger DOCUMENTATION type."""
        assert infer_template_type("Update README") == TemplateType.DOCUMENTATION
        assert infer_template_type("Write documentation") == TemplateType.DOCUMENTATION

    def test_python_keywords_detected(self) -> None:
        """Test: Python-related keywords trigger PYTHON_CODE type."""
        assert infer_template_type("Implement feature") == TemplateType.PYTHON_CODE
        assert infer_template_type("Create new module") == TemplateType.PYTHON_CODE

    def test_generic_for_unknown(self) -> None:
        """Test: Unknown tasks get GENERIC type."""
        assert infer_template_type("Do something") == TemplateType.GENERIC
        assert infer_template_type("") == TemplateType.GENERIC

    def test_file_hints_boost_detection(self) -> None:
        """Test: File hints improve template type detection."""
        result = infer_template_type(
            "Update code",
            file_hints=["src/module.py"],
        )
        assert result == TemplateType.PYTHON_CODE

    def test_test_file_hints(self) -> None:
        """Test: Test file hints trigger TEST_CODE type."""
        result = infer_template_type(
            "Update module",
            file_hints=["tests/test_module.py"],
        )
        assert result == TemplateType.TEST_CODE

    def test_config_file_hints(self) -> None:
        """Test: Config file hints trigger CONFIG type."""
        result = infer_template_type(
            "Update project",
            file_hints=["pyproject.toml"],
        )
        assert result == TemplateType.CONFIG

    def test_test_takes_priority(self) -> None:
        """Test: Test type takes priority when mixed signals present."""
        result = infer_template_type("Implement tests for feature")
        assert result == TemplateType.TEST_CODE


class TestGetTemplate:
    """Tests for get_template function."""

    def test_get_python_template(self) -> None:
        """Test: Get PYTHON_CODE template."""
        template = get_template(TemplateType.PYTHON_CODE)
        assert template == PYTHON_CODE_TEMPLATE

    def test_get_test_template(self) -> None:
        """Test: Get TEST_CODE template."""
        template = get_template(TemplateType.TEST_CODE)
        assert template == TEST_CODE_TEMPLATE

    def test_get_config_template(self) -> None:
        """Test: Get CONFIG template."""
        template = get_template(TemplateType.CONFIG)
        assert template == CONFIG_TEMPLATE

    def test_get_script_template(self) -> None:
        """Test: Get SCRIPT template."""
        template = get_template(TemplateType.SCRIPT)
        assert template == SCRIPT_TEMPLATE

    def test_get_documentation_template(self) -> None:
        """Test: Get DOCUMENTATION template."""
        template = get_template(TemplateType.DOCUMENTATION)
        assert template == DOCUMENTATION_TEMPLATE

    def test_get_generic_template(self) -> None:
        """Test: Get GENERIC template."""
        template = get_template(TemplateType.GENERIC)
        assert template == GENERIC_TEMPLATE


class TestFormatTemplate:
    """Tests for format_template function."""

    def test_formats_with_title(self) -> None:
        """Test: Template formats with task title."""
        result = format_template("Create user service")
        assert "Create user service" in result

    def test_formats_with_description(self) -> None:
        """Test: Template includes description when provided."""
        result = format_template(
            "Create service",
            task_description="Implement CRUD operations",
        )
        assert "Create service" in result
        assert "Implement CRUD operations" in result

    def test_explicit_template_type(self) -> None:
        """Test: Explicit template type is used."""
        result = format_template(
            "Something",
            template_type=TemplateType.TEST_CODE,
        )
        assert "pytest" in result.lower()

    def test_infers_template_type(self) -> None:
        """Test: Template type is inferred when not provided."""
        result = format_template("Add tests for module")
        assert "pytest" in result.lower()


class TestGetFewShotSection:
    """Tests for get_few_shot_section function."""

    def test_extracts_task_attributes(self, python_task: MockTask) -> None:
        """Test: Function extracts task attributes correctly."""
        result = get_few_shot_section(python_task)
        assert python_task.title in result

    def test_handles_missing_description(self) -> None:
        """Test: Handles task with no description."""
        task = MockTask(title="Simple task", description="")
        result = get_few_shot_section(task)
        assert "Simple task" in result

    def test_handles_no_file_hints(self) -> None:
        """Test: Handles task with no file hints."""
        task = MockTask(title="Task without hints", file_hints=None)
        result = get_few_shot_section(task)
        assert "Task without hints" in result

    def test_respects_explicit_template_type(self, python_task: MockTask) -> None:
        """Test: Explicit template type override works."""
        result = get_few_shot_section(
            python_task,
            template_type=TemplateType.TEST_CODE,
        )
        assert "pytest" in result.lower()

    def test_test_task_gets_test_template(self, test_task: MockTask) -> None:
        """Test: Test task gets TEST_CODE template."""
        result = get_few_shot_section(test_task)
        assert "pytest" in result.lower()

    def test_config_task_gets_config_template(self, config_task: MockTask) -> None:
        """Test: Config task gets CONFIG template."""
        result = get_few_shot_section(config_task)
        assert "[project]" in result

    def test_script_task_gets_script_template(self, script_task: MockTask) -> None:
        """Test: Script task gets SCRIPT template."""
        result = get_few_shot_section(script_task)
        assert "bash" in result.lower()

    def test_docs_task_gets_docs_template(self, docs_task: MockTask) -> None:
        """Test: Documentation task gets DOCUMENTATION template."""
        result = get_few_shot_section(docs_task)
        assert "markdown" in result.lower()


class TestTemplateQuality:
    """Tests for template content quality."""

    def test_python_template_has_type_hints(self) -> None:
        """Test: Python template example includes type hints."""
        assert "int" in PYTHON_CODE_TEMPLATE
        assert "->" in PYTHON_CODE_TEMPLATE

    def test_python_template_has_docstring(self) -> None:
        """Test: Python template example includes docstring."""
        assert '"""' in PYTHON_CODE_TEMPLATE

    def test_python_template_has_error_handling(self) -> None:
        """Test: Python template example includes error handling."""
        assert "raise" in PYTHON_CODE_TEMPLATE
        assert "ValueError" in PYTHON_CODE_TEMPLATE

    def test_test_template_has_class_structure(self) -> None:
        """Test: Test template uses class-based structure."""
        assert "class Test" in TEST_CODE_TEMPLATE

    def test_test_template_has_assertions(self) -> None:
        """Test: Test template includes assertions."""
        assert "assert" in TEST_CODE_TEMPLATE

    def test_test_template_has_exception_testing(self) -> None:
        """Test: Test template shows exception testing."""
        assert "pytest.raises" in TEST_CODE_TEMPLATE

    def test_config_template_has_proper_structure(self) -> None:
        """Test: Config template has proper TOML structure."""
        assert "[project]" in CONFIG_TEMPLATE
        assert "dependencies" in CONFIG_TEMPLATE

    def test_script_template_has_safety(self) -> None:
        """Test: Script template includes safety features."""
        assert "set -euo pipefail" in SCRIPT_TEMPLATE

    def test_templates_no_raw_braces(self) -> None:
        """Test: Templates don't have unescaped braces that break formatting."""
        for template_type, template in TEMPLATES.items():
            try:
                template.format(task_description="test")
            except KeyError as e:
                pytest.fail(f"Template {template_type.value} has unescaped placeholder: {e}")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_title(self) -> None:
        """Test: Empty title doesn't crash."""
        result = format_template("")
        assert result

    def test_none_like_values(self) -> None:
        """Test: None-like values are handled."""
        task = MockTask(title="", description="", file_hints=[])
        result = get_few_shot_section(task)
        assert result

    def test_unicode_in_task(self) -> None:
        """Test: Unicode characters are handled."""
        result = format_template("Add emoji support")
        assert "emoji" in result

    def test_multiline_description(self) -> None:
        """Test: Multiline descriptions work."""
        multiline = "Line 1\nLine 2\nLine 3"
        result = format_template("Task", task_description=multiline)
        assert "Line 1" in result
        assert "Line 2" in result
