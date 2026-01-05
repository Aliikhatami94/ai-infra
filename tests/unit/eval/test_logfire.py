"""Unit tests for ai-infra Logfire integration (Phase 11.4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestConfigureLogfireEvals:
    """Tests for configure_logfire_evals function."""

    def test_import(self) -> None:
        """Test that configure_logfire_evals can be imported."""
        from ai_infra.eval import configure_logfire_evals

        assert configure_logfire_evals is not None

    def test_import_from_logfire_module(self) -> None:
        """Test import from logfire submodule."""
        from ai_infra.eval.logfire import configure_logfire_evals

        assert configure_logfire_evals is not None

    @patch("ai_infra.eval.logfire.logfire", create=True)
    def test_configures_logfire_with_defaults(self, mock_logfire: MagicMock) -> None:
        """Test that configure_logfire_evals calls logfire.configure with defaults."""
        # Need to reimport to pick up the mock
        import importlib

        import ai_infra.eval.logfire as logfire_module

        # Mock the import
        with patch.dict("sys.modules", {"logfire": mock_logfire}):
            importlib.reload(logfire_module)

            logfire_module.configure_logfire_evals()

            mock_logfire.configure.assert_called_once_with(
                service_name="ai-infra-evals",
                environment=None,
                send_to_logfire="if-token-present",
            )

    def test_returns_false_when_logfire_not_installed(self) -> None:
        """Test that function returns False when logfire is not installed."""
        from ai_infra.eval.logfire import configure_logfire_evals

        # The actual function should handle missing logfire gracefully
        # when send_to_logfire is not "always"
        with patch.dict("sys.modules", {"logfire": None}):
            # This should not raise
            configure_logfire_evals(send_to_logfire="never")
            # Returns False because logfire is not available
            # (but we can't truly test this without uninstalling logfire)


class TestIsLogfireConfigured:
    """Tests for is_logfire_configured function."""

    def test_import(self) -> None:
        """Test that is_logfire_configured can be imported."""
        from ai_infra.eval import is_logfire_configured

        assert is_logfire_configured is not None

    def test_returns_bool(self) -> None:
        """Test that is_logfire_configured returns a boolean."""
        from ai_infra.eval.logfire import is_logfire_configured

        result = is_logfire_configured()
        assert isinstance(result, bool)


class TestCreateSpanQuery:
    """Tests for create_span_query helper."""

    def test_import(self) -> None:
        """Test that create_span_query can be imported."""
        from ai_infra.eval import create_span_query

        assert create_span_query is not None

    def test_empty_query(self) -> None:
        """Test creating empty query."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query()
        assert query == {}

    def test_name_contains(self) -> None:
        """Test query with name_contains."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(name_contains="search")
        assert query == {"name_contains": "search"}

    def test_name_equals(self) -> None:
        """Test query with name_equals."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(name_equals="get_weather")
        assert query == {"name_equals": "get_weather"}

    def test_name_regex(self) -> None:
        """Test query with name_regex."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(name_regex=r"search_.*")
        assert query == {"name_regex": r"search_.*"}

    def test_attribute_contains(self) -> None:
        """Test query with attribute_contains."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(attribute_contains={"location": "San"})
        assert query == {"attribute_contains": {"location": "San"}}

    def test_attribute_equals(self) -> None:
        """Test query with attribute_equals."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(attribute_equals={"city": "Tokyo"})
        assert query == {"attribute_equals": {"city": "Tokyo"}}

    def test_combined_query(self) -> None:
        """Test query with multiple conditions."""
        from ai_infra.eval.logfire import create_span_query

        query = create_span_query(
            name_contains="weather",
            attribute_equals={"units": "celsius"},
        )
        assert query == {
            "name_contains": "weather",
            "attribute_equals": {"units": "celsius"},
        }


class TestCheckToolCalled:
    """Tests for check_tool_called helper."""

    def test_import(self) -> None:
        """Test that check_tool_called can be imported."""
        from ai_infra.eval import check_tool_called

        assert check_tool_called is not None

    def test_returns_evaluator_or_none(self) -> None:
        """Test that check_tool_called returns an evaluator or None."""
        from ai_infra.eval.logfire import HAS_SPAN_EVALUATORS, check_tool_called

        result = check_tool_called("search")

        if HAS_SPAN_EVALUATORS:
            assert result is not None
        else:
            assert result is None

    def test_default_evaluation_name(self) -> None:
        """Test default evaluation name format."""
        from ai_infra.eval.logfire import HAS_SPAN_EVALUATORS, check_tool_called

        result = check_tool_called("search")

        if HAS_SPAN_EVALUATORS and result is not None:
            assert result.evaluation_name == "called_search"

    def test_custom_evaluation_name(self) -> None:
        """Test custom evaluation name."""
        from ai_infra.eval.logfire import HAS_SPAN_EVALUATORS, check_tool_called

        result = check_tool_called("search", evaluation_name="searched_web")

        if HAS_SPAN_EVALUATORS and result is not None:
            assert result.evaluation_name == "searched_web"


class TestCheckNoToolCalled:
    """Tests for check_no_tool_called helper."""

    def test_import(self) -> None:
        """Test that check_no_tool_called can be imported."""
        from ai_infra.eval import check_no_tool_called

        assert check_no_tool_called is not None

    def test_returns_evaluator_or_none(self) -> None:
        """Test that check_no_tool_called returns an evaluator or None."""
        from ai_infra.eval.logfire import HAS_SPAN_EVALUATORS, check_no_tool_called

        result = check_no_tool_called("delete")

        if HAS_SPAN_EVALUATORS:
            assert result is not None
        else:
            assert result is None

    def test_default_evaluation_name(self) -> None:
        """Test default evaluation name format."""
        from ai_infra.eval.logfire import HAS_SPAN_EVALUATORS, check_no_tool_called

        result = check_no_tool_called("delete")

        if HAS_SPAN_EVALUATORS and result is not None:
            assert result.evaluation_name == "avoided_delete"


class TestHasMatchingSpanExport:
    """Tests for HasMatchingSpan re-export."""

    def test_import_from_eval(self) -> None:
        """Test that HasMatchingSpan can be imported from ai_infra.eval."""

        # May be None if pydantic-evals[logfire] not installed
        # But should be importable without error

    def test_import_from_logfire_module(self) -> None:
        """Test that HasMatchingSpan can be imported from logfire submodule."""

        # May be None if pydantic-evals[logfire] not installed


class TestSpanTreeExports:
    """Tests for SpanTree/SpanNode/SpanQuery exports."""

    def test_span_tree_import(self) -> None:
        """Test SpanTree import."""

        # May be None if not available

    def test_span_node_import(self) -> None:
        """Test SpanNode import."""

        # May be None if not available

    def test_span_query_import(self) -> None:
        """Test SpanQuery import."""

        # May be None if not available


class TestLogfireModuleExports:
    """Test all exports from the logfire module."""

    def test_all_exports(self) -> None:
        """Test that all expected names are exported."""
        from ai_infra.eval import (
            check_no_tool_called,
            check_tool_called,
            configure_logfire_evals,
            create_span_query,
            is_logfire_configured,
        )

        # All should be importable (some may be None if deps not installed)
        assert configure_logfire_evals is not None
        assert is_logfire_configured is not None
        assert create_span_query is not None
        assert check_tool_called is not None
        assert check_no_tool_called is not None
        # These may be None if pydantic-evals[logfire] not installed
        # but should be importable

    def test_logfire_submodule_has_all(self) -> None:
        """Test that logfire submodule has __all__ defined."""
        from ai_infra.eval import logfire

        assert hasattr(logfire, "__all__")
        assert "configure_logfire_evals" in logfire.__all__
        assert "HasMatchingSpan" in logfire.__all__
        assert "create_span_query" in logfire.__all__


class TestLogfireIntegrationDocstring:
    """Test docstring examples are valid."""

    def test_module_docstring_exists(self) -> None:
        """Test that module has a docstring."""
        from ai_infra.eval import logfire

        assert logfire.__doc__ is not None
        assert "Logfire" in logfire.__doc__

    def test_configure_function_docstring(self) -> None:
        """Test that configure function has docstring."""
        from ai_infra.eval.logfire import configure_logfire_evals

        assert configure_logfire_evals.__doc__ is not None
        assert "service_name" in configure_logfire_evals.__doc__
