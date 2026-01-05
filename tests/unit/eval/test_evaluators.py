"""Unit tests for ai-infra custom evaluators (Phase 11.2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic_evals.evaluators import EvaluatorContext


class TestSemanticSimilarityEvaluator:
    """Tests for SemanticSimilarity evaluator."""

    def test_import(self) -> None:
        """Test that SemanticSimilarity can be imported."""
        from ai_infra.eval import SemanticSimilarity

        assert SemanticSimilarity is not None

    def test_top_level_import(self) -> None:
        """Test that SemanticSimilarity is available from ai_infra."""
        from ai_infra import SemanticSimilarity

        assert SemanticSimilarity is not None

    def test_cosine_similarity_function(self) -> None:
        """Test the cosine similarity helper function."""
        from ai_infra.eval.evaluators import _cosine_similarity

        # Identical vectors should have similarity 1.0
        vec1 = [1.0, 0.0, 0.0]
        assert _cosine_similarity(vec1, vec1) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0.0
        vec2 = [0.0, 1.0, 0.0]
        assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0)

        # Opposite vectors should have similarity -1.0
        vec3 = [-1.0, 0.0, 0.0]
        assert _cosine_similarity(vec1, vec3) == pytest.approx(-1.0)

    def test_semantic_similarity_with_mocked_embeddings(self) -> None:
        """Test SemanticSimilarity with mocked embeddings."""
        from ai_infra.eval import SemanticSimilarity

        # Create mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.embed.side_effect = [
            [1.0, 0.0, 0.0],  # output embedding
            [0.9, 0.1, 0.0],  # expected embedding (similar)
        ]

        evaluator = SemanticSimilarity(threshold=0.8, embeddings=mock_embeddings)

        ctx = EvaluatorContext(
            name="test",
            inputs="test input",
            metadata=None,
            expected_output="expected output",
            output="actual output",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)

        # Should have called embed twice
        assert mock_embeddings.embed.call_count == 2
        # Result should be an EvaluationReason
        assert hasattr(result, "value")
        assert hasattr(result, "reason")

    def test_semantic_similarity_skips_when_no_expected(self) -> None:
        """Test that SemanticSimilarity skips when no expected output."""
        from ai_infra.eval import SemanticSimilarity

        evaluator = SemanticSimilarity()

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="output",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value is True
        assert "Skipped" in result.reason


class TestToolUsageEvaluator:
    """Tests for ToolUsageEvaluator."""

    def test_import(self) -> None:
        """Test that ToolUsageEvaluator can be imported."""
        from ai_infra.eval import ToolUsageEvaluator

        assert ToolUsageEvaluator is not None

    def test_top_level_import(self) -> None:
        """Test that ToolUsageEvaluator is available from ai_infra."""
        from ai_infra import ToolUsageEvaluator

        assert ToolUsageEvaluator is not None

    def test_check_expected_tools_all_found(self) -> None:
        """Test checking expected tools when all are found."""
        from ai_infra.eval import ToolUsageEvaluator

        evaluator = ToolUsageEvaluator(
            expected_tools=["search", "calculate"],
            require_all=True,
        )

        tools_called = ["search_tool", "calculate_value"]
        assert evaluator._check_expected_tools(tools_called) is True

    def test_check_expected_tools_missing(self) -> None:
        """Test checking expected tools when some are missing."""
        from ai_infra.eval import ToolUsageEvaluator

        evaluator = ToolUsageEvaluator(
            expected_tools=["search", "calculate"],
            require_all=True,
        )

        tools_called = ["search_tool"]
        assert evaluator._check_expected_tools(tools_called) is False

    def test_check_forbidden_tools_none_called(self) -> None:
        """Test checking forbidden tools when none are called."""
        from ai_infra.eval import ToolUsageEvaluator

        evaluator = ToolUsageEvaluator(
            forbidden_tools=["delete", "drop"],
        )

        tools_called = ["search_tool", "read_file"]
        assert evaluator._check_forbidden_tools(tools_called) is True

    def test_check_forbidden_tools_called(self) -> None:
        """Test checking forbidden tools when one is called."""
        from ai_infra.eval import ToolUsageEvaluator

        evaluator = ToolUsageEvaluator(
            forbidden_tools=["delete", "drop"],
        )

        tools_called = ["search_tool", "delete_user"]
        assert evaluator._check_forbidden_tools(tools_called) is False

    def test_evaluate_without_span_tree(self) -> None:
        """Test evaluation when no span tree is available."""
        from ai_infra.eval import ToolUsageEvaluator

        evaluator = ToolUsageEvaluator(expected_tools=["search"])

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="output",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert "called_expected" in result
        assert "avoided_forbidden" in result
        assert "tools_called" in result


class TestRAGFaithfulness:
    """Tests for RAGFaithfulness evaluator."""

    def test_import(self) -> None:
        """Test that RAGFaithfulness can be imported."""
        from ai_infra.eval import RAGFaithfulness

        assert RAGFaithfulness is not None

    def test_top_level_import(self) -> None:
        """Test that RAGFaithfulness is available from ai_infra."""
        from ai_infra import RAGFaithfulness

        assert RAGFaithfulness is not None

    @pytest.mark.asyncio
    async def test_skips_when_no_context(self) -> None:
        """Test that RAGFaithfulness skips when no context in metadata."""
        from ai_infra.eval import RAGFaithfulness

        evaluator = RAGFaithfulness()

        ctx = EvaluatorContext(
            name="test",
            inputs="What is the policy?",
            metadata=None,
            expected_output=None,
            output="The policy is...",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = await evaluator.evaluate(ctx)
        assert result.value is True
        assert "Skipped" in result.reason

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_output(self) -> None:
        """Test that RAGFaithfulness returns 0 for empty output."""
        from ai_infra.eval import RAGFaithfulness

        evaluator = RAGFaithfulness()

        ctx = EvaluatorContext(
            name="test",
            inputs="What is the policy?",
            metadata={"context": "The policy is 30 day refund."},
            expected_output=None,
            output="",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = await evaluator.evaluate(ctx)
        assert result.value == 0.0


class TestContainsExpected:
    """Tests for ContainsExpected evaluator."""

    def test_import(self) -> None:
        """Test that ContainsExpected can be imported."""
        from ai_infra.eval import ContainsExpected

        assert ContainsExpected is not None

    def test_contains_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        from ai_infra.eval import ContainsExpected

        evaluator = ContainsExpected(case_sensitive=False)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output="Paris",
            output="The capital is PARIS.",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        assert evaluator.evaluate(ctx) is True

    def test_contains_case_sensitive(self) -> None:
        """Test case-sensitive matching."""
        from ai_infra.eval import ContainsExpected

        evaluator = ContainsExpected(case_sensitive=True)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output="Paris",
            output="The capital is PARIS.",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        assert evaluator.evaluate(ctx) is False

    def test_contains_not_found(self) -> None:
        """Test when expected is not found."""
        from ai_infra.eval import ContainsExpected

        evaluator = ContainsExpected()

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output="Paris",
            output="The capital is London.",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        assert evaluator.evaluate(ctx) is False

    def test_skips_when_no_expected(self) -> None:
        """Test that it returns True when no expected output."""
        from ai_infra.eval import ContainsExpected

        evaluator = ContainsExpected()

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="Any output",
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        assert evaluator.evaluate(ctx) is True


class TestLengthInRange:
    """Tests for LengthInRange evaluator."""

    def test_import(self) -> None:
        """Test that LengthInRange can be imported."""
        from ai_infra.eval import LengthInRange

        assert LengthInRange is not None

    def test_length_in_range(self) -> None:
        """Test when length is within range."""
        from ai_infra.eval import LengthInRange

        evaluator = LengthInRange(min_length=5, max_length=20)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="Hello world",  # 11 characters
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value is True

    def test_length_too_short(self) -> None:
        """Test when output is too short."""
        from ai_infra.eval import LengthInRange

        evaluator = LengthInRange(min_length=10)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="Hi",  # 2 characters
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value is False
        assert "too short" in result.reason

    def test_length_too_long(self) -> None:
        """Test when output is too long."""
        from ai_infra.eval import LengthInRange

        evaluator = LengthInRange(max_length=5)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="Hello world",  # 11 characters
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value is False
        assert "too long" in result.reason

    def test_count_words(self) -> None:
        """Test word counting mode."""
        from ai_infra.eval import LengthInRange

        evaluator = LengthInRange(min_length=2, max_length=5, count_words=True)

        ctx = EvaluatorContext(
            name="test",
            inputs="test",
            metadata=None,
            expected_output=None,
            output="Hello world from AI",  # 4 words
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value is True
        assert "words" in result.reason


class TestEvaluatorIntegrationWithDataset:
    """Test evaluators work with pydantic-evals Dataset."""

    def test_contains_expected_with_dataset(self) -> None:
        """Test ContainsExpected works with Dataset.evaluate_sync."""
        from ai_infra.eval import Case, ContainsExpected, Dataset

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output="hello"),
            ],
            evaluators=[ContainsExpected()],
        )

        def task(x: str) -> str:
            return "hello world"

        report = dataset.evaluate_sync(task, progress=False)
        assert report is not None
        assert len(report.cases) == 1

    def test_length_in_range_with_dataset(self) -> None:
        """Test LengthInRange works with Dataset.evaluate_sync."""
        from ai_infra.eval import Case, Dataset, LengthInRange

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output=None),
            ],
            evaluators=[LengthInRange(min_length=5, max_length=50)],
        )

        def task(x: str) -> str:
            return "This is a test response"

        report = dataset.evaluate_sync(task, progress=False)
        assert report is not None
        assert len(report.cases) == 1


class TestCustomEvaluatorsExport:
    """Test that all custom evaluators are properly exported."""

    def test_all_evaluators_from_eval_module(self) -> None:
        """Test all evaluators can be imported from ai_infra.eval."""
        from ai_infra.eval import (
            ContainsExpected,
            LengthInRange,
            RAGFaithfulness,
            SemanticSimilarity,
            ToolUsageEvaluator,
        )

        assert ContainsExpected is not None
        assert LengthInRange is not None
        assert RAGFaithfulness is not None
        assert SemanticSimilarity is not None
        assert ToolUsageEvaluator is not None

    def test_all_evaluators_from_top_level(self) -> None:
        """Test all evaluators can be imported from ai_infra."""
        from ai_infra import (
            ContainsExpected,
            LengthInRange,
            RAGFaithfulness,
            SemanticSimilarity,
            ToolUsageEvaluator,
        )

        assert ContainsExpected is not None
        assert LengthInRange is not None
        assert RAGFaithfulness is not None
        assert SemanticSimilarity is not None
        assert ToolUsageEvaluator is not None
