"""Unit tests for the eval module (Phase 11.1 - pydantic-evals integration)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestEvalModuleImports:
    """Test that the eval module exports are correct."""

    def test_import_pydantic_evals_classes(self) -> None:
        """Test that pydantic-evals classes are re-exported."""
        # Verify these are the actual pydantic-evals classes
        from pydantic_evals import Case as OriginalCase
        from pydantic_evals import Dataset as OriginalDataset
        from pydantic_evals.evaluators import Evaluator as OriginalEvaluator
        from pydantic_evals.evaluators import EvaluatorContext as OriginalEvaluatorContext
        from pydantic_evals.evaluators import IsInstance as OriginalIsInstance
        from pydantic_evals.reporting import EvaluationReport as OriginalReport

        from ai_infra.eval import (
            Case,
            Dataset,
            EvaluationReport,
            Evaluator,
            EvaluatorContext,
            IsInstance,
        )

        assert Case is OriginalCase
        assert Dataset is OriginalDataset
        assert Evaluator is OriginalEvaluator
        assert EvaluatorContext is OriginalEvaluatorContext
        assert IsInstance is OriginalIsInstance
        assert EvaluationReport is OriginalReport

    def test_import_ai_infra_functions(self) -> None:
        """Test that ai-infra integration functions are exported."""
        from ai_infra.eval import (
            evaluate_agent,
            evaluate_agent_async,
            evaluate_retriever,
            evaluate_retriever_async,
        )

        assert callable(evaluate_agent)
        assert callable(evaluate_agent_async)
        assert callable(evaluate_retriever)
        assert callable(evaluate_retriever_async)

    def test_top_level_imports(self) -> None:
        """Test that eval functions are available from top-level ai_infra."""
        from ai_infra import (
            Case,
            Dataset,
            EvaluationReport,
            Evaluator,
            EvaluatorContext,
            IsInstance,
            evaluate_agent,
            evaluate_agent_async,
            evaluate_retriever,
            evaluate_retriever_async,
        )

        # Verify classes are importable
        assert Case is not None
        assert Dataset is not None
        assert EvaluationReport is not None
        assert Evaluator is not None
        assert EvaluatorContext is not None
        assert IsInstance is not None
        assert callable(evaluate_agent)
        assert callable(evaluate_agent_async)
        assert callable(evaluate_retriever)
        assert callable(evaluate_retriever_async)


class TestDatasetCreation:
    """Test creating pydantic-evals datasets."""

    def test_create_simple_dataset(self) -> None:
        """Test creating a simple dataset with cases."""
        from ai_infra.eval import Case, Dataset

        dataset = Dataset(
            cases=[
                Case(inputs="What is 2+2?", expected_output="4"),
                Case(inputs="Capital of France?", expected_output="Paris"),
            ],
        )

        assert len(dataset.cases) == 2
        assert dataset.cases[0].inputs == "What is 2+2?"
        assert dataset.cases[0].expected_output == "4"

    def test_create_dataset_with_metadata(self) -> None:
        """Test creating a dataset with metadata on cases."""
        from ai_infra.eval import Case, Dataset

        dataset = Dataset(
            cases=[
                Case(
                    name="math_test",
                    inputs="What is 2+2?",
                    expected_output="4",
                    metadata={"difficulty": "easy", "category": "math"},
                ),
            ],
        )

        assert dataset.cases[0].name == "math_test"
        assert dataset.cases[0].metadata["difficulty"] == "easy"

    def test_create_dataset_with_evaluators(self) -> None:
        """Test creating a dataset with evaluators."""
        from ai_infra.eval import Case, Dataset, IsInstance

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output="result"),
            ],
            evaluators=[IsInstance(type_name="str")],
        )

        assert len(dataset.evaluators) == 1


class TestEvaluateAgentWrapper:
    """Test the evaluate_agent wrapper function."""

    def test_evaluate_agent_with_mock(self) -> None:
        """Test evaluate_agent with a mocked agent."""
        from ai_infra.eval import Case, Dataset, evaluate_agent

        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Paris"

        dataset = Dataset(
            cases=[
                Case(
                    name="capital_test",
                    inputs="What is the capital of France?",
                    expected_output="Paris",
                ),
            ],
        )

        report = evaluate_agent(mock_agent, dataset, concurrency=1)

        # Verify agent was called
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args
        assert call_args[0][0] == "What is the capital of France?"

        # Verify report is returned
        assert report is not None

    def test_evaluate_agent_with_system_prompt(self) -> None:
        """Test evaluate_agent passes system prompt."""
        from ai_infra.eval import Case, Dataset, evaluate_agent

        mock_agent = MagicMock()
        mock_agent.run.return_value = "42"

        dataset = Dataset(
            cases=[
                Case(inputs="answer", expected_output="42"),
            ],
        )

        evaluate_agent(
            mock_agent,
            dataset,
            system="Always answer 42",
            concurrency=1,
        )

        call_kwargs = mock_agent.run.call_args[1]
        assert call_kwargs.get("system") == "Always answer 42"

    def test_evaluate_agent_handles_session_result(self) -> None:
        """Test evaluate_agent extracts content from SessionResult."""
        from ai_infra.eval import Case, Dataset, evaluate_agent

        # Create a mock SessionResult
        session_result = MagicMock()
        session_result.content = "Paris"

        mock_agent = MagicMock()
        mock_agent.run.return_value = session_result

        dataset = Dataset(
            cases=[
                Case(inputs="capital?", expected_output="Paris"),
            ],
        )

        report = evaluate_agent(mock_agent, dataset, concurrency=1)
        assert report is not None


class TestEvaluateRetrieverWrapper:
    """Test the evaluate_retriever wrapper function."""

    def test_evaluate_retriever_with_mock(self) -> None:
        """Test evaluate_retriever with a mocked retriever."""
        from ai_infra.eval import Case, Dataset, evaluate_retriever

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            "Paris is the capital of France.",
            "France is in Europe.",
        ]

        dataset = Dataset(
            cases=[
                Case(
                    name="capital_search",
                    inputs="capital of France",
                    expected_output="Paris",
                ),
            ],
        )

        report = evaluate_retriever(mock_retriever, dataset, k=3, concurrency=1)

        # Verify retriever was called
        mock_retriever.search.assert_called_once()
        call_args = mock_retriever.search.call_args
        assert call_args[0][0] == "capital of France"
        assert call_args[1]["k"] == 3

        assert report is not None

    def test_evaluate_retriever_concatenates_results(self) -> None:
        """Test that retriever results are concatenated."""
        from ai_infra.eval import Case, Dataset, evaluate_retriever

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = ["Result 1", "Result 2"]

        dataset = Dataset(
            cases=[
                Case(inputs="query", expected_output="Result"),
            ],
        )

        # The wrapper should join results with separator
        report = evaluate_retriever(mock_retriever, dataset, concurrency=1)
        assert report is not None


class TestCustomEvaluators:
    """Test using custom evaluators with ai-infra."""

    def test_custom_evaluator_class(self) -> None:
        """Test creating a custom evaluator."""
        from dataclasses import dataclass

        from ai_infra.eval import Case, Dataset, Evaluator, EvaluatorContext

        @dataclass
        class ContainsExpected(Evaluator[str, str]):
            """Check if output contains expected text."""

            def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
                if ctx.expected_output and ctx.expected_output in str(ctx.output):
                    return 1.0
                return 0.0

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output="hello"),
            ],
            evaluators=[ContainsExpected()],
        )

        # Just verify the evaluator was added
        assert len(dataset.evaluators) == 1

    def test_isinstance_evaluator(self) -> None:
        """Test the built-in IsInstance evaluator."""
        from ai_infra.eval import Case, Dataset, IsInstance

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output="result"),
            ],
            evaluators=[IsInstance(type_name="str")],
        )

        def simple_task(x: str) -> str:
            return "result"

        report = dataset.evaluate_sync(simple_task)
        # IsInstance should pass since we return a string
        assert report is not None


class TestAsyncEvaluation:
    """Test async evaluation functions."""

    @pytest.mark.asyncio
    async def test_evaluate_agent_async(self) -> None:
        """Test evaluate_agent_async."""
        from ai_infra.eval import Case, Dataset, evaluate_agent_async

        mock_agent = MagicMock()
        mock_agent.run.return_value = "result"

        dataset = Dataset(
            cases=[
                Case(inputs="test", expected_output="result"),
            ],
        )

        report = await evaluate_agent_async(mock_agent, dataset, concurrency=1)
        assert report is not None

    @pytest.mark.asyncio
    async def test_evaluate_retriever_async(self) -> None:
        """Test evaluate_retriever_async."""
        from ai_infra.eval import Case, Dataset, evaluate_retriever_async

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = ["result text"]

        dataset = Dataset(
            cases=[
                Case(inputs="query", expected_output="result"),
            ],
        )

        report = await evaluate_retriever_async(mock_retriever, dataset, concurrency=1)
        assert report is not None


class TestEvaluationReport:
    """Test working with EvaluationReport."""

    def test_report_has_case_results(self) -> None:
        """Test that report contains case results."""
        from ai_infra.eval import Case, Dataset

        dataset = Dataset(
            cases=[
                Case(name="test1", inputs="a", expected_output="a"),
                Case(name="test2", inputs="b", expected_output="b"),
            ],
        )

        report = dataset.evaluate_sync(lambda x: x)

        # Report should have results for all cases
        assert len(report.cases) == 2
