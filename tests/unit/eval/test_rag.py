"""Unit tests for ai-infra RAG evaluation helpers (Phase 11.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic_evals.evaluators import EvaluatorContext


class TestRetrievalPrecision:
    """Tests for RetrievalPrecision evaluator."""

    def test_import(self) -> None:
        """Test that RetrievalPrecision can be imported."""
        from ai_infra.eval import RetrievalPrecision

        assert RetrievalPrecision is not None

    def test_perfect_precision(self) -> None:
        """Test precision when all retrieved docs are relevant."""
        from ai_infra.eval.rag import RetrievalPrecision

        evaluator = RetrievalPrecision()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2", "doc3"],
            output=["doc1", "doc2"],  # All retrieved are relevant
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)

    def test_partial_precision(self) -> None:
        """Test precision when some retrieved docs are not relevant."""
        from ai_infra.eval.rag import RetrievalPrecision

        evaluator = RetrievalPrecision()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc3"],
            output=["doc1", "doc2", "doc3"],  # 2/3 relevant
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(2 / 3)

    def test_zero_precision(self) -> None:
        """Test precision when no retrieved docs are relevant."""
        from ai_infra.eval.rag import RetrievalPrecision

        evaluator = RetrievalPrecision()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=["doc3", "doc4"],  # None relevant
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)

    def test_empty_retrieved(self) -> None:
        """Test precision when no docs are retrieved."""
        from ai_infra.eval.rag import RetrievalPrecision

        evaluator = RetrievalPrecision()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=[],
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)
        assert "No documents retrieved" in result.reason


class TestRetrievalRecall:
    """Tests for RetrievalRecall evaluator."""

    def test_import(self) -> None:
        """Test that RetrievalRecall can be imported."""
        from ai_infra.eval import RetrievalRecall

        assert RetrievalRecall is not None

    def test_perfect_recall(self) -> None:
        """Test recall when all relevant docs are retrieved."""
        from ai_infra.eval.rag import RetrievalRecall

        evaluator = RetrievalRecall()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=["doc1", "doc2", "doc3"],  # All relevant found
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        """Test recall when some relevant docs are missing."""
        from ai_infra.eval.rag import RetrievalRecall

        evaluator = RetrievalRecall()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2", "doc3"],
            output=["doc1", "doc4"],  # Only 1/3 relevant found
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1 / 3)

    def test_zero_recall(self) -> None:
        """Test recall when no relevant docs are found."""
        from ai_infra.eval.rag import RetrievalRecall

        evaluator = RetrievalRecall()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=["doc3", "doc4"],
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)

    def test_empty_expected(self) -> None:
        """Test recall when no expected docs (skip)."""
        from ai_infra.eval.rag import RetrievalRecall

        evaluator = RetrievalRecall()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=[],
            output=["doc1"],
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)
        assert "skipped" in result.reason.lower()


class TestDocumentHitRate:
    """Tests for DocumentHitRate evaluator."""

    def test_import(self) -> None:
        """Test that DocumentHitRate can be imported."""
        from ai_infra.eval import DocumentHitRate

        assert DocumentHitRate is not None

    def test_hit_found(self) -> None:
        """Test when at least one expected doc is found."""
        from ai_infra.eval.rag import DocumentHitRate

        evaluator = DocumentHitRate()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc5"],
            output=["doc2", "doc5", "doc3"],  # doc5 is a hit
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)

    def test_no_hit(self) -> None:
        """Test when no expected doc is found."""
        from ai_infra.eval.rag import DocumentHitRate

        evaluator = DocumentHitRate()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=["doc3", "doc4"],
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)


class TestMRR:
    """Tests for MRR (Mean Reciprocal Rank) evaluator."""

    def test_import(self) -> None:
        """Test that MRR can be imported."""
        from ai_infra.eval import MRR

        assert MRR is not None

    def test_first_position(self) -> None:
        """Test MRR when relevant doc is first."""
        from ai_infra.eval.rag import MRR

        evaluator = MRR()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1"],
            output=["doc1", "doc2", "doc3"],  # doc1 at rank 1
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)

    def test_third_position(self) -> None:
        """Test MRR when relevant doc is at rank 3."""
        from ai_infra.eval.rag import MRR

        evaluator = MRR()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1"],
            output=["doc2", "doc3", "doc1"],  # doc1 at rank 3
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        """Test MRR when relevant doc not found."""
        from ai_infra.eval.rag import MRR

        evaluator = MRR()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1"],
            output=["doc2", "doc3", "doc4"],
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)


class TestNDCG:
    """Tests for NDCG (Normalized Discounted Cumulative Gain) evaluator."""

    def test_import(self) -> None:
        """Test that NDCG can be imported."""
        from ai_infra.eval import NDCG

        assert NDCG is not None

    def test_perfect_ranking(self) -> None:
        """Test NDCG when all relevant docs are ranked first."""
        from ai_infra.eval.rag import NDCG

        evaluator = NDCG()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1", "doc2"],
            output=["doc1", "doc2", "doc3"],  # Perfect ranking
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)

    def test_imperfect_ranking(self) -> None:
        """Test NDCG when relevant docs are not ranked first."""
        from ai_infra.eval.rag import NDCG

        evaluator = NDCG()

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc1"],
            output=["doc2", "doc1"],  # Relevant doc at rank 2
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        # DCG = 1/log2(3) = 0.631, IDCG = 1/log2(2) = 1.0, NDCG = 0.631
        assert result.value < 1.0
        assert result.value > 0.5

    def test_with_k_limit(self) -> None:
        """Test NDCG with k limit."""
        from ai_infra.eval.rag import NDCG

        evaluator = NDCG(k=2)

        ctx = EvaluatorContext(
            name="test",
            inputs="query",
            metadata=None,
            expected_output=["doc3"],
            output=["doc1", "doc2", "doc3"],  # doc3 beyond k=2
            duration=0.1,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.0)  # No relevant in top-2


class TestCreateRetrievalDataset:
    """Tests for create_retrieval_dataset helper."""

    def test_import(self) -> None:
        """Test that create_retrieval_dataset can be imported."""
        from ai_infra.eval import create_retrieval_dataset

        assert create_retrieval_dataset is not None

    def test_creates_dataset(self) -> None:
        """Test that helper creates a proper dataset."""
        from ai_infra.eval.rag import create_retrieval_dataset

        dataset = create_retrieval_dataset(
            queries=["query1", "query2"],
            expected_docs=[["doc1", "doc2"], ["doc3"]],
        )

        assert len(dataset.cases) == 2
        assert dataset.cases[0].inputs == "query1"
        assert dataset.cases[0].expected_output == ["doc1", "doc2"]
        assert dataset.cases[1].inputs == "query2"
        assert dataset.cases[1].expected_output == ["doc3"]

    def test_default_evaluators(self) -> None:
        """Test that default evaluators are added."""
        from ai_infra.eval.rag import (
            DocumentHitRate,
            RetrievalPrecision,
            RetrievalRecall,
            create_retrieval_dataset,
        )

        dataset = create_retrieval_dataset(
            queries=["query1"],
            expected_docs=[["doc1"]],
        )

        # Should have 3 default evaluators
        assert len(dataset.evaluators) == 3

        evaluator_types = {type(e) for e in dataset.evaluators}
        assert RetrievalPrecision in evaluator_types
        assert RetrievalRecall in evaluator_types
        assert DocumentHitRate in evaluator_types

    def test_custom_evaluators(self) -> None:
        """Test with custom evaluators."""
        from ai_infra.eval.rag import MRR, create_retrieval_dataset

        custom_evaluators = [MRR()]

        dataset = create_retrieval_dataset(
            queries=["query1"],
            expected_docs=[["doc1"]],
            evaluators=custom_evaluators,
        )

        assert len(dataset.evaluators) == 1
        assert isinstance(dataset.evaluators[0], MRR)

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched query/expected_docs raises error."""
        from ai_infra.eval.rag import create_retrieval_dataset

        with pytest.raises(ValueError, match="must match"):
            create_retrieval_dataset(
                queries=["query1", "query2"],
                expected_docs=[["doc1"]],  # Only one list
            )


class TestEvaluateRagPipeline:
    """Tests for evaluate_rag_pipeline function."""

    def test_import(self) -> None:
        """Test that evaluate_rag_pipeline can be imported."""
        from ai_infra.eval import evaluate_rag_pipeline, evaluate_rag_pipeline_async

        assert evaluate_rag_pipeline is not None
        assert evaluate_rag_pipeline_async is not None

    def test_with_callable_generator(self) -> None:
        """Test evaluate_rag_pipeline with callable generator."""
        from ai_infra.eval import Case, Dataset
        from ai_infra.eval.rag import evaluate_rag_pipeline

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = ["Context about Paris"]

        # Simple generator function
        def generator(query: str, docs: list[str]) -> str:
            return f"Answer based on: {docs[0]}"

        dataset = Dataset(
            cases=[
                Case(inputs="What is the capital?", expected_output="Paris"),
            ],
        )

        report = evaluate_rag_pipeline(
            retriever=mock_retriever,
            generator=generator,
            dataset=dataset,
            k=3,
            progress=False,
        )

        assert report is not None
        assert len(report.cases) == 1
        mock_retriever.search.assert_called_once_with("What is the capital?", k=3)


class TestRAGExportsFromEvalModule:
    """Test that all RAG helpers are properly exported."""

    def test_all_exports_from_eval(self) -> None:
        """Test all RAG exports from ai_infra.eval."""
        from ai_infra.eval import (
            MRR,
            NDCG,
            DocumentHitRate,
            RetrievalPrecision,
            RetrievalRecall,
            create_retrieval_dataset,
            evaluate_rag_pipeline,
            evaluate_rag_pipeline_async,
        )

        assert RetrievalPrecision is not None
        assert RetrievalRecall is not None
        assert DocumentHitRate is not None
        assert MRR is not None
        assert NDCG is not None
        assert create_retrieval_dataset is not None
        assert evaluate_rag_pipeline is not None
        assert evaluate_rag_pipeline_async is not None
