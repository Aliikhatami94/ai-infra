"""RAG (Retrieval-Augmented Generation) evaluation helpers.

This module provides specialized evaluators and helpers for evaluating RAG pipelines,
including retriever accuracy metrics and end-to-end RAG evaluation.

Standard RAG metrics implemented:
- RetrievalPrecision: Fraction of retrieved docs that are relevant
- RetrievalRecall: Fraction of relevant docs that were retrieved
- DocumentHitRate: Whether any expected doc was retrieved (Hit@K)

Example - Evaluate retriever with precision/recall:
    >>> from ai_infra.eval import Dataset, Case
    >>> from ai_infra.eval.rag import RetrievalPrecision, RetrievalRecall
    >>>
    >>> dataset = Dataset(
    ...     cases=[
    ...         Case(
    ...             inputs="refund policy",
    ...             expected_output=["doc1", "doc3"],  # Expected doc IDs
    ...         ),
    ...     ],
    ...     evaluators=[
    ...         RetrievalPrecision(),
    ...         RetrievalRecall(),
    ...     ],
    ... )

Example - End-to-end RAG evaluation:
    >>> from ai_infra.eval.rag import evaluate_rag_pipeline
    >>>
    >>> report = evaluate_rag_pipeline(
    ...     retriever=my_retriever,
    ...     generator=my_agent,
    ...     dataset=rag_dataset,
    ... )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_evals import Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

if TYPE_CHECKING:
    from ai_infra.llm.agent import Agent
    from ai_infra.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalPrecision(Evaluator[Any, list[str]]):
    """Evaluator that measures retrieval precision.

    Precision = (# of relevant retrieved docs) / (# of retrieved docs)

    This evaluator expects:
    - output: List of retrieved document IDs/identifiers
    - expected_output: List of relevant document IDs

    Example:
        >>> evaluator = RetrievalPrecision()
        >>> # If retrieved = ["doc1", "doc2", "doc3"] and relevant = ["doc1", "doc3"]
        >>> # Precision = 2/3 = 0.667
    """

    def evaluate(self, ctx: EvaluatorContext[Any, list[str]]) -> EvaluationReason:
        """Calculate precision score."""
        retrieved = ctx.output or []
        expected = ctx.expected_output or []

        if not retrieved:
            return EvaluationReason(
                value=1.0 if not expected else 0.0,
                reason="No documents retrieved",
            )

        # Convert to sets for comparison
        retrieved_set = set(retrieved)
        expected_set = set(expected)

        # Precision = relevant retrieved / total retrieved
        relevant_retrieved = len(retrieved_set & expected_set)
        precision = relevant_retrieved / len(retrieved_set)

        return EvaluationReason(
            value=precision,
            reason=f"Retrieved {relevant_retrieved}/{len(retrieved_set)} relevant docs (precision={precision:.2%})",
        )


@dataclass
class RetrievalRecall(Evaluator[Any, list[str]]):
    """Evaluator that measures retrieval recall.

    Recall = (# of relevant retrieved docs) / (# of total relevant docs)

    This evaluator expects:
    - output: List of retrieved document IDs/identifiers
    - expected_output: List of all relevant document IDs

    Example:
        >>> evaluator = RetrievalRecall()
        >>> # If retrieved = ["doc1", "doc2"] and relevant = ["doc1", "doc3", "doc5"]
        >>> # Recall = 1/3 = 0.333
    """

    def evaluate(self, ctx: EvaluatorContext[Any, list[str]]) -> EvaluationReason:
        """Calculate recall score."""
        retrieved = ctx.output or []
        expected = ctx.expected_output or []

        if not expected:
            return EvaluationReason(
                value=1.0,
                reason="No relevant documents defined (skipped)",
            )

        # Convert to sets for comparison
        retrieved_set = set(retrieved)
        expected_set = set(expected)

        # Recall = relevant retrieved / total relevant
        relevant_retrieved = len(retrieved_set & expected_set)
        recall = relevant_retrieved / len(expected_set)

        return EvaluationReason(
            value=recall,
            reason=f"Found {relevant_retrieved}/{len(expected_set)} relevant docs (recall={recall:.2%})",
        )


@dataclass
class DocumentHitRate(Evaluator[Any, list[str]]):
    """Evaluator that measures if any expected document was retrieved (Hit@K).

    Returns 1.0 if at least one expected document is in the retrieved set,
    0.0 otherwise. This is useful for measuring retrieval success at a
    high level.

    Example:
        >>> evaluator = DocumentHitRate()
        >>> # If retrieved = ["doc2", "doc5"] and expected = ["doc1", "doc5"]
        >>> # Hit = True (doc5 was found), score = 1.0
    """

    def evaluate(self, ctx: EvaluatorContext[Any, list[str]]) -> EvaluationReason:
        """Check if any expected document was retrieved."""
        retrieved = ctx.output or []
        expected = ctx.expected_output or []

        if not expected:
            return EvaluationReason(
                value=1.0,
                reason="No expected documents defined (skipped)",
            )

        retrieved_set = set(retrieved)
        expected_set = set(expected)

        hit = bool(retrieved_set & expected_set)
        hits_found = list(retrieved_set & expected_set)

        return EvaluationReason(
            value=1.0 if hit else 0.0,
            reason=f"Hit: {hit} (found: {hits_found[:3]}{'...' if len(hits_found) > 3 else ''})",
        )


@dataclass
class MRR(Evaluator[Any, list[str]]):
    """Mean Reciprocal Rank evaluator.

    MRR = 1 / rank_of_first_relevant_doc

    If no relevant document is found, returns 0.0.

    Example:
        >>> evaluator = MRR()
        >>> # If retrieved = ["doc2", "doc3", "doc1"] and expected = ["doc1"]
        >>> # First relevant doc (doc1) is at position 3, so MRR = 1/3 = 0.333
    """

    def evaluate(self, ctx: EvaluatorContext[Any, list[str]]) -> EvaluationReason:
        """Calculate MRR score."""
        retrieved = ctx.output or []
        expected = ctx.expected_output or []

        if not expected:
            return EvaluationReason(
                value=1.0,
                reason="No expected documents defined (skipped)",
            )

        expected_set = set(expected)

        # Find rank of first relevant document (1-indexed)
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in expected_set:
                mrr = 1.0 / rank
                return EvaluationReason(
                    value=mrr,
                    reason=f"First relevant doc at rank {rank} (MRR={mrr:.3f})",
                )

        return EvaluationReason(
            value=0.0,
            reason="No relevant document found in retrieved results",
        )


@dataclass
class NDCG(Evaluator[Any, list[str]]):
    """Normalized Discounted Cumulative Gain evaluator.

    NDCG measures ranking quality, giving higher scores when relevant
    documents appear earlier in the results.

    Uses binary relevance: 1 if doc is in expected_output, 0 otherwise.

    Example:
        >>> evaluator = NDCG()
        >>> # Higher score when relevant docs are ranked first
    """

    k: int | None = None  # Limit to top-k results (None = all)

    def evaluate(self, ctx: EvaluatorContext[Any, list[str]]) -> EvaluationReason:
        """Calculate NDCG score."""
        import math

        retrieved = ctx.output or []
        expected = ctx.expected_output or []

        if not expected:
            return EvaluationReason(
                value=1.0,
                reason="No expected documents defined (skipped)",
            )

        expected_set = set(expected)

        # Apply k limit if specified
        if self.k is not None:
            retrieved = retrieved[: self.k]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            rel = 1.0 if doc_id in expected_set else 0.0
            dcg += rel / math.log2(i + 2)  # +2 because i starts at 0, log base is i+1+1

        # Calculate ideal DCG (all relevant docs ranked first)
        ideal_retrieved = min(len(expected_set), len(retrieved))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_retrieved))

        if idcg == 0:
            return EvaluationReason(
                value=0.0,
                reason="Cannot calculate NDCG (no relevant docs possible)",
            )

        ndcg = dcg / idcg

        return EvaluationReason(
            value=ndcg,
            reason=f"NDCG@{len(retrieved)}={ndcg:.3f}",
        )


async def evaluate_rag_pipeline_async(
    retriever: Retriever,
    generator: Agent | Callable[[str, list[str]], str],
    dataset: Dataset[str, str],
    *,
    k: int = 5,
    concurrency: int | None = 5,
    progress: bool = True,
) -> EvaluationReport:
    """Evaluate an end-to-end RAG pipeline.

    This function evaluates the complete RAG flow:
    1. Retriever fetches relevant documents
    2. Generator produces an answer using retrieved context
    3. Answer is evaluated against expected output

    The retrieved documents are passed to the generator and also stored
    in the evaluation metadata for use with RAGFaithfulness evaluator.

    Args:
        retriever: The ai-infra Retriever to fetch documents.
        generator: Either an ai-infra Agent or a callable that takes
            (query, context_docs) and returns the generated answer.
        dataset: A pydantic-evals Dataset with questions and expected answers.
        k: Number of documents to retrieve per query (default: 5).
        concurrency: Number of concurrent evaluations (default: 5).
        progress: Show progress bar during evaluation (default: True).

    Returns:
        EvaluationReport from pydantic-evals with RAG-specific metrics.

    Example:
        >>> from ai_infra import Agent, Retriever
        >>> from ai_infra.eval import Dataset, Case
        >>> from ai_infra.eval.rag import evaluate_rag_pipeline_async
        >>>
        >>> retriever = Retriever()
        >>> agent = Agent(model="gpt-4o-mini")
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="What is the refund policy?", expected_output="30 days"),
        ...     ],
        ... )
        >>> report = await evaluate_rag_pipeline_async(retriever, agent, dataset)
    """

    async def task(query: str) -> str:
        """RAG pipeline task function."""
        # Step 1: Retrieve relevant documents
        docs = await asyncio.to_thread(retriever.search, query, k=k)

        # Step 2: Generate answer with context
        if callable(generator) and not hasattr(generator, "run"):
            # Plain callable
            if asyncio.iscoroutinefunction(generator):
                answer = str(await generator(query, docs))
            else:
                answer = str(await asyncio.to_thread(generator, query, docs))
        else:
            # ai-infra Agent
            context = "\n\n".join(docs)
            prompt = (
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
            )
            result = await asyncio.to_thread(generator.run, prompt)
            answer = str(result.content) if hasattr(result, "content") else str(result)

        return answer

    # Run evaluation
    report = await dataset.evaluate(
        task,
        max_concurrency=concurrency,
        progress=progress,
    )

    return report


def evaluate_rag_pipeline(
    retriever: Retriever,
    generator: Agent | Callable[[str, list[str]], str],
    dataset: Dataset[str, str],
    *,
    k: int = 5,
    concurrency: int | None = 5,
    progress: bool = True,
) -> EvaluationReport:
    """Evaluate an end-to-end RAG pipeline (sync version).

    This function evaluates the complete RAG flow:
    1. Retriever fetches relevant documents
    2. Generator produces an answer using retrieved context
    3. Answer is evaluated against expected output

    Args:
        retriever: The ai-infra Retriever to fetch documents.
        generator: Either an ai-infra Agent or a callable that takes
            (query, context_docs) and returns the generated answer.
        dataset: A pydantic-evals Dataset with questions and expected answers.
        k: Number of documents to retrieve per query (default: 5).
        concurrency: Number of concurrent evaluations (default: 5).
        progress: Show progress bar during evaluation (default: True).

    Returns:
        EvaluationReport from pydantic-evals with RAG-specific metrics.

    Example:
        >>> from ai_infra import Agent, Retriever
        >>> from ai_infra.eval import Dataset, Case
        >>> from ai_infra.eval.rag import evaluate_rag_pipeline
        >>>
        >>> retriever = Retriever()
        >>> agent = Agent(model="gpt-4o-mini")
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="What is the refund policy?", expected_output="30 days"),
        ...     ],
        ... )
        >>> report = evaluate_rag_pipeline(retriever, agent, dataset)
    """

    def task(query: str) -> str:
        """RAG pipeline task function."""
        # Step 1: Retrieve relevant documents
        docs = retriever.search(query, k=k)

        # Step 2: Generate answer with context
        if callable(generator) and not hasattr(generator, "run"):
            # Plain callable
            answer = generator(query, docs)
        else:
            # ai-infra Agent
            context = "\n\n".join(docs)
            prompt = (
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
            )
            result = generator.run(prompt)
            answer = str(result.content) if hasattr(result, "content") else str(result)

        return answer

    return dataset.evaluate_sync(
        task,
        max_concurrency=concurrency,
        progress=progress,
    )


def create_retrieval_dataset(
    queries: list[str],
    expected_docs: list[list[str]],
    *,
    evaluators: list[Evaluator] | None = None,
) -> Dataset[str, list[str]]:
    """Helper function to create a dataset for retriever evaluation.

    This is a convenience function for creating datasets where the expected
    output is a list of document IDs that should be retrieved.

    Args:
        queries: List of search queries.
        expected_docs: List of expected document ID lists, one per query.
        evaluators: Optional list of evaluators. Defaults to
            [RetrievalPrecision(), RetrievalRecall(), DocumentHitRate()].

    Returns:
        A pydantic-evals Dataset configured for retrieval evaluation.

    Example:
        >>> from ai_infra.eval.rag import create_retrieval_dataset
        >>>
        >>> dataset = create_retrieval_dataset(
        ...     queries=["refund policy", "shipping info"],
        ...     expected_docs=[["doc1", "doc3"], ["doc2", "doc5"]],
        ... )
    """
    from pydantic_evals import Case

    if len(queries) != len(expected_docs):
        msg = f"Number of queries ({len(queries)}) must match expected_docs ({len(expected_docs)})"
        raise ValueError(msg)

    cases = [
        Case(inputs=query, expected_output=docs) for query, docs in zip(queries, expected_docs)
    ]

    if evaluators is None:
        evaluators = [
            RetrievalPrecision(),
            RetrievalRecall(),
            DocumentHitRate(),
        ]

    return Dataset(cases=cases, evaluators=evaluators)
