"""Integration between ai-infra components and pydantic-evals.

This module provides wrapper functions that adapt ai-infra's Agent and Retriever
classes to work seamlessly with pydantic-evals' Dataset.evaluate() API.

Design Philosophy:
- Thin wrappers over pydantic-evals, not a replacement
- Type-safe with proper generics support
- Async-first with sync convenience wrappers
- Integrate with ai-infra's logging and tracing
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport

if TYPE_CHECKING:
    from ai_infra.llm.agent import Agent
    from ai_infra.retriever import Retriever

logger = logging.getLogger(__name__)


async def evaluate_agent_async(
    agent: Agent,
    dataset: Dataset[str, str],
    *,
    concurrency: int | None = 5,
    system: str | None = None,
    progress: bool = True,
    **agent_kwargs: Any,
) -> EvaluationReport:
    """Evaluate an ai-infra Agent against a pydantic-evals Dataset.

    This function wraps the agent's `run` method to work with pydantic-evals'
    Dataset.evaluate() API. Each case's `inputs` is passed as the prompt,
    and the agent's response is compared against `expected_output`.

    Args:
        agent: The ai-infra Agent to evaluate.
        dataset: A pydantic-evals Dataset with string inputs and expected outputs.
        concurrency: Number of concurrent evaluations (default: 5). Set to None
            for unlimited concurrency.
        system: Optional system prompt to use for all evaluations.
        progress: Show progress bar during evaluation (default: True).
        **agent_kwargs: Additional kwargs passed to agent.run().

    Returns:
        EvaluationReport from pydantic-evals with scores and assertions.

    Example:
        >>> from ai_infra import Agent
        >>> from ai_infra.eval import evaluate_agent_async
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> agent = Agent(model="gpt-4o-mini")
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="What is 2+2?", expected_output="4"),
        ...     ],
        ... )
        >>> report = await evaluate_agent_async(agent, dataset)
        >>> report.print()
    """

    async def task(prompt: str) -> str:
        """Task function that wraps agent.run() for pydantic-evals."""
        # Run the agent (synchronous - we're in async context but Agent.run is sync)
        result = await asyncio.to_thread(
            agent.run,
            prompt,
            system=system,
            **agent_kwargs,
        )

        # Handle SessionResult if session is configured
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    # Run evaluation using pydantic-evals
    report = await dataset.evaluate(
        task,
        max_concurrency=concurrency,
        progress=progress,
    )

    return report


def evaluate_agent(
    agent: Agent,
    dataset: Dataset[str, str],
    *,
    concurrency: int | None = 5,
    system: str | None = None,
    progress: bool = True,
    **agent_kwargs: Any,
) -> EvaluationReport:
    """Evaluate an ai-infra Agent against a pydantic-evals Dataset (sync).

    This is a synchronous wrapper that uses Dataset.evaluate_sync().

    Args:
        agent: The ai-infra Agent to evaluate.
        dataset: A pydantic-evals Dataset with string inputs and expected outputs.
        concurrency: Number of concurrent evaluations (default: 5). Set to None
            for unlimited concurrency.
        system: Optional system prompt to use for all evaluations.
        progress: Show progress bar during evaluation (default: True).
        **agent_kwargs: Additional kwargs passed to agent.run().

    Returns:
        EvaluationReport from pydantic-evals with scores and assertions.

    Example:
        >>> from ai_infra import Agent
        >>> from ai_infra.eval import evaluate_agent
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> agent = Agent(model="gpt-4o-mini")
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="What is 2+2?", expected_output="4"),
        ...     ],
        ... )
        >>> report = evaluate_agent(agent, dataset)
        >>> report.print()
    """
    return dataset.evaluate_sync(
        lambda prompt: _run_agent_sync(agent, prompt, system=system, **agent_kwargs),
        max_concurrency=concurrency,
        progress=progress,
    )


def _run_agent_sync(
    agent: Agent,
    prompt: str,
    *,
    system: str | None = None,
    **agent_kwargs: Any,
) -> str:
    """Run agent synchronously and extract string result."""
    result = agent.run(prompt, system=system, **agent_kwargs)

    # Handle SessionResult if session is configured
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


async def evaluate_retriever_async(
    retriever: Retriever,
    dataset: Dataset[str, str | list[str]],
    *,
    k: int = 5,
    concurrency: int | None = 10,
    progress: bool = True,
) -> EvaluationReport:
    """Evaluate an ai-infra Retriever against a pydantic-evals Dataset.

    This function wraps the retriever's `search` method to work with pydantic-evals.
    Each case's `inputs` is used as the search query, and results are compared
    against `expected_output`.

    The evaluator concatenates retrieved results and checks if the expected
    output text is contained within the results.

    Args:
        retriever: The ai-infra Retriever to evaluate.
        dataset: A pydantic-evals Dataset with string queries and expected outputs.
            expected_output can be:
            - A string: Check if any result contains this text
            - A list of strings: Check if all expected strings are found
        k: Number of results to retrieve per query (default: 5).
        concurrency: Number of concurrent evaluations (default: 10). Set to None
            for unlimited concurrency.
        progress: Show progress bar during evaluation (default: True).

    Returns:
        EvaluationReport from pydantic-evals with scores and assertions.

    Example:
        >>> from ai_infra import Retriever
        >>> from ai_infra.eval import evaluate_retriever_async
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> retriever = Retriever()
        >>> retriever.add("Paris is the capital of France.")
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="capital of France", expected_output="Paris"),
        ...     ],
        ... )
        >>> report = await evaluate_retriever_async(retriever, dataset)
    """

    async def task(query: str) -> str:
        """Task function that wraps retriever.search() for pydantic-evals."""
        # Run search (synchronous - we're in async context)
        results = await asyncio.to_thread(retriever.search, query, k=k)

        # Return concatenated results for text matching
        return "\n---\n".join(results)

    # Run evaluation
    report = await dataset.evaluate(
        task,
        max_concurrency=concurrency,
        progress=progress,
    )

    return report


def evaluate_retriever(
    retriever: Retriever,
    dataset: Dataset[str, str | list[str]],
    *,
    k: int = 5,
    concurrency: int | None = 10,
    progress: bool = True,
) -> EvaluationReport:
    """Evaluate an ai-infra Retriever against a pydantic-evals Dataset (sync).

    This is a synchronous wrapper that uses Dataset.evaluate_sync().

    Args:
        retriever: The ai-infra Retriever to evaluate.
        dataset: A pydantic-evals Dataset with string queries and expected outputs.
        k: Number of results to retrieve per query (default: 5).
        concurrency: Number of concurrent evaluations (default: 10). Set to None
            for unlimited concurrency.
        progress: Show progress bar during evaluation (default: True).

    Returns:
        EvaluationReport from pydantic-evals with scores and assertions.

    Example:
        >>> from ai_infra import Retriever
        >>> from ai_infra.eval import evaluate_retriever
        >>> from pydantic_evals import Case, Dataset
        >>>
        >>> retriever = Retriever()
        >>> retriever.add("Paris is the capital of France.")
        >>>
        >>> dataset = Dataset(
        ...     cases=[
        ...         Case(inputs="capital of France", expected_output="Paris"),
        ...     ],
        ... )
        >>> report = evaluate_retriever(retriever, dataset)
    """

    def task(query: str) -> str:
        """Task function that wraps retriever.search()."""
        results = retriever.search(query, k=k)
        return "\n---\n".join(results)

    return dataset.evaluate_sync(
        task,
        max_concurrency=concurrency,
        progress=progress,
    )
