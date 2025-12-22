#!/usr/bin/env python
"""Parallel Execution in Graphs: Concurrent Node Execution.

This example demonstrates:
- Running multiple nodes in parallel
- Fan-out and fan-in patterns
- Parallel data processing
- Combining parallel results
- Error handling in parallel execution

Parallel execution improves workflow performance by running
independent operations concurrently.

Key concepts:
- Fan-out: One node triggers multiple parallel nodes
- Fan-in: Multiple nodes converge to one
- State merging: Parallel results combined in state
- asyncio.gather: Underlying parallel mechanism
"""

import asyncio
import time
from typing import TypedDict

from ai_infra import Graph

# =============================================================================
# Example 1: Basic Fan-Out Pattern
# =============================================================================


def example_basic_fanout() -> None:
    """Simple fan-out: one source to multiple targets."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Fan-Out Pattern")
    print("=" * 60)

    def start(state: dict) -> dict:
        return {"input": state.get("data", "initial")}

    def process_a(state: dict) -> dict:
        return {"result_a": f"A processed: {state['input']}"}

    def process_b(state: dict) -> dict:
        return {"result_b": f"B processed: {state['input']}"}

    def process_c(state: dict) -> dict:
        return {"result_c": f"C processed: {state['input']}"}

    def combine(state: dict) -> dict:
        return {
            "combined": [
                state.get("result_a"),
                state.get("result_b"),
                state.get("result_c"),
            ]
        }

    # Fan-out: start connects to A, B, C
    # Fan-in: A, B, C all connect to combine
    graph = Graph(
        nodes={
            "start": start,
            "process_a": process_a,
            "process_b": process_b,
            "process_c": process_c,
            "combine": combine,
        },
        edges=[
            ("start", "process_a"),
            ("start", "process_b"),
            ("start", "process_c"),
            ("process_a", "combine"),
            ("process_b", "combine"),
            ("process_c", "combine"),
        ],
        entry="start",
    )

    result = graph.run({"data": "test_data"})

    print("\nInput: test_data")
    print(f"Combined results: {result.get('combined')}")


# =============================================================================
# Example 2: Async Parallel Processing
# =============================================================================


class ParallelState(TypedDict, total=False):
    """State for parallel async workflow."""

    urls: list[str]
    fetch_results: dict[str, str]
    processed: list[str]


async def example_async_parallel() -> None:
    """Parallel async operations with timing."""
    print("\n" + "=" * 60)
    print("Example 2: Async Parallel Processing")
    print("=" * 60)

    async def fetch_url(url: str) -> tuple[str, str]:
        """Simulate fetching a URL."""
        await asyncio.sleep(0.2)  # Simulate network latency
        return url, f"Content from {url}"

    async def parallel_fetch(state: ParallelState) -> dict:
        """Fetch all URLs in parallel."""
        start_time = time.time()

        # Fetch all URLs concurrently
        results = await asyncio.gather(*[fetch_url(url) for url in state["urls"]])

        elapsed = time.time() - start_time
        print(f"  Parallel fetch completed in {elapsed:.2f}s")

        return {"fetch_results": dict(results)}

    async def sequential_fetch_for_comparison(urls: list[str]) -> dict[str, str]:
        """Sequential fetch for timing comparison."""
        start_time = time.time()
        results = {}
        for url in urls:
            url, content = await fetch_url(url)
            results[url] = content
        elapsed = time.time() - start_time
        print(f"  Sequential fetch would take {elapsed:.2f}s")
        return results

    def process_results(state: ParallelState) -> dict:
        """Process fetched results."""
        processed = [f"Processed: {content}" for content in state["fetch_results"].values()]
        return {"processed": processed}

    graph = Graph(
        nodes={
            "fetch": parallel_fetch,
            "process": process_results,
        },
        edges=[
            ("fetch", "process"),
        ],
        entry="fetch",
        state_schema=ParallelState,
    )

    urls = [
        "https://api1.example.com",
        "https://api2.example.com",
        "https://api3.example.com",
        "https://api4.example.com",
        "https://api5.example.com",
    ]

    print(f"\nFetching {len(urls)} URLs...")

    # Compare parallel vs sequential timing
    await sequential_fetch_for_comparison(urls)
    result = await graph.arun({"urls": urls})

    print(f"\nProcessed {len(result.get('processed', []))} results")


# =============================================================================
# Example 3: Parallel with State Aggregation
# =============================================================================


class AggregateState(TypedDict, total=False):
    """State for aggregation workflow."""

    items: list[int]
    sum_result: int
    product_result: int
    max_result: int
    min_result: int
    aggregated: dict


def example_parallel_aggregation() -> None:
    """Run multiple aggregations in parallel."""
    print("\n" + "=" * 60)
    print("Example 3: Parallel with State Aggregation")
    print("=" * 60)

    def init_items(state: AggregateState) -> dict:
        return {"items": state.get("items", [1, 2, 3, 4, 5])}

    def compute_sum(state: AggregateState) -> dict:
        return {"sum_result": sum(state["items"])}

    def compute_product(state: AggregateState) -> dict:
        result = 1
        for item in state["items"]:
            result *= item
        return {"product_result": result}

    def compute_max(state: AggregateState) -> dict:
        return {"max_result": max(state["items"])}

    def compute_min(state: AggregateState) -> dict:
        return {"min_result": min(state["items"])}

    def aggregate_results(state: AggregateState) -> dict:
        return {
            "aggregated": {
                "sum": state.get("sum_result"),
                "product": state.get("product_result"),
                "max": state.get("max_result"),
                "min": state.get("min_result"),
            }
        }

    graph = Graph(
        nodes={
            "init": init_items,
            "sum": compute_sum,
            "product": compute_product,
            "max": compute_max,
            "min": compute_min,
            "aggregate": aggregate_results,
        },
        edges=[
            # Fan-out from init
            ("init", "sum"),
            ("init", "product"),
            ("init", "max"),
            ("init", "min"),
            # Fan-in to aggregate
            ("sum", "aggregate"),
            ("product", "aggregate"),
            ("max", "aggregate"),
            ("min", "aggregate"),
        ],
        entry="init",
        state_schema=AggregateState,
    )

    result = graph.run({"items": [2, 4, 6, 8, 10]})

    print("\nItems: [2, 4, 6, 8, 10]")
    print(f"Aggregated results: {result.get('aggregated')}")


# =============================================================================
# Example 4: Parallel Pipeline Stages
# =============================================================================


class PipelineState(TypedDict, total=False):
    """State for parallel pipeline."""

    raw_items: list[str]
    validated: list[str]
    transformed: list[str]
    enriched: list[str]
    final: list[str]


async def example_parallel_pipeline() -> None:
    """Process items through parallel pipeline stages."""
    print("\n" + "=" * 60)
    print("Example 4: Parallel Pipeline Stages")
    print("=" * 60)

    async def validate_item(item: str) -> str | None:
        """Validate a single item."""
        await asyncio.sleep(0.05)  # Simulate validation
        if item.startswith("valid_"):
            return item
        return None

    async def transform_item(item: str) -> str:
        """Transform a single item."""
        await asyncio.sleep(0.05)  # Simulate transformation
        return item.upper()

    async def enrich_item(item: str) -> str:
        """Enrich a single item."""
        await asyncio.sleep(0.05)  # Simulate enrichment
        return f"{item}_enriched"

    async def validate_all(state: PipelineState) -> dict:
        """Validate all items in parallel."""
        results = await asyncio.gather(*[validate_item(item) for item in state["raw_items"]])
        valid = [r for r in results if r is not None]
        return {"validated": valid}

    async def transform_all(state: PipelineState) -> dict:
        """Transform all items in parallel."""
        results = await asyncio.gather(*[transform_item(item) for item in state["validated"]])
        return {"transformed": results}

    async def enrich_all(state: PipelineState) -> dict:
        """Enrich all items in parallel."""
        results = await asyncio.gather(*[enrich_item(item) for item in state["transformed"]])
        return {"enriched": results}

    def finalize(state: PipelineState) -> dict:
        return {"final": state["enriched"]}

    graph = Graph(
        nodes={
            "validate": validate_all,
            "transform": transform_all,
            "enrich": enrich_all,
            "finalize": finalize,
        },
        edges=[
            ("validate", "transform"),
            ("transform", "enrich"),
            ("enrich", "finalize"),
        ],
        entry="validate",
        state_schema=PipelineState,
    )

    raw_items = [
        "valid_item1",
        "invalid_item",
        "valid_item2",
        "valid_item3",
    ]

    print(f"\nRaw items: {raw_items}")

    result = await graph.arun({"raw_items": raw_items})

    print(f"Final items: {result.get('final')}")


# =============================================================================
# Example 5: Parallel with Early Exit
# =============================================================================


class SearchState(TypedDict, total=False):
    """State for parallel search."""

    query: str
    results: dict[str, str | None]
    first_found: str | None


async def example_parallel_search() -> None:
    """Search multiple sources, use first result."""
    print("\n" + "=" * 60)
    print("Example 5: Parallel with Early Exit (First Found)")
    print("=" * 60)

    async def search_source(source: str, query: str, delay: float) -> tuple[str, str | None]:
        """Search a source with simulated delay."""
        await asyncio.sleep(delay)
        # Simulate some sources finding the query
        if source == "database" and query == "test":
            return source, f"Found '{query}' in {source}"
        if source == "cache":
            return source, f"Found '{query}' in {source}"
        return source, None

    async def parallel_search(state: SearchState) -> dict:
        """Search all sources in parallel."""
        query = state["query"]

        # Search with different latencies
        tasks = [
            search_source("cache", query, 0.1),
            search_source("database", query, 0.3),
            search_source("external_api", query, 0.5),
        ]

        results = await asyncio.gather(*tasks)
        results_dict = dict(results)

        # Find first non-None result
        first_found = None
        for source, result in results:
            if result is not None:
                first_found = result
                break

        return {
            "results": results_dict,
            "first_found": first_found,
        }

    graph = Graph(
        nodes={"search": parallel_search},
        edges=[],
        entry="search",
        state_schema=SearchState,
    )

    print("\nSearching for 'test' across multiple sources...")

    result = await graph.arun({"query": "test"})

    print(f"\nAll results: {result.get('results')}")
    print(f"First found: {result.get('first_found')}")


# =============================================================================
# Example 6: Parallel Batch Processing
# =============================================================================


class BatchState(TypedDict, total=False):
    """State for batch processing."""

    items: list[int]
    batch_size: int
    batches: list[list[int]]
    batch_results: list[int]
    total: int


async def example_batch_processing() -> None:
    """Process items in parallel batches."""
    print("\n" + "=" * 60)
    print("Example 6: Parallel Batch Processing")
    print("=" * 60)

    def create_batches(state: BatchState) -> dict:
        """Split items into batches."""
        items = state["items"]
        batch_size = state.get("batch_size", 3)
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        return {"batches": batches}

    async def process_batch(batch: list[int]) -> int:
        """Process a single batch (return sum)."""
        await asyncio.sleep(0.1)  # Simulate processing
        return sum(batch)

    async def process_all_batches(state: BatchState) -> dict:
        """Process all batches in parallel."""
        results = await asyncio.gather(*[process_batch(batch) for batch in state["batches"]])
        return {"batch_results": list(results)}

    def compute_total(state: BatchState) -> dict:
        """Compute total from batch results."""
        return {"total": sum(state["batch_results"])}

    graph = Graph(
        nodes={
            "batch": create_batches,
            "process": process_all_batches,
            "total": compute_total,
        },
        edges=[
            ("batch", "process"),
            ("process", "total"),
        ],
        entry="batch",
        state_schema=BatchState,
    )

    items = list(range(1, 11))  # [1, 2, 3, ..., 10]

    print(f"\nProcessing items: {items}")
    print("Batch size: 3")

    result = await graph.arun({"items": items, "batch_size": 3})

    print(f"Batches: {result.get('batches')}")
    print(f"Batch results: {result.get('batch_results')}")
    print(f"Total: {result.get('total')}")


# =============================================================================
# Example 7: Parallel with Timeout
# =============================================================================


class TimeoutState(TypedDict, total=False):
    """State for timeout example."""

    sources: list[str]
    results: dict[str, str]
    timed_out: list[str]


async def example_parallel_with_timeout() -> None:
    """Handle timeouts in parallel operations."""
    print("\n" + "=" * 60)
    print("Example 7: Parallel with Timeout Handling")
    print("=" * 60)

    async def fetch_with_timeout(
        source: str, delay: float, timeout: float = 0.3
    ) -> tuple[str, str | None, bool]:
        """Fetch from source with timeout."""
        try:
            await asyncio.wait_for(asyncio.sleep(delay), timeout=timeout)
            return source, f"Result from {source}", False
        except TimeoutError:
            return source, None, True

    async def parallel_fetch(state: TimeoutState) -> dict:
        """Fetch from all sources with individual timeouts."""
        sources_with_delays = [
            ("fast_api", 0.1),
            ("slow_api", 0.5),
            ("medium_api", 0.2),
            ("very_slow_api", 1.0),
        ]

        tasks = [fetch_with_timeout(source, delay) for source, delay in sources_with_delays]

        raw_results = await asyncio.gather(*tasks)

        results = {}
        timed_out = []

        for source, result, did_timeout in raw_results:
            if did_timeout:
                timed_out.append(source)
            else:
                results[source] = result

        return {"results": results, "timed_out": timed_out}

    graph = Graph(
        nodes={"fetch": parallel_fetch},
        edges=[],
        entry="fetch",
        state_schema=TimeoutState,
    )

    print("\nFetching from sources with 0.3s timeout...")

    result = await graph.arun({})

    print(f"\nSuccessful: {list(result.get('results', {}).keys())}")
    print(f"Timed out: {result.get('timed_out')}")


# =============================================================================
# Example 8: Diamond Pattern (Fan-out, Process, Fan-in)
# =============================================================================


def example_diamond_pattern() -> None:
    """Classic diamond pattern: split, process, merge."""
    print("\n" + "=" * 60)
    print("Example 8: Diamond Pattern (Split/Process/Merge)")
    print("=" * 60)

    def split(state: dict) -> dict:
        """Split input for parallel processing."""
        value = state.get("value", 10)
        return {"base": value}

    def add_path(state: dict) -> dict:
        """Add 5 to base value."""
        return {"add_result": state["base"] + 5}

    def multiply_path(state: dict) -> dict:
        """Multiply base value by 2."""
        return {"multiply_result": state["base"] * 2}

    def subtract_path(state: dict) -> dict:
        """Subtract 3 from base value."""
        return {"subtract_result": state["base"] - 3}

    def merge(state: dict) -> dict:
        """Merge all path results."""
        return {
            "merged": {
                "original": state.get("base"),
                "added": state.get("add_result"),
                "multiplied": state.get("multiply_result"),
                "subtracted": state.get("subtract_result"),
            }
        }

    graph = Graph(
        nodes={
            "split": split,
            "add": add_path,
            "multiply": multiply_path,
            "subtract": subtract_path,
            "merge": merge,
        },
        edges=[
            # Fan-out
            ("split", "add"),
            ("split", "multiply"),
            ("split", "subtract"),
            # Fan-in
            ("add", "merge"),
            ("multiply", "merge"),
            ("subtract", "merge"),
        ],
        entry="split",
    )

    result = graph.run({"value": 10})

    print("\nInput value: 10")
    print(f"Diamond pattern results: {result.get('merged')}")

    # Visualize the graph
    print("\n--- Graph Structure ---")
    print(graph.get_arch_diagram())


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Parallel Execution Examples")
    print("Concurrent Node Execution Patterns")
    print("=" * 60)

    # Sync examples
    example_basic_fanout()
    example_parallel_aggregation()
    example_diamond_pattern()

    # Async examples
    await example_async_parallel()
    await example_parallel_pipeline()
    await example_parallel_search()
    await example_batch_processing()
    await example_parallel_with_timeout()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
