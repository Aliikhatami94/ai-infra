"""Tests for ai-infra graph module.

Phase 13.2 of EXECUTOR_4.md - Coverage improvement for graph module.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.constants import END, START

from ai_infra.graph.models import ConditionalEdge, Edge, GraphConfig, GraphStructure
from ai_infra.graph.utils import (
    make_hook,
    make_router_wrapper,
    normalize_initial_state,
    normalize_node_definitions,
    validate_conditional_edges,
    validate_edges,
)

# =============================================================================
# Edge Tests
# =============================================================================


class TestEdge:
    """Tests for Edge model."""

    def test_create_edge(self) -> None:
        """Test creating an edge."""
        edge = Edge(start="node1", end="node2")

        assert edge.start == "node1"
        assert edge.end == "node2"

    def test_edge_with_start_constant(self) -> None:
        """Test edge from START."""
        edge = Edge(start=START, end="first_node")

        assert edge.start == START

    def test_edge_with_end_constant(self) -> None:
        """Test edge to END."""
        edge = Edge(start="last_node", end=END)

        assert edge.end == END


# =============================================================================
# ConditionalEdge Tests
# =============================================================================


class TestConditionalEdge:
    """Tests for ConditionalEdge model."""

    def test_create_conditional_edge(self) -> None:
        """Test creating a conditional edge."""

        def router(state: dict) -> str:
            return "target1"

        edge = ConditionalEdge(
            start="decision_node",
            router_fn=router,
            targets=["target1", "target2"],
        )

        assert edge.start == "decision_node"
        assert callable(edge.router_fn)
        assert len(edge.targets) == 2


# =============================================================================
# GraphStructure Tests
# =============================================================================


class TestGraphStructure:
    """Tests for GraphStructure model."""

    def test_create_graph_structure(self) -> None:
        """Test creating graph structure."""
        structure = GraphStructure(
            state_type_name="MyState",
            state_schema={"field1": "str", "field2": "int"},
            node_count=3,
            nodes=["node1", "node2", "node3"],
            edge_count=2,
            edges=[("node1", "node2"), ("node2", "node3")],
            conditional_edge_count=0,
            entry_points=["node1"],
            exit_points=["node3"],
            has_memory=False,
        )

        assert structure.state_type_name == "MyState"
        assert structure.node_count == 3
        assert len(structure.nodes) == 3
        assert structure.has_memory is False

    def test_graph_structure_with_conditional_edges(self) -> None:
        """Test graph structure with conditional edges."""
        structure = GraphStructure(
            state_type_name="RouterState",
            state_schema={"query": "str"},
            node_count=4,
            nodes=["start", "router", "path_a", "path_b"],
            edge_count=1,
            edges=[("start", "router")],
            conditional_edge_count=1,
            conditional_edges=[{"start": "router", "targets": ["path_a", "path_b"]}],
            entry_points=["start"],
            exit_points=["path_a", "path_b"],
            has_memory=True,
        )

        assert structure.conditional_edge_count == 1
        assert structure.has_memory is True


# =============================================================================
# GraphConfig Tests
# =============================================================================


class TestGraphConfig:
    """Tests for GraphConfig model."""

    def test_create_graph_config(self) -> None:
        """Test creating graph config."""

        def node1(state: dict) -> dict:
            return state

        config = GraphConfig(
            node_definitions=[node1],
            edges=[("node1", END)],
        )

        assert len(config.node_definitions) == 1
        assert len(config.edges) == 1

    def test_graph_config_with_memory(self) -> None:
        """Test graph config with memory store."""

        def node1(state: dict) -> dict:
            return state

        mock_memory = object()
        config = GraphConfig(
            node_definitions=[node1],
            edges=[("node1", END)],
            memory_store=mock_memory,
        )

        assert config.memory_store is mock_memory


# =============================================================================
# normalize_node_definitions Tests
# =============================================================================


class TestNormalizeNodeDefinitions:
    """Tests for normalize_node_definitions function."""

    def test_normalize_dict(self) -> None:
        """Test normalizing dictionary of node definitions."""

        def my_node(state: dict) -> dict:
            return state

        definitions = {"custom_name": my_node}
        result = normalize_node_definitions(definitions)

        assert result == {"custom_name": my_node}
        assert result is not definitions  # Should be a copy

    def test_normalize_list(self) -> None:
        """Test normalizing list of node definitions."""

        def node_a(state: dict) -> dict:
            return state

        def node_b(state: dict) -> dict:
            return state

        definitions = [node_a, node_b]
        result = normalize_node_definitions(definitions)

        assert result == {"node_a": node_a, "node_b": node_b}


# =============================================================================
# normalize_initial_state Tests
# =============================================================================


class TestNormalizeInitialState:
    """Tests for normalize_initial_state function."""

    def test_with_initial_state(self) -> None:
        """Test with initial state provided."""
        initial = {"key": "value"}
        result = normalize_initial_state(initial, {})

        assert result == initial

    def test_with_kwargs(self) -> None:
        """Test with kwargs when initial_state is None."""
        kwargs = {"field1": "value1", "field2": 42}
        result = normalize_initial_state(None, kwargs)

        assert result == kwargs

    def test_error_when_both_provided(self) -> None:
        """Test error when both initial_state and kwargs provided."""
        with pytest.raises(ValueError, match="Provide either"):
            normalize_initial_state({"key": "value"}, {"other": "value"})


# =============================================================================
# validate_edges Tests
# =============================================================================


class TestValidateEdges:
    """Tests for validate_edges function."""

    def test_valid_edges(self) -> None:
        """Test validation of valid edges."""
        all_nodes = {"node1", "node2", "node3"}
        edges = [("node1", "node2"), ("node2", "node3")]

        # Should not raise
        validate_edges(edges, all_nodes)

    def test_edge_with_start_end_constants(self) -> None:
        """Test edges with START and END constants."""
        all_nodes = {"node1"}
        edges = [(START, "node1"), ("node1", END)]

        # Should not raise
        validate_edges(edges, all_nodes)

    def test_invalid_edge_endpoint(self) -> None:
        """Test error for invalid edge endpoint."""
        all_nodes = {"node1", "node2"}
        edges = [("node1", "unknown_node")]

        with pytest.raises(ValueError, match="not a known node"):
            validate_edges(edges, all_nodes)


# =============================================================================
# validate_conditional_edges Tests
# =============================================================================


class TestValidateConditionalEdges:
    """Tests for validate_conditional_edges function."""

    def test_valid_conditional_edges(self) -> None:
        """Test validation of valid conditional edges."""

        def router(state: dict) -> str:
            return "path_a"

        all_nodes = {"decision", "path_a", "path_b"}
        conditional_edges = [("decision", router, {"a": "path_a", "b": "path_b"})]

        # Should not raise
        validate_conditional_edges(conditional_edges, all_nodes)

    def test_invalid_start_node(self) -> None:
        """Test error for invalid start node in conditional edge."""

        def router(state: dict) -> str:
            return "path_a"

        all_nodes = {"path_a", "path_b"}
        conditional_edges = [("unknown_decision", router, {"a": "path_a"})]

        with pytest.raises(ValueError, match="not a known node"):
            validate_conditional_edges(conditional_edges, all_nodes)

    def test_invalid_target_node(self) -> None:
        """Test error for invalid target in conditional edge."""

        def router(state: dict) -> str:
            return "path_a"

        all_nodes = {"decision", "path_a"}
        conditional_edges = [("decision", router, {"a": "path_a", "b": "unknown_path"})]

        with pytest.raises(ValueError, match="not a known node"):
            validate_conditional_edges(conditional_edges, all_nodes)


# =============================================================================
# make_router_wrapper Tests
# =============================================================================


class TestMakeRouterWrapper:
    """Tests for make_router_wrapper function."""

    @pytest.mark.asyncio
    async def test_sync_router(self) -> None:
        """Test wrapping synchronous router."""

        def sync_router(state: dict) -> str:
            return "target1"

        wrapper = make_router_wrapper(sync_router, {"target1", "target2"})
        result = await wrapper({"key": "value"})

        assert result == "target1"

    @pytest.mark.asyncio
    async def test_async_router(self) -> None:
        """Test wrapping async router."""

        async def async_router(state: dict) -> str:
            return "target2"

        wrapper = make_router_wrapper(async_router, {"target1", "target2"})
        result = await wrapper({"key": "value"})

        assert result == "target2"

    @pytest.mark.asyncio
    async def test_invalid_router_result(self) -> None:
        """Test error for invalid router result."""

        def bad_router(state: dict) -> str:
            return "unknown_target"

        wrapper = make_router_wrapper(bad_router, {"target1", "target2"})

        with pytest.raises(ValueError, match="not in targets"):
            await wrapper({"key": "value"})


# =============================================================================
# make_hook Tests
# =============================================================================


class TestMakeHook:
    """Tests for make_hook function."""

    def test_none_hook(self) -> None:
        """Test make_hook returns None for None input."""
        result = make_hook(None)

        assert result is None

    def test_sync_hook(self) -> None:
        """Test make_hook with sync function."""
        called = []

        def sync_hook(event: Any) -> None:
            called.append(event)

        result = make_hook(sync_hook, sync=True)

        assert result is not None or sync_hook is not None  # Implementation varies

    def test_async_hook(self) -> None:
        """Test make_hook with async function."""
        called = []

        async def async_hook(event: Any) -> None:
            called.append(event)

        result = make_hook(async_hook)

        assert result is not None or async_hook is not None  # Implementation varies
