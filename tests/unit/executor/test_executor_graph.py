"""Tests for ExecutorGraph class.

Phase 1.2.5: Tests for the complete graph implementation.
"""

from __future__ import annotations

from ai_infra.executor.state import ExecutorGraphState

# =============================================================================
# Test state initialization without full graph (to avoid forward reference issue)
# =============================================================================


class TestExecutorGraphStateCreation:
    """Tests for ExecutorGraphState creation."""

    def test_creates_minimal_state(self):
        """Test minimal state creation."""
        state = ExecutorGraphState(
            roadmap_path="ROADMAP.md",
            todos=[],
            current_task=None,
            context="",
            prompt="",
            agent_result=None,
            files_modified=[],
            verified=False,
            last_checkpoint_sha=None,
            error=None,
            retry_count=0,
            completed_count=0,
            max_tasks=None,
            should_continue=True,
            interrupt_requested=False,
            run_memory={},
        )

        assert state["roadmap_path"] == "ROADMAP.md"
        assert state["todos"] == []
        assert state["should_continue"] is True


class TestExecutorGraphImports:
    """Tests for ExecutorGraph imports and initialization."""

    def test_graph_module_imports(self):
        """Test that the graph module imports correctly."""
        from ai_infra.executor.graph import ExecutorGraph, create_executor_graph

        assert ExecutorGraph is not None
        assert create_executor_graph is not None

    def test_nodes_import(self):
        """Test that all nodes import correctly."""
        from ai_infra.executor.nodes import (
            build_context_node,
            checkpoint_node,
            decide_next_node,
            execute_task_node,
            handle_failure_node,
            parse_roadmap_node,
            pick_task_node,
            rollback_node,
            verify_task_node,
        )

        assert all(
            callable(f)
            for f in [
                build_context_node,
                checkpoint_node,
                decide_next_node,
                execute_task_node,
                handle_failure_node,
                parse_roadmap_node,
                pick_task_node,
                rollback_node,
                verify_task_node,
            ]
        )

    def test_edge_routes_import(self):
        """Test that edge routes import correctly."""
        from ai_infra.executor.edges.routes import (
            route_after_decide,
            route_after_execute,
            route_after_failure,
            route_after_pick,
            route_after_rollback,
            route_after_verify,
        )

        assert all(
            callable(f)
            for f in [
                route_after_decide,
                route_after_execute,
                route_after_failure,
                route_after_pick,
                route_after_rollback,
                route_after_verify,
            ]
        )


class TestExecutorGraphBindings:
    """Tests for ExecutorGraph node bindings."""

    def test_bound_nodes_created(self):
        """Test that bound nodes are created without agent."""
        from ai_infra.executor.graph import ExecutorGraph

        # Create executor without building the full graph
        executor = object.__new__(ExecutorGraph)
        executor.agent = None
        executor.checkpointer = None
        executor.verifier = None
        executor.project_context = None
        executor.run_memory = None
        executor.project_memory = None  # Phase 1.4
        executor.todo_manager = None  # Phase 1.4.2
        executor.check_level = None
        executor.use_llm_normalization = False
        executor.sync_roadmap = True  # Phase 1.4.2
        executor.dry_run = False  # Phase 1.4
        executor.pause_destructive = False  # Phase 1.4
        executor.max_retries = 3  # Phase 1.4
        executor.enable_planning = False  # Phase 1.4
        executor.adaptive_mode = None  # Phase 1.4
        executor._model_router = None  # Phase 1.1
        executor._routing_metrics = None  # Phase 1.1
        # Phase 7.2: Subagent routing
        executor.use_subagents = False
        executor.subagent_config = None  # Phase 7.4
        # Phase 8.1: Skills system
        executor.skills_db = None
        executor.enable_learning = False
        # Phase 15.3: MCP integration
        executor.mcp_servers = []
        executor.mcp_discover_timeout = 30.0
        executor.mcp_tool_timeout = 60.0
        executor.mcp_auto_discover = True
        executor._mcp_manager = None
        # Phase 16.5.11.3: Orchestrator routing
        executor.use_orchestrator = True
        executor.orchestrator_model = "gpt-4o-mini"
        executor.orchestrator_confidence_threshold = 0.7

        bound_nodes = executor._create_bound_nodes()

        assert "parse_roadmap" in bound_nodes
        assert "pick_task" in bound_nodes
        assert "build_context" in bound_nodes
        assert "execute_task" in bound_nodes
        assert "verify_task" in bound_nodes
        assert "checkpoint" in bound_nodes
        # Phase 2.1: rollback node has been removed from active flow
        assert "rollback" not in bound_nodes
        assert "handle_failure" in bound_nodes
        assert "decide_next" in bound_nodes
        # Phase 1.4: New nodes
        assert "validate_code" in bound_nodes
        assert "repair_code" in bound_nodes
        assert "write_files" in bound_nodes

    def test_edges_created(self):
        """Test that edges are created correctly."""
        from ai_infra.executor.graph import ExecutorGraph

        # Create executor without building the full graph
        executor = object.__new__(ExecutorGraph)
        executor.enable_planning = False  # Phase 1.4
        executor.adaptive_mode = None  # Phase 1.4

        edges = executor._create_edges()

        # Should have all edges (regular + conditional)
        # Phase 1.4: Added validate_code, repair_code, write_files edges
        assert len(edges) >= 10  # At least 10 edges


class TestFactoryFunction:
    """Tests for create_executor_graph factory."""

    def test_factory_signature(self):
        """Test factory function has correct signature."""
        import inspect

        from ai_infra.executor.graph import create_executor_graph

        sig = inspect.signature(create_executor_graph)
        params = list(sig.parameters.keys())

        assert "agent" in params
        assert "roadmap_path" in params


# =============================================================================
# Phase 1.6: Recursion Limit Tests
# =============================================================================


class TestRecursionLimit:
    """Tests for configurable recursion limit (Phase 1.6)."""

    def test_default_recursion_limit_is_100(self) -> None:
        """Test: DEFAULT_RECURSION_LIMIT constant is 100."""
        from ai_infra.executor.graph import DEFAULT_RECURSION_LIMIT

        assert DEFAULT_RECURSION_LIMIT == 100

    def test_executor_uses_default_limit(self, tmp_path) -> None:
        """Test: ExecutorGraph uses DEFAULT_RECURSION_LIMIT by default."""
        from ai_infra.executor.graph import DEFAULT_RECURSION_LIMIT, ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Tasks\n- [ ] Task 1")

        executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
        )

        assert executor.recursion_limit == DEFAULT_RECURSION_LIMIT
        assert executor.recursion_limit == 100

    def test_custom_recursion_limit(self, tmp_path) -> None:
        """Test: ExecutorGraph accepts custom recursion_limit."""
        from ai_infra.executor.graph import ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Tasks\n- [ ] Task 1")

        executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
            recursion_limit=50,
        )

        assert executor.recursion_limit == 50

    def test_higher_recursion_limit(self, tmp_path) -> None:
        """Test: ExecutorGraph accepts higher recursion limit."""
        from ai_infra.executor.graph import ExecutorGraph

        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Tasks\n- [ ] Task 1")

        executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
            recursion_limit=500,
        )

        assert executor.recursion_limit == 500
