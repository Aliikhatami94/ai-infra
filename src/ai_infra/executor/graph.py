"""ExecutorGraph - Graph-based executor implementation.

Phase 1.2.1: ExecutorGraph class using ai_infra.graph.Graph.
Phase 1.6: Increased recursion limit to 100 (from LangGraph default of 25).
Phase 1.6.1: Tracing integration via ai_infra.tracing.
Phase 1.6.2: Streaming output via astream_events.
Phase 2.1: Shell tool integration for autonomous command execution.

This module implements the graph-based executor that replaces the imperative
loop with a structured, observable state machine.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.constants import END, START

from ai_infra.executor.adaptive import AdaptiveMode
from ai_infra.executor.edges.routes import (
    route_after_analyze_failure,
    route_after_decide,
    route_after_pick,
    route_after_repair_test,
    route_after_replan,
    route_after_validate,
    route_after_verify,
    route_after_write,
)
from ai_infra.executor.nodes import (
    analyze_failure_node,
    build_context_node,
    checkpoint_node,
    decide_next_node,
    execute_task_node,
    handle_failure_node,
    parse_roadmap_node,
    pick_task_node,
    plan_task_node,
    repair_code_node,
    repair_test_node,
    replan_task_node,
    validate_code_node,
    verify_task_node,
    write_files_node,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.graph import Graph

# Phase 2.1: Shell tool imports
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent
    from ai_infra.executor.checkpoint import Checkpointer
    from ai_infra.executor.context import ProjectContext
    from ai_infra.executor.graph_tracing import (
        TracingConfig,
    )
    from ai_infra.executor.hitl import HITLManager, InterruptConfig
    from ai_infra.executor.observability import ExecutorCallbacks
    from ai_infra.executor.project_memory import ProjectMemory
    from ai_infra.executor.run_memory import RunMemory
    from ai_infra.executor.streaming import ExecutorStreamEvent, StreamingConfig
    from ai_infra.executor.todolist import TodoListManager
    from ai_infra.executor.verifier import CheckLevel, TaskVerifier
    from ai_infra.llm.shell.session import ShellSession

logger = get_logger("executor.graph")


# =============================================================================
# Phase 1.6: Recursion Limit Configuration
# =============================================================================

# Default recursion limit for graph execution.
# LangGraph's default is 25, which is too low for retry/replan cycles.
# The validate/repair loop (max 2 repairs × 5 nodes per cycle = 10) plus
# normal task flow can exceed 25 transitions for complex tasks.
DEFAULT_RECURSION_LIMIT = 100


class ExecutorGraph:
    """Graph-based executor using ai_infra.graph.Graph.

    This class wraps the compiled graph and provides the interface for
    executing tasks from a ROADMAP.md file.

    The graph flow (Phase 2.1):
        START -> parse_roadmap -> pick_task -> [build_context -> execute_task
        -> validate_code -> (repair_code loop) -> write_files -> verify_task
        -> checkpoint -> decide_next] (loop) -> END

    Error handling flow (Phase 2.1 - Simplified):
        execute_task/verify_task -> handle_failure -> decide_next (skip/stop)
        (Rollback has been removed - pre-write validation prevents bad writes)

    Example:
        ```python
        from ai_infra.executor.graph import ExecutorGraph
        from ai_infra import Agent

        executor = ExecutorGraph(
            agent=Agent(model="claude-sonnet-4-20250514"),
            roadmap_path="ROADMAP.md",
        )

        # Run to completion
        final_state = await executor.arun()
        print(f"Completed {final_state['tasks_completed_count']} tasks")

        # Or stream node execution
        async for node, state in executor.astream():
            print(f"Executed: {node}")
        ```

    Attributes:
        graph: The compiled Graph instance.
        agent: The Agent for task execution.
        checkpointer: Git checkpointer for code changes.
        verifier: Task verification component.
        project_context: Project context builder.
        shell_session: Shell session for command execution (Phase 2.1).
    """

    def __init__(
        self,
        *,
        agent: Agent | None = None,
        roadmap_path: str = "ROADMAP.md",
        checkpointer: Checkpointer | None = None,
        verifier: TaskVerifier | None = None,
        project_context: ProjectContext | None = None,
        run_memory: RunMemory | None = None,
        project_memory: ProjectMemory | None = None,
        todo_manager: TodoListManager | None = None,
        callbacks: ExecutorCallbacks | None = None,
        check_level: CheckLevel | None = None,
        use_llm_normalization: bool = True,
        sync_roadmap: bool = True,
        max_tasks: int | None = None,
        max_retries: int = 3,
        dry_run: bool = False,
        pause_destructive: bool = True,
        enable_planning: bool = False,
        adaptive_mode: AdaptiveMode | str = AdaptiveMode.NO_ADAPT,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        graph_checkpointer: Any = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        # Phase 2.1: Shell tool configuration
        enable_shell: bool = True,
        shell_timeout: float = 120.0,
        shell_workspace: Path | str | None = None,
        shell_allowed_commands: tuple[str, ...] | None = None,
        # Phase 3.3: Autonomous verification configuration
        enable_autonomous_verify: bool = False,
        verify_timeout: float = 300.0,
    ):
        """Initialize the ExecutorGraph.

        Phase 1.4: Integration with Existing Components.
        Phase 1.6: Configurable recursion limit (default: 100).
        Phase 2.1: Shell tool integration for autonomous command execution.
        Phase 2.1.2: ProjectMemory integration for cross-run insights.
        Phase 2.2.1: Token and duration tracking via callbacks.
        Phase 2.2.2: Configurable retry count.
        Phase 2.3.1: Adaptive replanning mode.
        Phase 2.3.2: Dry run mode.
        Phase 2.3.3: Pause before destructive operations.
        Phase 2.4: Deprecate adaptive replanning in favor of targeted repair.
        Phase 2.4.2: Pre-execution planning.

        Args:
            agent: Agent instance for task execution.
            roadmap_path: Path to the ROADMAP.md file.
            checkpointer: Git checkpointer for code commits (Phase 1.4.1).
            verifier: TaskVerifier for completion checks (Phase 1.4.3).
            project_context: ProjectContext for building task context.
            run_memory: RunMemory for task-to-task context.
            project_memory: ProjectMemory for cross-run insights (Phase 2.1.2).
            todo_manager: TodoListManager for ROADMAP sync (Phase 1.4.2).
            callbacks: ExecutorCallbacks for token tracking (Phase 2.2.1).
            check_level: Verification level to use.
            use_llm_normalization: Whether to use LLM for parsing.
            sync_roadmap: Whether to sync completions to ROADMAP.md.
            max_tasks: Maximum tasks to execute (None = unlimited).
            max_retries: Maximum retry attempts per task (Phase 2.2.2, default: 3).
            dry_run: Preview actions without executing (Phase 2.3.2, default: False).
            pause_destructive: Pause on destructive operations (Phase 2.3.3, default: True).
            enable_planning: Enable pre-execution planning (Phase 2.4.2, default: False).
            adaptive_mode: DEPRECATED (Phase 2.4). Replanning mode (default: NO_ADAPT).
            recursion_limit: Max graph transitions before abort (Phase 1.6, default: 100).
            graph_checkpointer: LangGraph checkpointer for state persistence.
            interrupt_before: Nodes to pause before.
            interrupt_after: Nodes to pause after.
            enable_shell: Enable shell tool for command execution (Phase 2.1, default: True).
            shell_timeout: Default timeout for shell commands (Phase 2.1, default: 120s).
            shell_workspace: Working directory for shell commands (Phase 2.1, default: roadmap directory).
            shell_allowed_commands: Optional allowlist of permitted commands (Phase 2.4, default: None = all).
            enable_autonomous_verify: Enable autonomous verification agent (Phase 3.3, default: False).
            verify_timeout: Timeout for autonomous verification (Phase 3.3, default: 300s).
        """
        self.agent = agent
        self.roadmap_path = roadmap_path
        self.checkpointer = checkpointer
        self.verifier = verifier
        self.project_context = project_context
        self.run_memory = run_memory
        self.todo_manager = todo_manager
        self.callbacks = callbacks
        self.check_level = check_level
        self.use_llm_normalization = use_llm_normalization
        self.sync_roadmap = sync_roadmap
        self.max_tasks = max_tasks
        self.max_retries = max_retries
        self.dry_run = dry_run  # Phase 2.3.2
        self.pause_destructive = pause_destructive  # Phase 2.3.3
        self.enable_planning = enable_planning  # Phase 2.4.2
        self.recursion_limit = recursion_limit  # Phase 1.6

        # Phase 2.1: Shell tool configuration
        self.enable_shell = enable_shell
        self.shell_timeout = shell_timeout
        self.shell_workspace = (
            Path(shell_workspace) if shell_workspace else Path(roadmap_path).parent
        )
        self.shell_allowed_commands = shell_allowed_commands  # Phase 2.4: Optional allowlist
        self._shell_session: ShellSession | None = None  # Initialized in arun/astream

        # Phase 3.3: Autonomous verification configuration
        self.enable_autonomous_verify = enable_autonomous_verify
        self.verify_timeout = verify_timeout

        # Phase 2.3.1: Normalize adaptive_mode to AdaptiveMode enum
        # Phase 2.4: Deprecated in favor of targeted repair nodes
        if isinstance(adaptive_mode, str):
            self.adaptive_mode = AdaptiveMode(adaptive_mode.replace("-", "_"))
        else:
            self.adaptive_mode = adaptive_mode

        # Emit deprecation warning if adaptive_mode is used (Phase 2.4)
        if self.adaptive_mode != AdaptiveMode.NO_ADAPT:
            import warnings

            warnings.warn(
                "adaptive_mode is deprecated and will be removed in a future version. "
                "The executor now uses targeted repair nodes (repair_test_node, "
                "handle_failure_node) instead of adaptive replanning. "
                "Use repair_count and test_repair_count for retry control.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Ensure .executor directory exists for todos.json
        executor_dir = Path(roadmap_path).parent / ".executor"
        executor_dir.mkdir(parents=True, exist_ok=True)

        # Phase 2.1.2: Load or create ProjectMemory for cross-run insights
        if project_memory is not None:
            self.project_memory = project_memory
        else:
            # Auto-load from .executor/project-memory.json if exists
            from ai_infra.executor.project_memory import ProjectMemory as PM

            project_root = Path(roadmap_path).parent
            self.project_memory = PM.load(project_root)
            logger.debug(f"Loaded project memory from {project_root}")

        # Build bound nodes with dependencies injected
        bound_nodes = self._create_bound_nodes()

        # Build the graph
        self.graph = Graph(
            state_schema=ExecutorGraphState,
            nodes=bound_nodes,
            edges=self._create_edges(),
            checkpointer=graph_checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )

        logger.info("ExecutorGraph initialized")

    def _create_bound_nodes(self) -> dict[str, Any]:
        """Create node functions with dependencies bound.

        Phase 2.1.2: Passes project_memory to context and checkpoint nodes.
        Phase 2.3.1: Adds analyze_failure and replan_task nodes.
        Phase 2.4.2: Adds plan_task node when enable_planning=True.
        Phase 2.4.3: Wraps LLM-using nodes with metrics tracking.

        Returns:
            Dict mapping node names to bound callables.
        """
        from ai_infra.executor.metrics import track_node_metrics

        # Helper to wrap nodes with metrics tracking
        def wrap_with_metrics(node_name: str, node_fn: Any, uses_llm: bool = False) -> Any:
            """Wrap a node with metrics tracking if it uses LLM."""
            if uses_llm:
                return track_node_metrics(node_name)(node_fn)
            return node_fn

        nodes = {
            "parse_roadmap": partial(
                parse_roadmap_node,
                agent=self.agent,
                use_llm_normalization=self.use_llm_normalization,
                todo_manager=self.todo_manager,  # Phase 2.3.4: For initial todos.json
            ),
            "pick_task": pick_task_node,
            # Phase 2.4.3: Wrap LLM-using nodes with metrics tracking
            "build_context": wrap_with_metrics(
                "build_context",
                partial(
                    build_context_node,
                    project_context=self.project_context,
                    run_memory=self.run_memory,
                    project_memory=self.project_memory,
                ),
                uses_llm=True,  # Uses semantic search
            ),
            "execute_task": wrap_with_metrics(
                "execute_task",
                partial(
                    execute_task_node,
                    agent=self.agent,
                    dry_run=self.dry_run,  # Phase 2.3.2
                    pause_destructive=self.pause_destructive,  # Phase 2.3.3
                ),
                uses_llm=True,  # Primary LLM consumer
            ),
            "verify_task": wrap_with_metrics(
                "verify_task",
                partial(
                    verify_task_node,
                    verifier=self.verifier,
                    check_level=self.check_level,
                ),
                uses_llm=True,  # May use LLM for verification
            ),
            # Phase 1.1: Pre-write validation node
            "validate_code": validate_code_node,
            # Phase 1.2: Surgical repair node (with metrics - uses LLM)
            "repair_code": wrap_with_metrics(
                "repair_code",
                partial(
                    repair_code_node,
                    agent=self.agent,
                ),
                uses_llm=True,
            ),
            # Phase 1.3: Separated file writing node
            "write_files": partial(
                write_files_node,
                dry_run=self.dry_run,
            ),
            "checkpoint": partial(
                checkpoint_node,
                checkpointer=self.checkpointer,
                todo_manager=self.todo_manager,
                run_memory=self.run_memory,
                project_memory=self.project_memory,
                sync_roadmap=self.sync_roadmap,
            ),
            # Phase 2.1: Removed rollback node - no longer in active flow
            "handle_failure": partial(
                handle_failure_node,
                max_retries=self.max_retries,
            ),
            # Phase 2.6: Repair test node for targeted test failure repair
            "repair_test": wrap_with_metrics(
                "repair_test",
                partial(
                    repair_test_node,
                    repair_agent=self.agent,
                ),
                uses_llm=True,
            ),
            # Phase 2.3.4: Pass todo_manager to decide_next for state persistence
            "decide_next": partial(
                decide_next_node,
                todo_manager=self.todo_manager,
            ),
        }

        # Phase 2.4.2: Add planning node if enabled (with metrics tracking)
        if self.enable_planning:
            nodes["plan_task"] = wrap_with_metrics(
                "plan_task",
                partial(
                    plan_task_node,
                    planner_agent=self.agent,
                ),
                uses_llm=True,
            )

        # Phase 2.3.1: Add adaptive replanning nodes if enabled (with metrics)
        if self.adaptive_mode != AdaptiveMode.NO_ADAPT:
            nodes["analyze_failure"] = wrap_with_metrics(
                "analyze_failure",
                partial(
                    analyze_failure_node,
                    analyzer_agent=self.agent,
                ),
                uses_llm=True,
            )
            nodes["replan_task"] = wrap_with_metrics(
                "replan_task",
                partial(
                    replan_task_node,
                    planner_agent=self.agent,
                ),
                uses_llm=True,
            )

        return nodes

    def _create_edges(self) -> list:
        """Create edge definitions for the graph.

        Phase 1.4: Adds pre-write validation, repair, and write nodes.
        Phase 2.3.1: Adds adaptive replanning edges if enabled.
        Phase 2.4.2: Adds planning node edge if enable_planning=True.

        New flow (Phase 1.4):
            execute_task -> validate_code -> (repair_code loop) -> write_files -> verify_task

        Returns:
            List of Edge and ConditionalEdge objects.
        """
        from ai_infra.graph import ConditionalEdge, Edge

        # Determine target after pick_task based on planning mode
        after_pick_target = "plan_task" if self.enable_planning else "build_context"

        edges = [
            # Entry point
            Edge(start=START, end="parse_roadmap"),
            # Main flow
            Edge(start="parse_roadmap", end="pick_task"),
            # Conditional: pick_task -> plan_task/build_context OR END
            ConditionalEdge(
                start="pick_task",
                router_fn=route_after_pick,
                targets=[after_pick_target, END],
            ),
        ]

        # Phase 2.4.2: Add planning edge if enabled
        if self.enable_planning:
            edges.append(Edge(start="plan_task", end="build_context"))

        edges.extend(
            [
                Edge(start="build_context", end="execute_task"),
                # Phase 1.4: execute_task now goes to validate_code instead of verify_task
                Edge(start="execute_task", end="validate_code"),
                # Phase 1.4: Validation -> Repair loop or Write
                ConditionalEdge(
                    start="validate_code",
                    router_fn=route_after_validate,
                    targets=["write_files", "repair_code", "handle_failure"],
                ),
                # Phase 1.4: Repair loops back to validate
                Edge(start="repair_code", end="validate_code"),
                # Phase 1.4: Write files -> Verify task
                ConditionalEdge(
                    start="write_files",
                    router_fn=route_after_write,
                    targets=["verify_task", "handle_failure"],
                ),
                Edge(start="checkpoint", end="decide_next"),
                # Phase 2.1: Simplified - handle_failure always goes to decide_next
                # Rollback has been removed from the active flow
                Edge(start="handle_failure", end="decide_next"),
                # Conditional: decide_next -> pick_task OR END
                ConditionalEdge(
                    start="decide_next",
                    router_fn=route_after_decide,
                    targets=["pick_task", END],
                ),
            ]
        )

        # Phase 2.3.1: Add adaptive replanning edges
        if self.adaptive_mode != AdaptiveMode.NO_ADAPT:
            # verify_task routes to analyze_failure or checkpoint
            edges.append(
                ConditionalEdge(
                    start="verify_task",
                    router_fn=route_after_verify,
                    targets=["checkpoint", "analyze_failure"],
                )
            )
            # analyze_failure routes based on classification
            edges.append(
                ConditionalEdge(
                    start="analyze_failure",
                    router_fn=route_after_analyze_failure,
                    targets=["replan_task", "handle_failure", "decide_next"],
                )
            )
            # replan_task always goes back to build_context
            edges.append(
                ConditionalEdge(
                    start="replan_task",
                    router_fn=route_after_replan,
                    targets=["build_context"],
                )
            )
        else:
            # Phase 2.6: verify_task routes to checkpoint, repair_test, or handle_failure
            edges.append(
                ConditionalEdge(
                    start="verify_task",
                    router_fn=route_after_verify,
                    targets=["checkpoint", "repair_test", "handle_failure"],
                )
            )
            # Phase 2.6: repair_test loops back to verify_task
            edges.append(
                ConditionalEdge(
                    start="repair_test",
                    router_fn=route_after_repair_test,
                    targets=["verify_task", "handle_failure"],
                )
            )

        return edges

    def get_initial_state(self) -> ExecutorGraphState:
        """Get the initial state for graph execution.

        Phase 2.1: Adds shell tool state fields.
        Phase 2.3.1: Adds adaptive replanning state fields.
        Phase 2.3.2: Adds dry_run state field.
        Phase 2.3.3: Adds pause_destructive state fields.
        Phase 2.4.2: Adds enable_planning and task_plan fields.
        Phase 2.4.3: Adds node_metrics for per-node cost tracking.

        Returns:
            ExecutorGraphState with default values.
        """
        return ExecutorGraphState(
            roadmap_path=self.roadmap_path,
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
            tasks_tasks_completed_count=0,
            max_tasks=self.max_tasks,
            max_retries=self.max_retries,  # Phase 2.2.2
            dry_run=self.dry_run,  # Phase 2.3.2
            # Phase 2.3.3: Pause destructive fields
            pause_destructive=self.pause_destructive,
            pause_reason=None,
            detected_destructive_ops=None,
            pending_result=None,
            should_continue=True,
            interrupt_requested=False,
            run_memory={},
            # Phase 2.1 & 2.2: Shell tool fields
            enable_shell=self.enable_shell,
            shell_session_active=False,
            shell_results=[],
            shell_error=None,
            # Phase 2.3.1: Adaptive replanning fields
            adaptive_mode=self.adaptive_mode.value if self.adaptive_mode else None,
            failure_classification=None,
            failure_reason=None,
            suggested_fix=None,
            execution_plan=None,
            replan_count=0,
            # Phase 2.4.2: Pre-execution planning fields
            enable_planning=self.enable_planning,
            task_plan=None,
            # Phase 2.4.3: Per-node cost tracking
            node_metrics={},
            enable_node_metrics=True,
            # Phase 3.3: Autonomous verification fields
            enable_autonomous_verify=self.enable_autonomous_verify,
            verify_timeout=self.verify_timeout,
            autonomous_verify_result=None,
        )

    async def arun(
        self,
        initial_state: ExecutorGraphState | None = None,
        config: dict[str, Any] | None = None,
    ) -> ExecutorGraphState:
        """Run the executor graph to completion.

        Phase 1.6: Uses configurable recursion_limit (default: 100).
        Phase 2.1: Manages shell session lifecycle.
        Phase 2.2.1: Tracks duration and tokens in result.

        Args:
            initial_state: Optional initial state (defaults to get_initial_state()).
            config: Optional LangGraph config (thread_id, recursion_limit, etc.).

        Returns:
            Final ExecutorGraphState after execution, with duration_ms and tokens_used.
        """
        state = initial_state or self.get_initial_state()
        config = config or {}

        # Phase 1.6: Set recursion limit if not already specified
        if "recursion_limit" not in config:
            config["recursion_limit"] = self.recursion_limit

        logger.info(
            f"Starting executor graph run (roadmap: {state['roadmap_path']}, "
            f"recursion_limit: {config['recursion_limit']})"
        )

        # Phase 2.1: Initialize shell session if enabled
        session_token = None
        if self.enable_shell:
            await self._start_shell_session()
            session_token = set_current_session(self._shell_session)
            logger.debug(f"Shell session started (workspace: {self.shell_workspace})")

        try:
            # Phase 2.2.1: Track execution duration
            start_time = time.time()

            result = await self.graph.arun(state, config=config)

            # Phase 2.2.1: Calculate duration and get tokens from callbacks
            duration_ms = int((time.time() - start_time) * 1000)
            result["duration_ms"] = duration_ms

            # Get tokens from callbacks if available
            tokens_used = 0
            if self.callbacks is not None:
                metrics = self.callbacks.get_metrics()
                tokens_used = metrics.total_tokens
            result["tokens_used"] = tokens_used

            logger.info(
                f"Executor graph completed. Tasks: {result.get('tasks_completed_count', 0)}, "
                f"Duration: {duration_ms}ms, Tokens: {tokens_used}"
            )
            return result

        finally:
            # Phase 2.1: Clean up shell session
            if self.enable_shell and self._shell_session is not None:
                await self._close_shell_session()
                if session_token is not None:
                    set_current_session(None)
                logger.debug("Shell session closed")

    def run(
        self,
        initial_state: ExecutorGraphState | None = None,
        config: dict[str, Any] | None = None,
    ) -> ExecutorGraphState:
        """Run the executor graph synchronously.

        Phase 1.6: Uses configurable recursion_limit (default: 100).

        Args:
            initial_state: Optional initial state.
            config: Optional LangGraph config (thread_id, recursion_limit, etc.).

        Returns:
            Final ExecutorGraphState after execution.
        """
        state = initial_state or self.get_initial_state()
        config = config or {}

        # Phase 1.6: Set recursion limit if not already specified
        if "recursion_limit" not in config:
            config["recursion_limit"] = self.recursion_limit

        logger.info(
            f"Starting executor graph run (roadmap: {state['roadmap_path']}, "
            f"recursion_limit: {config['recursion_limit']})"
        )

        result = self.graph.run(state, config=config)

        logger.info(f"Executor graph completed. Tasks: {result.get('tasks_completed_count', 0)}")
        return result

    async def astream(
        self,
        initial_state: ExecutorGraphState | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Stream node execution from the graph.

        Phase 1.6: Uses configurable recursion_limit (default: 100).
        Phase 2.1: Manages shell session lifecycle.

        Yields:
            Tuples of (node_name, state) after each node execution.

        Args:
            initial_state: Optional initial state.
            config: Optional LangGraph config (thread_id, recursion_limit, etc.).
        """
        state = initial_state or self.get_initial_state()
        config = config or {}

        # Phase 1.6: Set recursion limit if not already specified
        if "recursion_limit" not in config:
            config["recursion_limit"] = self.recursion_limit

        logger.info(
            f"Starting executor graph stream (roadmap: {state['roadmap_path']}, "
            f"recursion_limit: {config['recursion_limit']})"
        )

        # Phase 2.1: Initialize shell session if enabled
        session_token = None
        if self.enable_shell:
            await self._start_shell_session()
            session_token = set_current_session(self._shell_session)
            logger.debug(f"Shell session started (workspace: {self.shell_workspace})")

        try:
            async for event in self.graph.astream(state, config=config):
                for node_name, node_state in event.items():
                    yield node_name, node_state
        finally:
            # Phase 2.1: Clean up shell session
            if self.enable_shell and self._shell_session is not None:
                await self._close_shell_session()
                if session_token is not None:
                    set_current_session(None)
                logger.debug("Shell session closed")

    # =========================================================================
    # Phase 2.1: Shell Session Lifecycle
    # =========================================================================

    async def _start_shell_session(self) -> None:
        """Start the shell session for command execution.

        Phase 2.1: Initializes a ShellSession with executor configuration.
        The session persists across all task executions in this run.
        """
        from ai_infra.llm.shell.session import SessionConfig, ShellSession
        from ai_infra.llm.shell.types import ShellConfig

        shell_config = ShellConfig(timeout=self.shell_timeout)
        session_config = SessionConfig(
            workspace_root=self.shell_workspace,
            shell_config=shell_config,
        )
        self._shell_session = ShellSession(session_config)
        await self._shell_session.start()

        # Phase 2.1.1: Add shell tool to agent if agent exists
        if self.agent is not None and self.enable_shell:
            shell_tool = create_shell_tool(
                default_timeout=self.shell_timeout,
                allowed_commands=self.shell_allowed_commands,  # Phase 2.4
            )
            # Add shell tool to agent's tools if not already present
            # Check for both "run_shell" (default) and "configured_run_shell" (factory)
            tool_names = [getattr(t, "name", None) for t in self.agent.tools]
            if "run_shell" not in tool_names and "configured_run_shell" not in tool_names:
                self.agent.tools.append(shell_tool)
                logger.debug("Added shell tool to executor agent")

    async def _close_shell_session(self) -> None:
        """Close the shell session and clean up resources.

        Phase 2.1: Gracefully closes the shell session after execution.
        """
        if self._shell_session is not None:
            await self._shell_session.close()
            self._shell_session = None

    # =========================================================================
    # Phase 1.6.2: Streaming Output
    # =========================================================================

    async def astream_events(
        self,
        initial_state: ExecutorGraphState | None = None,
        config: dict[str, Any] | None = None,
        streaming_config: StreamingConfig | None = None,
    ) -> AsyncIterator[ExecutorStreamEvent]:
        """Stream execution events from the graph.

        Phase 1.6.2: Implements streaming output for real-time progress.

        Yields ExecutorStreamEvent instances for each significant
        event during execution (node transitions, task progress, etc.).

        Args:
            initial_state: Optional initial state.
            config: Optional LangGraph config.
            streaming_config: Optional streaming configuration.

        Yields:
            ExecutorStreamEvent instances.

        Example:
            ```python
            from ai_infra.executor.streaming import stream_to_console

            async for event in executor.astream_events():
                stream_to_console(event)
            ```
        """
        from ai_infra.executor.streaming import (
            StreamingConfig,
            create_node_end_event,
            create_node_start_event,
            create_run_end_event,
            create_run_start_event,
            create_task_complete_event,
            create_task_failed_event,
            create_task_start_event,
        )

        if streaming_config is None:
            streaming_config = StreamingConfig()

        state = initial_state or self.get_initial_state()
        config = config or {}

        # Emit run start
        total_tasks = len(state.get("todos", []))
        yield create_run_start_event(
            roadmap_path=state.get("roadmap_path", "ROADMAP.md"),
            total_tasks=total_tasks,
        )

        run_start = time.time()
        completed = 0
        failed = 0
        current_task = None
        task_start_time = run_start

        # Stream from underlying graph
        async for event in self.graph.astream(state, config=config):
            for node_name, node_state in event.items():
                node_start = time.time()

                # Emit node start
                if streaming_config.show_node_transitions:
                    yield create_node_start_event(
                        node_name=node_name,
                        state=node_state,
                        include_state=streaming_config.include_state_snapshot,
                    )

                # Track task progress
                if node_name == "pick_task":
                    current_task = node_state.get("current_task")
                    task_start_time = time.time()
                    if current_task and streaming_config.show_task_progress:
                        yield create_task_start_event(
                            task=current_task,
                            task_number=node_state.get("tasks_completed_count", 0) + 1,
                            total_tasks=len(node_state.get("todos", [])),
                        )

                elif node_name == "checkpoint":
                    if current_task and streaming_config.show_task_progress:
                        task_duration = (time.time() - task_start_time) * 1000
                        yield create_task_complete_event(
                            task=current_task,
                            duration_ms=task_duration,
                            files_modified=len(node_state.get("files_modified", [])),
                        )
                        completed += 1
                        current_task = None

                elif node_name == "handle_failure":
                    if streaming_config.show_task_progress:
                        error = node_state.get("error", {})
                        error_msg = (
                            error.get("message", "Unknown error") if error else "Unknown error"
                        )
                        task_to_report = current_task or {"title": "Task"}
                        yield create_task_failed_event(
                            task=task_to_report,
                            error=error_msg,
                            duration_ms=(time.time() - task_start_time) * 1000,
                        )
                        failed += 1

                # Emit node end
                if streaming_config.show_node_transitions:
                    node_duration = (time.time() - node_start) * 1000
                    yield create_node_end_event(
                        node_name=node_name,
                        duration_ms=node_duration,
                        state=node_state,
                        include_state=streaming_config.include_state_snapshot,
                    )

        # Emit run end
        run_duration = (time.time() - run_start) * 1000
        yield create_run_end_event(
            completed=completed,
            failed=failed,
            skipped=0,
            duration_ms=run_duration,
        )

    # =========================================================================
    # Phase 1.6.1: Tracing Integration
    # =========================================================================

    async def arun_with_tracing(
        self,
        initial_state: ExecutorGraphState | None = None,
        config: dict[str, Any] | None = None,
        tracing_config: TracingConfig | None = None,
        run_id: str | None = None,
    ) -> ExecutorGraphState:
        """Run the executor graph with tracing enabled.

        Phase 1.6.1: Runs execution with full tracing via ai_infra.tracing.

        Each node execution is traced, and state transitions are visible
        in the tracing UI (LangSmith, OpenTelemetry, etc.).

        Args:
            initial_state: Optional initial state.
            config: Optional LangGraph config.
            tracing_config: Tracing configuration.
            run_id: Optional run ID (auto-generated if not provided).

        Returns:
            Final ExecutorGraphState after execution.

        Example:
            ```python
            from ai_infra.executor.graph_tracing import TracingConfig

            result = await executor.arun_with_tracing(
                tracing_config=TracingConfig.production(),
            )
            ```
        """
        from ai_infra.executor.graph_tracing import (
            TracingConfig,
            create_tracing_callbacks,
        )

        if tracing_config is None:
            tracing_config = TracingConfig()

        state = initial_state or self.get_initial_state()
        config = config or {}

        # Generate run ID if not provided
        actual_run_id = run_id or f"executor-{uuid.uuid4().hex[:8]}"

        # Create tracing callbacks
        callbacks = create_tracing_callbacks(
            config=tracing_config,
            run_id=actual_run_id,
            roadmap_path=state.get("roadmap_path", "ROADMAP.md"),
        )

        logger.info(
            f"Starting traced executor run {actual_run_id} (roadmap: {state['roadmap_path']})"
        )

        try:
            result = await self.graph.arun(state, config=config)

            callbacks.end_run(
                completed=result.get("tasks_completed_count", 0),
                failed=len(result.get("failed_todos", [])),
            )

            logger.info(
                f"Traced executor run {actual_run_id} completed. "
                f"Tasks: {result.get('tasks_completed_count', 0)}"
            )
            return result

        except Exception as e:
            callbacks.end_run(error=e)
            raise

    def get_mermaid(self) -> str:
        """Get Mermaid diagram of the graph structure.

        Returns:
            Mermaid diagram string.
        """
        return self.graph.get_arch_diagram()

    def request_interrupt(self) -> None:
        """Request graceful interrupt of execution.

        This sets the interrupt flag which will be checked at the next
        decision point (decide_next node).
        """
        logger.info("Interrupt requested")
        # Note: Actual interrupt implementation depends on graph checkpointer
        # and state update mechanism. This is a placeholder for the pattern.

    # =========================================================================
    # Phase 1.5.2: Resume Logic
    # =========================================================================

    async def aresume(
        self,
        hitl_manager: HITLManager,
        decision: str | None = None,
        notes: str | None = None,
    ) -> ExecutorGraphState:
        """Resume execution from an interrupted state.

        Phase 1.5.2: Implements resume logic for HITL.

        Args:
            hitl_manager: HITLManager instance with pending interrupt.
            decision: Decision to apply ('approve', 'reject', 'abort').
                     If None, uses existing decision from hitl_manager.
            notes: Optional notes about the decision.

        Returns:
            Final ExecutorGraphState after resumed execution.

        Raises:
            ValueError: If no pending interrupt or invalid decision.

        Example:
            ```python
            manager = HITLManager(project_root)
            if manager.has_pending_interrupt():
                result = await executor.aresume(
                    hitl_manager=manager,
                    decision="approve",
                    notes="Reviewed and approved"
                )
            ```
        """

        # Check for pending interrupt
        if not hitl_manager.has_pending_interrupt():
            # Check if there's a decision already applied
            state = hitl_manager.get_pending_state()
            if state is None:
                raise ValueError("No pending interrupt to resume from")

        # Apply decision if provided
        if decision is not None:
            hitl_manager.apply_decision(decision, notes=notes)

        # Check if we should continue
        if not hitl_manager.should_continue_after_decision():
            logger.info("Execution aborted by human decision")
            hitl_manager.clear_interrupt_state()
            # Return empty state indicating abort
            return self.get_initial_state()

        # Get resume config (thread_id for continuation)
        config = hitl_manager.get_resume_config()

        # Handle different decision types
        action = hitl_manager.get_decision_action()
        logger.info(f"Resuming execution with action: {action}")

        # Clear interrupt state before resuming
        hitl_manager.clear_interrupt_state()

        # Resume execution
        # Note: For 'skip' action, we could modify state to skip current task
        # For now, we just continue and let the graph handle it
        result = await self.graph.arun(None, config=config)

        logger.info(f"Resumed execution completed. Tasks: {result.get('tasks_completed_count', 0)}")
        return result

    def resume(
        self,
        hitl_manager: HITLManager,
        decision: str | None = None,
        notes: str | None = None,
    ) -> ExecutorGraphState:
        """Resume execution synchronously.

        Phase 1.5.2: Synchronous version of aresume.

        Args:
            hitl_manager: HITLManager instance with pending interrupt.
            decision: Decision to apply.
            notes: Optional notes.

        Returns:
            Final ExecutorGraphState.
        """
        import asyncio

        return asyncio.run(self.aresume(hitl_manager, decision, notes))

    def has_pending_interrupt(self, hitl_manager: HITLManager) -> bool:
        """Check if there's a pending interrupt.

        Args:
            hitl_manager: HITLManager instance.

        Returns:
            True if waiting for human decision.
        """
        return hitl_manager.has_pending_interrupt()


# =============================================================================
# Factory function for simpler instantiation
# =============================================================================


def create_executor_graph(
    agent: Agent,
    roadmap_path: str = "ROADMAP.md",
    **kwargs: Any,
) -> ExecutorGraph:
    """Create an ExecutorGraph with common defaults.

    Args:
        agent: Agent instance for task execution.
        roadmap_path: Path to ROADMAP.md file.
        **kwargs: Additional arguments passed to ExecutorGraph.

    Returns:
        Configured ExecutorGraph instance.

    Example:
        ```python
        from ai_infra import Agent
        from ai_infra.executor.graph import create_executor_graph

        agent = Agent(model="claude-sonnet-4-20250514")
        executor = create_executor_graph(agent, "ROADMAP.md")
        result = await executor.arun()
        ```
    """
    return ExecutorGraph(
        agent=agent,
        roadmap_path=roadmap_path,
        **kwargs,
    )


def create_executor_with_hitl(
    agent: Agent,
    roadmap_path: str = "ROADMAP.md",
    hitl_config: InterruptConfig | None = None,
    **kwargs: Any,
) -> ExecutorGraph:
    """Create an ExecutorGraph with Human-in-the-Loop configuration.

    Phase 1.5.1: Factory function with HITL config.

    Args:
        agent: Agent instance for task execution.
        roadmap_path: Path to ROADMAP.md file.
        hitl_config: InterruptConfig for HITL points.
        **kwargs: Additional arguments passed to ExecutorGraph.

    Returns:
        ExecutorGraph configured with interrupt points.

    Example:
        ```python
        from ai_infra import Agent
        from ai_infra.executor.graph import create_executor_with_hitl
        from ai_infra.executor.hitl import InterruptConfig

        agent = Agent(model="claude-sonnet-4-20250514")

        # Require approval before each task
        executor = create_executor_with_hitl(
            agent,
            "ROADMAP.md",
            hitl_config=InterruptConfig.approval_mode(),
        )

        result = await executor.arun()
        ```
    """
    from ai_infra.executor.hitl import InterruptConfig, get_interrupt_lists

    # Use no_interrupt as default
    config = hitl_config or InterruptConfig.no_interrupt()
    interrupt_before, interrupt_after = get_interrupt_lists(config)

    return ExecutorGraph(
        agent=agent,
        roadmap_path=roadmap_path,
        interrupt_before=interrupt_before or None,
        interrupt_after=interrupt_after or None,
        **kwargs,
    )


__all__ = [
    "ExecutorGraph",
    "create_executor_graph",
    "create_executor_with_hitl",
]
