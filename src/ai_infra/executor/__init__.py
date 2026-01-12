"""Executor: Autonomous task execution from ROADMAP.md files.

The Executor module provides infrastructure for:
- Parsing tasks from ROADMAP.md files
- Building rich context for task execution (ProjectContext)
- Running tasks autonomously with Agent(deep=True)
- Verifying task completion
- Analyzing failures and identifying patterns
- Managing execution state and checkpoints
- Git checkpointing for task completion

Example:
    >>> from ai_infra.executor import Executor, ExecutorConfig
    >>>
    >>> executor = Executor(
    ...     roadmap="./ROADMAP.md",
    ...     config=ExecutorConfig(max_tasks=5),
    ... )
    >>> await executor.run()
"""

from ai_infra.executor.adaptive import (
    AdaptiveMode,
    PlanAnalyzer,
    PlanSuggestion,
    SuggestionResult,
    SuggestionSafety,
    SuggestionType,
    analyze_failure_for_plan_fix,
)
from ai_infra.executor.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointResult,
    CommitInfo,
    GitNotFoundError,
    GitOperationError,
    NotAGitRepoError,
    RollbackResult,
)
from ai_infra.executor.checkpointer import ExecutorCheckpointer
from ai_infra.executor.context import (
    ContextBudget,
    ContextCache,
    ContextResult,
    ProjectContext,
)
from ai_infra.executor.dependencies import (
    ChangeAnalysis,
    ChangeDetector,
    DependencyTracker,
    DependencyType,
    DependencyWarning,
    FileDependency,
    ImpactLevel,
    ImportInfo,
    ImportParser,
    ParallelGroup,
    TaskDependencyGraph,
    TaskNode,
)
from ai_infra.executor.failure import (
    FailureAnalyzer,
    FailureCategory,
    FailureRecord,
    FailureSeverity,
    FailureStats,
)
from ai_infra.executor.graph import (
    ExecutorGraph,
    create_executor_graph,
    create_executor_with_hitl,
)
from ai_infra.executor.graph_tracing import (
    ExecutorTracingCallbacks,
    TracingConfig,
    create_traced_nodes,
    create_tracing_callbacks,
    traced_node,
)
from ai_infra.executor.hitl import (
    HITLDecision,
    HITLManager,
    HITLState,
    InterruptConfig,
    InterruptPoint,
    get_interrupt_lists,
)
from ai_infra.executor.learning import (
    FailurePattern,
    LearningStats,
    LearningStore,
    PatternType,
    PromptRefiner,
    SuccessPattern,
    TaskType,
)
from ai_infra.executor.loop import (
    ExecutionResult,
    ExecutionStatus,
    Executor,
    ExecutorConfig,
    ReviewInfo,
    RunStatus,
    RunSummary,
    VerifyMode,
)
from ai_infra.executor.models import Task, TaskStatus
from ai_infra.executor.observability import (
    ExecutorCallbacks,
    ExecutorMetrics,
    TaskMetrics,
    get_executor_tracer,
    log_recovery_action,
    log_task_context,
    log_verification_result,
)
from ai_infra.executor.outcome_extractor import (
    ExtractionResult,
    extract_outcome,
    extract_outcome_sync,
)
from ai_infra.executor.parser import ParseError, ParserConfig, RoadmapParser
from ai_infra.executor.project_memory import (
    FileInfo,
    ProjectMemory,
)
from ai_infra.executor.recovery import (
    CheckpointMetadata,
    FileSnapshot,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    RollbackPreview,
    SelectiveRollbackResult,
)
from ai_infra.executor.roadmap import (
    ParsedTask,
    Phase,
    Priority,
    Roadmap,
    Section,
    Subtask,
)
from ai_infra.executor.run_memory import (
    FileAction,
    RunMemory,
    TaskOutcome,
)
from ai_infra.executor.state import ExecutorState, StateSummary, TaskState
from ai_infra.executor.state_migration import (
    MAX_RUN_MEMORY_ENTRIES,
    MAX_STATE_SIZE_BYTES,
    CheckpointTrigger,
    GraphStatePersistence,
    MemoryIntegration,
    StatePruning,
)
from ai_infra.executor.streaming import (
    ExecutorStreamEvent,
    JsonFormatter,
    MinimalFormatter,
    OutputFormat,
    PlainFormatter,
    StreamEventType,
    StreamingConfig,
    create_interrupt_event,
    create_node_end_event,
    create_node_error_event,
    create_node_start_event,
    create_resume_event,
    create_run_end_event,
    create_run_start_event,
    create_task_complete_event,
    create_task_failed_event,
    create_task_start_event,
    get_formatter,
    stream_to_console,
)
from ai_infra.executor.testing import (
    ChaosAgent,
    MockAgent,
    MockResponse,
    TestProject,
)
from ai_infra.executor.todolist import (
    GroupStrategy,
    TodoItem,
    TodoListManager,
    TodoStatus,
)
from ai_infra.executor.verifier import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    TaskVerifier,
    VerificationResult,
)
from ai_infra.executor.workspace import (
    CrossProjectDependency,
    DependencyScope,
    ProjectInfo,
    ProjectType,
    Workspace,
    WorkspaceCheckpointResult,
    WorkspaceRollbackResult,
)

__all__ = [
    # Adaptive Planning (Phase 5.5)
    "AdaptiveMode",
    "PlanAnalyzer",
    "PlanSuggestion",
    "SuggestionResult",
    "SuggestionSafety",
    "SuggestionType",
    "analyze_failure_for_plan_fix",
    # Checkpoint
    "CheckpointError",
    "CheckpointResult",
    "Checkpointer",
    "CommitInfo",
    "GitNotFoundError",
    "GitOperationError",
    "NotAGitRepoError",
    "RollbackResult",
    # Context (Phase 5.1)
    "ContextBudget",
    "ContextCache",
    "ContextResult",
    "ProjectContext",
    # Dependencies
    "ChangeAnalysis",
    "ChangeDetector",
    "DependencyTracker",
    "DependencyType",
    "DependencyWarning",
    "FileDependency",
    "ImpactLevel",
    "ImportInfo",
    "ImportParser",
    # Parallel Execution (Phase 5.1)
    "ParallelGroup",
    "TaskDependencyGraph",
    "TaskNode",
    # Recovery (Phase 4.2)
    "CheckpointMetadata",
    "FileSnapshot",
    "RecoveryManager",
    "RecoveryResult",
    "RecoveryStrategy",
    "RollbackPreview",
    "SelectiveRollbackResult",
    # Observability (Phase 4.3)
    "ExecutorCallbacks",
    "ExecutorMetrics",
    "TaskMetrics",
    "get_executor_tracer",
    "log_recovery_action",
    "log_task_context",
    "log_verification_result",
    # TodoList (Phase 5.12)
    "GroupStrategy",
    "TodoItem",
    "TodoListManager",
    "TodoStatus",
    # State Migration (Phase 1.3)
    "CheckpointTrigger",
    "ExecutorCheckpointer",
    "GraphStatePersistence",
    "MAX_RUN_MEMORY_ENTRIES",
    "MAX_STATE_SIZE_BYTES",
    "MemoryIntegration",
    "StatePruning",
    # HITL (Phase 1.5)
    "ExecutorGraph",
    "HITLDecision",
    "HITLManager",
    "HITLState",
    "InterruptConfig",
    "InterruptPoint",
    "create_executor_graph",
    "create_executor_with_hitl",
    "get_interrupt_lists",
    # Tracing (Phase 1.6.1)
    "ExecutorTracingCallbacks",
    "TracingConfig",
    "create_traced_nodes",
    "create_tracing_callbacks",
    "traced_node",
    # Streaming (Phase 1.6.2)
    "ExecutorStreamEvent",
    "JsonFormatter",
    "MinimalFormatter",
    "OutputFormat",
    "PlainFormatter",
    "StreamEventType",
    "StreamingConfig",
    "create_interrupt_event",
    "create_node_end_event",
    "create_node_error_event",
    "create_node_start_event",
    "create_resume_event",
    "create_run_end_event",
    "create_run_start_event",
    "create_task_complete_event",
    "create_task_failed_event",
    "create_task_start_event",
    "get_formatter",
    "stream_to_console",
    # Verifier
    "CheckLevel",
    "CheckResult",
    "CheckStatus",
    # Loop
    "ExecutionResult",
    "ExecutionStatus",
    "Executor",
    "ExecutorConfig",
    "ReviewInfo",
    "VerifyMode",
    # State
    "ExecutorState",
    # Failure
    "FailureAnalyzer",
    "FailureCategory",
    "FailureRecord",
    "FailureSeverity",
    "FailureStats",
    # Learning (Phase 5.3)
    "FailurePattern",
    "LearningStats",
    "LearningStore",
    "PatternType",
    "PromptRefiner",
    "SuccessPattern",
    "TaskType",
    # Run Memory (Phase 5.8.1)
    "FileAction",
    "RunMemory",
    "TaskOutcome",
    # Project Memory (Phase 5.8.2)
    "FileInfo",
    "ProjectMemory",
    "RunSummary",
    # Outcome Extraction (Phase 5.8.3)
    "ExtractionResult",
    "extract_outcome",
    "extract_outcome_sync",
    # Workspace (Phase 5.4)
    "CrossProjectDependency",
    "DependencyScope",
    "ProjectInfo",
    "ProjectType",
    "Workspace",
    "WorkspaceCheckpointResult",
    "WorkspaceRollbackResult",
    # Parser
    "ParseError",
    "ParsedTask",
    "ParserConfig",
    # Roadmap
    "Phase",
    "Priority",
    "Roadmap",
    "RoadmapParser",
    "RunStatus",
    "RunSummary",
    "Section",
    "StateSummary",
    "Subtask",
    # Models
    "Task",
    "TaskState",
    "TaskStatus",
    "TaskVerifier",
    "VerificationResult",
    # Testing (Phase 4.4)
    "ChaosAgent",
    "MockAgent",
    "MockResponse",
    "TestProject",
]
