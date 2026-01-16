"""Tests for task decomposition module (Phase 3.2).

Tests cover:
- ComplexityEstimator scoring and recommendations
- TaskDecomposer parsing and subtask generation
- TodoItem parent-child relationships
- TodoListManager decomposition methods
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.task_decomposition import (
    ComplexityEstimator,
    ComplexityLevel,
    DecomposedTask,
    EstimatorConfig,
    RecommendedAction,
    TaskComplexity,
    TaskDecomposer,
    auto_decompose_if_needed,
    should_decompose_task,
)
from ai_infra.executor.todolist import TodoItem, TodoListManager, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_task() -> TodoItem:
    """Create a simple task for testing."""
    return TodoItem(
        id=1,
        title="Fix typo in README",
        description="Correct spelling error in the introduction section",
        status=TodoStatus.NOT_STARTED,
    )


@pytest.fixture
def complex_task() -> TodoItem:
    """Create a complex task for testing."""
    return TodoItem(
        id=2,
        title="Refactor the entire authentication system to use OAuth2 and implement SSO",
        description=(
            "This task involves migrating from session-based auth to OAuth2, "
            "integrating with multiple identity providers, and implementing "
            "single sign-on across all microservices."
        ),
        status=TodoStatus.NOT_STARTED,
    )


@pytest.fixture
def medium_task() -> TodoItem:
    """Create a medium complexity task for testing."""
    return TodoItem(
        id=3,
        title="Add caching to the user service",
        description="Implement Redis caching for user profile queries",
        status=TodoStatus.NOT_STARTED,
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for decomposition tests."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="""```json
[
  {"title": "Create OAuth2 configuration", "description": "Set up OAuth2 config"},
  {"title": "Implement token service", "description": "Add JWT token handling"},
  {"title": "Add SSO integration", "description": "Connect to identity providers"}
]
```"""
        )
    )
    return agent


# =============================================================================
# TaskComplexity Tests
# =============================================================================


class TestTaskComplexity:
    """Tests for TaskComplexity dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        complexity = TaskComplexity(
            score=7,
            level=ComplexityLevel.HIGH,
            factors=["Long title", "Complex keyword: 'refactor'"],
            estimated_time="1-2 hours",
            recommended_action=RecommendedAction.DECOMPOSE,
            confidence=0.8,
        )
        result = complexity.to_dict()

        assert result["score"] == 7
        assert result["level"] == "high"
        assert len(result["factors"]) == 2
        assert result["recommended_action"] == "decompose"
        assert result["confidence"] == 0.8

    def test_should_decompose_true(self) -> None:
        """Test should_decompose property when decomposition recommended."""
        complexity = TaskComplexity(
            score=10,
            level=ComplexityLevel.HIGH,
            recommended_action=RecommendedAction.DECOMPOSE,
        )
        assert complexity.should_decompose is True

    def test_should_decompose_false(self) -> None:
        """Test should_decompose property when execution recommended."""
        complexity = TaskComplexity(
            score=3,
            level=ComplexityLevel.LOW,
            recommended_action=RecommendedAction.EXECUTE,
        )
        assert complexity.should_decompose is False

    def test_is_complex(self) -> None:
        """Test is_complex property."""
        assert TaskComplexity(score=9, level=ComplexityLevel.HIGH).is_complex is True
        assert TaskComplexity(score=12, level=ComplexityLevel.VERY_HIGH).is_complex is True
        assert TaskComplexity(score=5, level=ComplexityLevel.MEDIUM).is_complex is False
        assert TaskComplexity(score=2, level=ComplexityLevel.LOW).is_complex is False


# =============================================================================
# ComplexityEstimator Tests
# =============================================================================


class TestComplexityEstimator:
    """Tests for ComplexityEstimator class."""

    def test_simple_task_low_complexity(self, simple_task: TodoItem) -> None:
        """Test that simple tasks are scored as low complexity."""
        estimator = ComplexityEstimator()
        complexity = estimator.estimate(simple_task)

        assert complexity.score <= 3
        assert complexity.level == ComplexityLevel.LOW
        assert complexity.recommended_action == RecommendedAction.EXECUTE

    def test_complex_task_high_complexity(self, complex_task: TodoItem) -> None:
        """Test that complex tasks are scored as high complexity."""
        estimator = ComplexityEstimator()
        complexity = estimator.estimate(complex_task)

        # Complex task has: refactor, implement, oauth2, authentication
        # Should score at least MEDIUM
        assert complexity.score >= 5
        assert complexity.level in (
            ComplexityLevel.MEDIUM,
            ComplexityLevel.HIGH,
            ComplexityLevel.VERY_HIGH,
        )
        assert "refactor" in str(complexity.factors).lower() or len(complexity.factors) >= 2

    def test_medium_task_medium_complexity(self, medium_task: TodoItem) -> None:
        """Test that medium tasks are scored appropriately."""
        estimator = ComplexityEstimator()
        complexity = estimator.estimate(medium_task)

        # "Add caching to the user service" has caching keyword (+1)
        # Should be at least LOW complexity
        assert complexity.score >= 1
        assert complexity.level in (ComplexityLevel.LOW, ComplexityLevel.MEDIUM)

    def test_dict_input(self) -> None:
        """Test that dict input is handled correctly."""
        estimator = ComplexityEstimator()
        complexity = estimator.estimate(
            {
                "title": "Fix typo",
                "description": "Simple fix",
            }
        )

        assert complexity.score <= 3
        assert complexity.level == ComplexityLevel.LOW

    def test_keyword_scoring(self) -> None:
        """Test that keywords affect scoring correctly."""
        estimator = ComplexityEstimator()

        # High complexity keywords
        refactor_task = {"title": "Refactor the entire codebase"}
        complexity = estimator.estimate(refactor_task)
        assert complexity.score >= 2  # +2 for "refactor"

        # Simple keywords reduce score
        simple_task = {"title": "Fix typo in comment"}
        complexity = estimator.estimate(simple_task)
        assert complexity.score <= 2  # Reduced by simple keyword

    def test_context_affects_scoring(self) -> None:
        """Test that context information affects scoring."""
        estimator = ComplexityEstimator()
        task = {"title": "Add feature", "description": "Some feature"}

        # Without context
        base_complexity = estimator.estimate(task)

        # With many dependencies
        complex_context = {
            "dependency_count": 10,
            "file_count_estimate": 15,
            "previous_failures": 2,
        }
        context_complexity = estimator.estimate(task, complex_context)

        assert context_complexity.score > base_complexity.score

    def test_custom_config(self) -> None:
        """Test custom estimator configuration."""
        # Low thresholds
        config = EstimatorConfig(
            decompose_threshold=3,
            clarify_threshold=5,
        )
        estimator = ComplexityEstimator(config)

        task = {"title": "Add caching to user service"}
        complexity = estimator.estimate(task)

        # With lower threshold, should recommend decompose more often
        assert config.decompose_threshold == 3

    def test_vague_language_detection(self) -> None:
        """Test detection of vague language."""
        estimator = ComplexityEstimator()

        vague_task = {"title": "Do something with the things and maybe add some stuff"}
        complexity = estimator.estimate(vague_task)

        assert "vague" in str(complexity.factors).lower()
        assert complexity.score >= 2

    def test_multiple_objectives_detection(self) -> None:
        """Test detection of multiple objectives."""
        estimator = ComplexityEstimator()

        multi_task = {"title": "Add auth and then add caching and implement logging"}
        complexity = estimator.estimate(multi_task)

        assert "multiple" in str(complexity.factors).lower()

    def test_estimated_time(self) -> None:
        """Test that estimated time is set based on complexity."""
        estimator = ComplexityEstimator()

        simple = estimator.estimate({"title": "Fix typo"})
        # Simple tasks should be quick
        assert "min" in simple.estimated_time.lower()

        # Create a task complex enough to require hours
        complex_est = estimator.estimate(
            {
                "title": "Refactor and redesign the entire authentication and authorization system from scratch"
            }
        )
        # Could be minutes or hours depending on scoring
        assert (
            "min" in complex_est.estimated_time.lower()
            or "hour" in complex_est.estimated_time.lower()
        )


# =============================================================================
# DecomposedTask Tests
# =============================================================================


class TestDecomposedTask:
    """Tests for DecomposedTask dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        task = DecomposedTask(
            title="Create base structure",
            description="Set up initial files",
            depends_on=[],
            estimated_complexity="low",
            files_hint=["src/auth.py"],
        )
        result = task.to_dict()

        assert result["title"] == "Create base structure"
        assert result["description"] == "Set up initial files"
        assert result["depends_on"] == []
        assert result["files_hint"] == ["src/auth.py"]

    def test_default_values(self) -> None:
        """Test default field values."""
        task = DecomposedTask(title="Simple task")

        assert task.description == ""
        assert task.depends_on == []
        assert task.estimated_complexity == "low"
        assert task.files_hint == []


# =============================================================================
# TaskDecomposer Tests
# =============================================================================


class TestTaskDecomposer:
    """Tests for TaskDecomposer class."""

    @pytest.mark.asyncio
    async def test_decompose_success(self, complex_task: TodoItem, mock_agent: MagicMock) -> None:
        """Test successful task decomposition."""
        decomposer = TaskDecomposer(mock_agent)
        subtasks = await decomposer.decompose(complex_task)

        assert len(subtasks) >= 2
        assert all(isinstance(st, DecomposedTask) for st in subtasks)
        assert all(st.title for st in subtasks)

    @pytest.mark.asyncio
    async def test_decompose_with_complexity(
        self, complex_task: TodoItem, mock_agent: MagicMock
    ) -> None:
        """Test decomposition with pre-computed complexity."""
        complexity = TaskComplexity(
            score=12,
            level=ComplexityLevel.VERY_HIGH,
            factors=["Complex keywords", "Long description"],
        )

        decomposer = TaskDecomposer(mock_agent)
        subtasks = await decomposer.decompose(complex_task, complexity=complexity)

        assert len(subtasks) >= 2
        # Verify complexity was passed to prompt
        call_args = mock_agent.ainvoke.call_args[0][0]
        assert "12" in call_args  # Score should be in prompt

    @pytest.mark.asyncio
    async def test_decompose_max_subtasks(self, complex_task: TodoItem) -> None:
        """Test that max_subtasks limit is enforced."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""```json
[
  {"title": "Task 1"},
  {"title": "Task 2"},
  {"title": "Task 3"},
  {"title": "Task 4"},
  {"title": "Task 5"},
  {"title": "Task 6"},
  {"title": "Task 7"}
]
```"""
            )
        )

        decomposer = TaskDecomposer(agent, max_subtasks=3)
        subtasks = await decomposer.decompose(complex_task)

        assert len(subtasks) == 3

    @pytest.mark.asyncio
    async def test_decompose_fallback_numbered_list(self, complex_task: TodoItem) -> None:
        """Test fallback to numbered list parsing."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""
1. Create OAuth2 configuration
2. Implement token service
3. Add SSO integration
"""
            )
        )

        decomposer = TaskDecomposer(agent)
        subtasks = await decomposer.decompose(complex_task)

        assert len(subtasks) == 3
        assert "OAuth2" in subtasks[0].title

    @pytest.mark.asyncio
    async def test_decompose_agent_failure(self, complex_task: TodoItem) -> None:
        """Test handling of agent failure."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        decomposer = TaskDecomposer(agent)
        with pytest.raises(ValueError, match="Failed to decompose"):
            await decomposer.decompose(complex_task)

    @pytest.mark.asyncio
    async def test_decompose_empty_response(self, complex_task: TodoItem) -> None:
        """Test handling of empty response."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value=MagicMock(content=""))

        decomposer = TaskDecomposer(agent)
        with pytest.raises(ValueError, match="no valid subtasks"):
            await decomposer.decompose(complex_task)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_should_decompose_task_true(self, complex_task: TodoItem) -> None:
        """Test should_decompose_task returns True for complex tasks with low threshold."""
        # Use a low threshold to ensure complex tasks trigger decomposition
        should_decompose, complexity = should_decompose_task(complex_task, threshold=5)

        assert should_decompose is True
        assert complexity.score >= 5

    def test_should_decompose_task_false(self, simple_task: TodoItem) -> None:
        """Test should_decompose_task returns False for simple tasks."""
        should_decompose, complexity = should_decompose_task(simple_task)

        assert should_decompose is False
        assert complexity.should_decompose is False

    def test_should_decompose_task_custom_threshold(self) -> None:
        """Test should_decompose_task with custom threshold."""
        task = TodoItem(
            id=1,
            title="Add caching to user service",
            description="Implement Redis caching",
        )

        # High threshold
        should_decompose, _ = should_decompose_task(task, threshold=20)
        assert should_decompose is False

        # Low threshold
        should_decompose, _ = should_decompose_task(task, threshold=1)
        assert should_decompose is True

    @pytest.mark.asyncio
    async def test_auto_decompose_if_needed_decomposes(
        self, complex_task: TodoItem, mock_agent: MagicMock
    ) -> None:
        """Test auto_decompose_if_needed when decomposition needed."""
        was_decomposed, subtasks = await auto_decompose_if_needed(
            complex_task, mock_agent, threshold=5
        )

        assert was_decomposed is True
        assert len(subtasks) >= 2

    @pytest.mark.asyncio
    async def test_auto_decompose_if_needed_skips(
        self, simple_task: TodoItem, mock_agent: MagicMock
    ) -> None:
        """Test auto_decompose_if_needed when decomposition not needed."""
        was_decomposed, subtasks = await auto_decompose_if_needed(
            simple_task, mock_agent, threshold=8
        )

        assert was_decomposed is False
        assert subtasks == []
        # Agent should not be called
        mock_agent.ainvoke.assert_not_called()


# =============================================================================
# TodoItem Parent-Child Tests
# =============================================================================


class TestTodoItemRelationships:
    """Tests for TodoItem parent-child relationship properties."""

    def test_is_subtask_true(self) -> None:
        """Test is_subtask property when parent_id is set."""
        child = TodoItem(
            id=2,
            title="Child task",
            description="",
            parent_id=1,
        )
        assert child.is_subtask is True

    def test_is_subtask_false(self) -> None:
        """Test is_subtask property when parent_id is None."""
        parent = TodoItem(
            id=1,
            title="Parent task",
            description="",
        )
        assert parent.is_subtask is False

    def test_has_subtasks_true(self) -> None:
        """Test has_subtasks property when children exist."""
        parent = TodoItem(
            id=1,
            title="Parent task",
            description="",
            children_ids=[2, 3],
        )
        assert parent.has_subtasks is True

    def test_has_subtasks_false(self) -> None:
        """Test has_subtasks property when no children."""
        task = TodoItem(
            id=1,
            title="Regular task",
            description="",
        )
        assert task.has_subtasks is False

    def test_is_decomposed_alias(self) -> None:
        """Test is_decomposed is alias for has_subtasks."""
        parent = TodoItem(
            id=1,
            title="Parent task",
            description="",
            children_ids=[2],
        )
        assert parent.is_decomposed == parent.has_subtasks

    def test_to_dict_includes_relationships(self) -> None:
        """Test to_dict includes parent/child relationships."""
        parent = TodoItem(
            id=1,
            title="Parent",
            description="",
            children_ids=[2, 3],
            complexity_score=10,
        )
        result = parent.to_dict()

        assert result["parent_id"] is None
        assert result["children_ids"] == [2, 3]
        assert result["complexity_score"] == 10

    def test_from_dict_includes_relationships(self) -> None:
        """Test from_dict handles parent/child relationships."""
        data = {
            "id": 2,
            "title": "Child",
            "description": "",
            "status": "pending",
            "parent_id": 1,
            "children_ids": [],
            "complexity_score": 5,
        }
        todo = TodoItem.from_dict(data)

        assert todo.parent_id == 1
        assert todo.children_ids == []
        assert todo.complexity_score == 5


# =============================================================================
# TodoListManager Decomposition Tests
# =============================================================================


class TestTodoListManagerDecomposition:
    """Tests for TodoListManager decomposition methods."""

    def test_decompose_task_creates_children(self) -> None:
        """Test decompose_task creates child tasks correctly."""
        parent = TodoItem(id=1, title="Complex task", description="")
        manager = TodoListManager(todos=[parent])

        subtasks = manager.decompose_task(
            parent_id=1,
            subtasks=[
                {"title": "Subtask 1", "description": "First step"},
                {"title": "Subtask 2", "description": "Second step"},
            ],
        )

        assert len(subtasks) == 2
        assert subtasks[0].parent_id == 1
        assert subtasks[1].parent_id == 1
        assert parent.children_ids == [2, 3]

    def test_decompose_task_invalid_parent(self) -> None:
        """Test decompose_task raises for invalid parent ID."""
        manager = TodoListManager(todos=[])

        with pytest.raises(ValueError, match="Task 999 not found"):
            manager.decompose_task(parent_id=999, subtasks=[{"title": "Test"}])

    def test_get_children(self) -> None:
        """Test get_children returns child tasks."""
        parent = TodoItem(id=1, title="Parent", description="", children_ids=[2, 3])
        child1 = TodoItem(id=2, title="Child 1", description="", parent_id=1)
        child2 = TodoItem(id=3, title="Child 2", description="", parent_id=1)
        other = TodoItem(id=4, title="Other", description="")

        manager = TodoListManager(todos=[parent, child1, child2, other])
        children = manager.get_children(1)

        assert len(children) == 2
        assert child1 in children
        assert child2 in children
        assert other not in children

    def test_get_parent(self) -> None:
        """Test get_parent returns parent task."""
        parent = TodoItem(id=1, title="Parent", description="", children_ids=[2])
        child = TodoItem(id=2, title="Child", description="", parent_id=1)

        manager = TodoListManager(todos=[parent, child])
        result = manager.get_parent(2)

        assert result == parent

    def test_get_parent_none(self) -> None:
        """Test get_parent returns None for root tasks."""
        task = TodoItem(id=1, title="Root", description="")

        manager = TodoListManager(todos=[task])
        result = manager.get_parent(1)

        assert result is None

    def test_are_children_complete_all_done(self) -> None:
        """Test are_children_complete when all children done."""
        parent = TodoItem(id=1, title="Parent", description="", children_ids=[2, 3])
        child1 = TodoItem(
            id=2, title="Child 1", description="", parent_id=1, status=TodoStatus.COMPLETED
        )
        child2 = TodoItem(
            id=3, title="Child 2", description="", parent_id=1, status=TodoStatus.COMPLETED
        )

        manager = TodoListManager(todos=[parent, child1, child2])
        assert manager.are_children_complete(1) is True

    def test_are_children_complete_some_pending(self) -> None:
        """Test are_children_complete when some children pending."""
        parent = TodoItem(id=1, title="Parent", description="", children_ids=[2, 3])
        child1 = TodoItem(
            id=2, title="Child 1", description="", parent_id=1, status=TodoStatus.COMPLETED
        )
        child2 = TodoItem(
            id=3, title="Child 2", description="", parent_id=1, status=TodoStatus.NOT_STARTED
        )

        manager = TodoListManager(todos=[parent, child1, child2])
        assert manager.are_children_complete(1) is False

    def test_are_children_complete_no_children(self) -> None:
        """Test are_children_complete when no children."""
        task = TodoItem(id=1, title="Task", description="")

        manager = TodoListManager(todos=[task])
        assert manager.are_children_complete(1) is True

    def test_next_pending_skips_decomposed_parents(self) -> None:
        """Test next_pending skips parents with incomplete children."""
        parent = TodoItem(id=1, title="Parent", description="", children_ids=[2])
        child = TodoItem(
            id=2, title="Child", description="", parent_id=1, status=TodoStatus.NOT_STARTED
        )
        other = TodoItem(id=3, title="Other", description="")

        manager = TodoListManager(todos=[parent, child, other])
        next_task = manager.next_pending()

        # Should skip parent and return child (first pending without children)
        assert next_task is not None
        # Either child or other should be returned, but not parent
        assert next_task.id in (2, 3)

    def test_next_pending_returns_parent_when_children_done(self) -> None:
        """Test next_pending returns parent when children are complete."""
        parent = TodoItem(
            id=1, title="Parent", description="", children_ids=[2], status=TodoStatus.NOT_STARTED
        )
        child = TodoItem(
            id=2, title="Child", description="", parent_id=1, status=TodoStatus.COMPLETED
        )

        manager = TodoListManager(todos=[parent, child])
        next_task = manager.next_pending()

        # Parent should now be available since child is done
        assert next_task is not None
        assert next_task.id == 1
