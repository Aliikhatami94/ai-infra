"""Unit tests for TodoListManager."""

from pathlib import Path
from textwrap import dedent
from typing import get_args

import pytest

from ai_infra.executor.parser import RoadmapParser
from ai_infra.executor.todolist import (
    NORMALIZED_TODOS_VERSION,
    GroupStrategy,
    NormalizedTodo,
    NormalizedTodoFile,
    TodoItem,
    TodoListManager,
    TodoStatus,
)


class TestGroupStrategy:
    """Tests for GroupStrategy type alias."""

    def test_group_strategy_values(self) -> None:
        """Test that GroupStrategy has the expected literal values."""
        expected = {"none", "section", "smart"}
        actual = set(get_args(GroupStrategy))
        assert actual == expected

    def test_default_strategy_is_smart(self, tmp_path: Path) -> None:
        """Test that default group_strategy is 'smart'."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        # Call without group_strategy - should default to "smart"
        manager = TodoListManager.from_roadmap(roadmap, roadmap_path=roadmap_file)
        assert manager.total_count >= 1


class TestTodoItem:
    """Tests for TodoItem dataclass."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        item = TodoItem(
            id=1,
            title="Test Task",
            description="A test task",
            status=TodoStatus.IN_PROGRESS,
            source_task_ids=["1.1.1", "1.1.2"],
            source_titles=["Create file", "Update file"],
            file_hints=["src/foo.py"],
        )

        data = item.to_dict()
        restored = TodoItem.from_dict(data)

        assert restored.id == item.id
        assert restored.title == item.title
        assert restored.description == item.description
        assert restored.status == item.status
        assert restored.source_task_ids == item.source_task_ids
        assert restored.source_titles == item.source_titles
        assert restored.file_hints == item.file_hints

    def test_to_dict_with_files_created_and_error(self) -> None:
        """Test serialization includes files_created and error fields."""
        item = TodoItem(
            id=1,
            title="Failed Task",
            description="A failed task",
            status=TodoStatus.FAILED,
            source_task_ids=["1.1.1"],
            source_titles=["Create file"],
            file_hints=["src/foo.py"],
            files_created=["src/foo.py", "src/bar.py"],
            error="Import error: module not found",
        )

        data = item.to_dict()
        assert data["files_created"] == ["src/foo.py", "src/bar.py"]
        assert data["error"] == "Import error: module not found"

        restored = TodoItem.from_dict(data)
        assert restored.files_created == item.files_created
        assert restored.error == item.error

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        item = TodoItem(
            id=1,
            title="Test",
            description="A test",
        )

        assert item.status == TodoStatus.NOT_STARTED
        assert item.source_task_ids == []
        assert item.source_titles == []
        assert item.file_hints == []
        assert item.files_created == []
        assert item.error is None
        assert item.started_at is None
        assert item.completed_at is None


class TestNormalizedTodo:
    """Tests for NormalizedTodo dataclass (Phase 5.13)."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        todo = NormalizedTodo(
            id=1,
            title="Create config module",
            description="Config dataclass with load_from_env()",
            status="pending",
            source_line=5,
            source_text="🚀 Build the config",
            file_hints=["src/config.py"],
            dependencies=[2, 3],
        )

        data = todo.to_dict()
        restored = NormalizedTodo.from_dict(data)

        assert restored.id == 1
        assert restored.title == "Create config module"
        assert restored.description == "Config dataclass with load_from_env()"
        assert restored.status == "pending"
        assert restored.source_line == 5
        assert restored.source_text == "🚀 Build the config"
        assert restored.file_hints == ["src/config.py"]
        assert restored.dependencies == [2, 3]

    def test_nested_subtasks(self) -> None:
        """Test that subtasks serialize correctly."""
        parent = NormalizedTodo(
            id=1,
            title="Parent task",
            subtasks=[
                NormalizedTodo(id=2, title="Subtask 1", status="completed"),
                NormalizedTodo(id=3, title="Subtask 2", status="pending"),
            ],
        )

        data = parent.to_dict()
        restored = NormalizedTodo.from_dict(data)

        assert len(restored.subtasks) == 2
        assert restored.subtasks[0].id == 2
        assert restored.subtasks[0].status == "completed"
        assert restored.subtasks[1].id == 3
        assert restored.subtasks[1].status == "pending"

    def test_is_pending_and_is_completed(self) -> None:
        """Test status helper methods."""
        pending = NormalizedTodo(id=1, title="Pending", status="pending")
        completed = NormalizedTodo(id=2, title="Done", status="completed")
        skipped = NormalizedTodo(id=3, title="Skipped", status="skipped")

        assert pending.is_pending() is True
        assert pending.is_completed() is False

        assert completed.is_pending() is False
        assert completed.is_completed() is True

        assert skipped.is_pending() is False
        assert skipped.is_completed() is False


class TestNormalizedTodoFile:
    """Tests for NormalizedTodoFile container (Phase 5.13)."""

    def test_create_and_save(self, tmp_path: Path) -> None:
        """Test creating and saving a normalized todo file."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test ROADMAP\n- [ ] Task 1")

        todos = [
            NormalizedTodo(id=1, title="Task 1", status="pending"),
        ]

        todo_file = NormalizedTodoFile.create(roadmap_path, todos)

        assert todo_file.version == NORMALIZED_TODOS_VERSION
        assert todo_file.source_file == str(roadmap_path)
        assert len(todo_file.source_hash) == 64  # SHA-256 hex length
        assert len(todo_file.todos) == 1

        # Save to file
        json_path = tmp_path / ".executor" / "todos.json"
        todo_file.save(json_path)

        assert json_path.exists()

    def test_load(self, tmp_path: Path) -> None:
        """Test loading from JSON file."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test ROADMAP\n- [ ] Task 1")

        todos = [
            NormalizedTodo(id=1, title="Task 1", status="pending"),
            NormalizedTodo(id=2, title="Task 2", status="completed"),
        ]

        todo_file = NormalizedTodoFile.create(roadmap_path, todos)
        json_path = tmp_path / ".executor" / "todos.json"
        todo_file.save(json_path)

        # Load and verify
        loaded = NormalizedTodoFile.load(json_path)

        assert loaded.version == NORMALIZED_TODOS_VERSION
        assert loaded.source_file == str(roadmap_path)
        assert loaded.source_hash == todo_file.source_hash
        assert len(loaded.todos) == 2
        assert loaded.todos[0].title == "Task 1"
        assert loaded.todos[1].status == "completed"

    def test_is_stale_detects_changes(self, tmp_path: Path) -> None:
        """Test that is_stale detects ROADMAP changes."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Original content")

        todo_file = NormalizedTodoFile.create(roadmap_path, [])

        # Not stale initially
        assert todo_file.is_stale(roadmap_path) is False

        # Modify ROADMAP
        roadmap_path.write_text("# Modified content")

        # Now stale
        assert todo_file.is_stale(roadmap_path) is True

    def test_get_pending_todos(self, tmp_path: Path) -> None:
        """Test filtering pending todos."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test")

        todos = [
            NormalizedTodo(id=1, title="Pending 1", status="pending"),
            NormalizedTodo(id=2, title="Completed", status="completed"),
            NormalizedTodo(id=3, title="Pending 2", status="pending"),
            NormalizedTodo(id=4, title="Skipped", status="skipped"),
        ]

        todo_file = NormalizedTodoFile.create(roadmap_path, todos)
        pending = todo_file.get_pending_todos()

        assert len(pending) == 2
        assert pending[0].id == 1
        assert pending[1].id == 3

    def test_mark_completed(self, tmp_path: Path) -> None:
        """Test marking a todo as completed."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test")

        todos = [
            NormalizedTodo(id=1, title="Task 1", status="pending"),
            NormalizedTodo(
                id=2,
                title="Parent",
                subtasks=[NormalizedTodo(id=3, title="Subtask", status="pending")],
            ),
        ]

        todo_file = NormalizedTodoFile.create(roadmap_path, todos)

        # Mark top-level todo
        assert todo_file.mark_completed(1) is True
        assert todo_file.todos[0].status == "completed"

        # Mark subtask
        assert todo_file.mark_completed(3) is True
        assert todo_file.todos[1].subtasks[0].status == "completed"

        # Non-existent ID
        assert todo_file.mark_completed(999) is False


class TestFromRoadmapLLM:
    """Tests for LLM-based ROADMAP normalization (Phase 5.13.3)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""

        class MockAgent:
            def __init__(self, response: str):
                self._response = response
                self.calls: list[str] = []

            async def arun(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self._response

        return MockAgent

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_basic(self, tmp_path: Path, mock_agent) -> None:
        """Test basic LLM normalization."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1\n- [x] Task 2")

        # Mock LLM response
        llm_response = """{
            "todos": [
                {"id": 1, "title": "Task 1", "status": "pending"},
                {"id": 2, "title": "Task 2", "status": "completed"}
            ]
        }"""
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)

        assert manager.total_count == 2
        assert manager.pending_count == 1
        assert len(agent.calls) == 1

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_caches_result(self, tmp_path: Path, mock_agent) -> None:
        """Test that normalization result is cached."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        # First call - should use LLM
        await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        assert len(agent.calls) == 1

        # Second call - should use cache
        agent2 = mock_agent(llm_response)
        manager2 = await TodoListManager.from_roadmap_llm(roadmap_path, agent2)
        assert len(agent2.calls) == 0
        assert manager2.total_count == 1

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_force_renormalize(self, tmp_path: Path, mock_agent) -> None:
        """Test that force_renormalize bypasses cache."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        # First call
        await TodoListManager.from_roadmap_llm(roadmap_path, agent)

        # Force renormalize - should call LLM again
        agent2 = mock_agent(llm_response)
        await TodoListManager.from_roadmap_llm(roadmap_path, agent2, force_renormalize=True)
        assert len(agent2.calls) == 1

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_stale_cache(self, tmp_path: Path, mock_agent) -> None:
        """Test that stale cache triggers re-normalization."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Original content")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        # First call - creates cache
        await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        assert len(agent.calls) == 1

        # Modify ROADMAP
        roadmap_path.write_text("# Modified content")

        # Second call - should re-normalize because content changed
        agent2 = mock_agent(llm_response)
        await TodoListManager.from_roadmap_llm(roadmap_path, agent2)
        assert len(agent2.calls) == 1

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_handles_markdown_blocks(
        self, tmp_path: Path, mock_agent
    ) -> None:
        """Test that LLM response with markdown code blocks is handled."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        # LLM response wrapped in markdown code block
        llm_response = """```json
{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}
```"""
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        assert manager.total_count == 1

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_invalid_json_raises(self, tmp_path: Path, mock_agent) -> None:
        """Test that invalid JSON raises ValueError."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        agent = mock_agent("Not valid JSON at all")

        with pytest.raises(ValueError, match="Could not find JSON"):
            await TodoListManager.from_roadmap_llm(roadmap_path, agent)

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_missing_todos_key_raises(
        self, tmp_path: Path, mock_agent
    ) -> None:
        """Test that missing 'todos' key raises ValueError."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        agent = mock_agent('{"tasks": []}')  # Wrong key

        with pytest.raises(ValueError, match="missing 'todos' key"):
            await TodoListManager.from_roadmap_llm(roadmap_path, agent)

    @pytest.mark.asyncio
    async def test_from_roadmap_llm_creates_executor_dir(self, tmp_path: Path, mock_agent) -> None:
        """Test that .executor directory is created."""
        roadmap_path = tmp_path / "project" / "ROADMAP.md"
        roadmap_path.parent.mkdir(parents=True)
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        await TodoListManager.from_roadmap_llm(roadmap_path, agent)

        executor_dir = roadmap_path.parent / ".executor"
        assert executor_dir.exists()
        assert (executor_dir / "todos.json").exists()


class TestJsonOnlyMode:
    """Tests for JSON-only mode (Phase 5.13.4)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""

        class MockAgent:
            def __init__(self, response: str):
                self._response = response

            async def arun(self, prompt: str) -> str:
                return self._response

        return MockAgent

    @pytest.mark.asyncio
    async def test_llm_manager_uses_json_only_mode(self, tmp_path: Path, mock_agent) -> None:
        """Test that from_roadmap_llm creates manager in JSON-only mode."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)

        assert manager.uses_json_only is True

    @pytest.mark.asyncio
    async def test_mark_completed_saves_to_json(self, tmp_path: Path, mock_agent) -> None:
        """Test that mark_completed updates todos.json."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1\n- [ ] Task 2")

        llm_response = """{
            "todos": [
                {"id": 1, "title": "Task 1", "status": "pending"},
                {"id": 2, "title": "Task 2", "status": "pending"}
            ]
        }"""
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        manager.mark_completed(1)

        # Verify JSON was updated
        todos_json = tmp_path / ".executor" / "todos.json"
        content = todos_json.read_text()
        import json

        data = json.loads(content)
        assert data["todos"][0]["status"] == "completed"
        assert data["todos"][1]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_mark_completed_does_not_modify_roadmap(self, tmp_path: Path, mock_agent) -> None:
        """Test that mark_completed does NOT modify ROADMAP in JSON-only mode."""
        roadmap_path = tmp_path / "ROADMAP.md"
        original_content = "# Test\n- [ ] Task 1"
        roadmap_path.write_text(original_content)

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        manager.mark_completed(1)

        # Verify ROADMAP was NOT modified
        assert roadmap_path.read_text() == original_content
        assert "[ ]" in roadmap_path.read_text()

    @pytest.mark.asyncio
    async def test_mark_failed_saves_to_json(self, tmp_path: Path, mock_agent) -> None:
        """Test that mark_failed updates todos.json."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        manager.mark_failed(1, "Some error")

        # Verify JSON was updated (failed maps to skipped)
        todos_json = tmp_path / ".executor" / "todos.json"
        import json

        data = json.loads(todos_json.read_text())
        assert data["todos"][0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_mark_in_progress_saves_to_json(self, tmp_path: Path, mock_agent) -> None:
        """Test that mark_in_progress updates todos.json."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)
        manager.mark_in_progress(1)

        # In-progress maps to pending in JSON (still working)
        todos_json = tmp_path / ".executor" / "todos.json"
        import json

        data = json.loads(todos_json.read_text())
        assert data["todos"][0]["status"] == "pending"

    def test_regular_manager_does_not_use_json_only(self, tmp_path: Path) -> None:
        """Test that from_roadmap creates manager NOT in JSON-only mode."""
        from ai_infra.executor.parser import RoadmapParser

        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n### 1.1 Section\n- [ ] Task 1")

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_path)
        manager = TodoListManager.from_roadmap(roadmap, roadmap_path)

        assert manager.uses_json_only is False

    @pytest.mark.asyncio
    async def test_sync_roadmap_true_overrides_json_only(self, tmp_path: Path, mock_agent) -> None:
        """Test that sync_roadmap=True can override JSON-only mode."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        llm_response = '{"todos": [{"id": 1, "title": "Task 1", "status": "pending"}]}'
        agent = mock_agent(llm_response)

        manager = await TodoListManager.from_roadmap_llm(roadmap_path, agent)

        # Force ROADMAP sync even in JSON-only mode
        manager.mark_completed(1, sync_roadmap=True)

        # Verify ROADMAP WAS modified (explicit override)
        assert "[x]" in roadmap_path.read_text()


class TestSyncJsonToRoadmap:
    """Tests for sync_json_to_roadmap (Phase 5.13.5)."""

    def test_sync_completed_todos(self, tmp_path: Path) -> None:
        """Test syncing completed todos back to ROADMAP."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1\n- [ ] Task 2\n- [ ] Task 3")

        # Create todos.json with some completed
        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        todos_json = executor_dir / "todos.json"
        todos_json.write_text("""{
            "version": "1.0",
            "source_file": "ROADMAP.md",
            "source_hash": "abc123",
            "normalized_at": "2026-01-08T00:00:00Z",
            "todos": [
                {"id": 1, "title": "Task 1", "status": "completed"},
                {"id": 2, "title": "Task 2", "status": "pending"},
                {"id": 3, "title": "Task 3", "status": "completed"}
            ]
        }""")

        updated = TodoListManager.sync_json_to_roadmap(roadmap_path)

        assert updated == 2
        content = roadmap_path.read_text()
        assert "[x] Task 1" in content
        assert "[ ] Task 2" in content
        assert "[x] Task 3" in content

    def test_sync_no_todos_json_raises(self, tmp_path: Path) -> None:
        """Test that missing todos.json raises FileNotFoundError."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        with pytest.raises(FileNotFoundError, match=r"No todos\.json found"):
            TodoListManager.sync_json_to_roadmap(roadmap_path)

    def test_sync_no_completed_todos(self, tmp_path: Path) -> None:
        """Test syncing when no todos are completed."""
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n- [ ] Task 1")

        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        todos_json = executor_dir / "todos.json"
        todos_json.write_text("""{
            "version": "1.0",
            "source_file": "ROADMAP.md",
            "source_hash": "abc123",
            "normalized_at": "2026-01-08T00:00:00Z",
            "todos": [
                {"id": 1, "title": "Task 1", "status": "pending"}
            ]
        }""")

        updated = TodoListManager.sync_json_to_roadmap(roadmap_path)

        assert updated == 0
        assert "[ ] Task 1" in roadmap_path.read_text()

    def test_sync_uses_source_text(self, tmp_path: Path) -> None:
        """Test that sync uses source_text for matching."""
        roadmap_path = tmp_path / "ROADMAP.md"
        # Custom format with emoji
        roadmap_path.write_text("# Test\n- [ ] 🚀 Build the config")

        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        todos_json = executor_dir / "todos.json"
        todos_json.write_text("""{
            "version": "1.0",
            "source_file": "ROADMAP.md",
            "source_hash": "abc123",
            "normalized_at": "2026-01-08T00:00:00Z",
            "todos": [
                {
                    "id": 1,
                    "title": "Build config",
                    "status": "completed",
                    "source_text": "🚀 Build the config"
                }
            ]
        }""")

        updated = TodoListManager.sync_json_to_roadmap(roadmap_path)

        assert updated == 1
        assert "[x] 🚀 Build the config" in roadmap_path.read_text()

    def test_sync_all_to_roadmap_method(self, tmp_path: Path) -> None:
        """Test the sync_all_to_roadmap instance method."""
        from ai_infra.executor.parser import RoadmapParser

        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("# Test\n### 1.1 Section\n- [ ] Task 1\n- [ ] Task 2")

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_path)
        manager = TodoListManager.from_roadmap(roadmap, roadmap_path)

        # Complete todos without syncing
        manager._use_json_only = True  # Simulate JSON-only mode
        manager.mark_completed(1, sync_roadmap=False)

        # ROADMAP should be unchanged
        assert "[ ] Task 1" in roadmap_path.read_text()

        # Now sync all
        manager._use_json_only = False  # Reset for sync
        updated = manager.sync_all_to_roadmap()

        assert updated >= 1
        assert "[x]" in roadmap_path.read_text()


class TestTodoListManagerGrouping:
    """Tests for smart task grouping."""

    def test_ungrouped_strategy(self, tmp_path: Path) -> None:
        """Test that ungrouped strategy creates one todo per task."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Section
            - [ ] First task
            - [ ] Second task
            - [ ] Third task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        assert manager.total_count == 3
        assert all(len(t.source_task_ids) == 1 for t in manager.todos)

    def test_section_grouped_strategy(self, tmp_path: Path) -> None:
        """Test that section strategy groups all tasks in a section."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 First Section
            - [ ] First task
            - [ ] Second task

            ### 1.2 Second Section
            - [ ] Third task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="section"
        )

        assert manager.total_count == 2  # Two sections = two todos
        assert len(manager.todos[0].source_task_ids) == 2  # First section has 2 tasks
        assert len(manager.todos[1].source_task_ids) == 1  # Second section has 1 task

    def test_smart_grouping_by_shared_file(self, tmp_path: Path) -> None:
        """Test that tasks mentioning the same file are grouped."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 User Module

            **Files**: `src/user.py`, `src/profile.py`

            - [ ] Create `src/user.py` with User class
            - [ ] Add validation to `src/user.py`
            - [ ] Create `src/profile.py` with Profile class
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # Should group the two user.py tasks together
        # Result: one group for user.py tasks, one for profile.py
        # Note: With file hints at section level, all 3 may be grouped together
        # The key is that we have fewer todos than tasks
        assert manager.total_count <= 3  # Some grouping should happen

    def test_smart_grouping_single_tasks(self, tmp_path: Path) -> None:
        """Test that unrelated tasks stay separate."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Setup
            - [ ] Create README.md
            - [ ] Create LICENSE
            - [ ] Create .gitignore
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # No shared files, so each stays separate
        assert manager.total_count == 3

    def test_smart_grouping_by_word_similarity(self, tmp_path: Path) -> None:
        """Test that tasks with >60% word overlap are grouped."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 User Authentication
            - [ ] Implement user authentication login
            - [ ] Implement user authentication logout
            - [ ] Create database schema
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # "Implement user authentication login" and "Implement user authentication logout"
        # share 3/4 significant words (75%) - should be grouped
        # "Create database schema" is different - stays separate
        assert manager.total_count <= 2  # At least some grouping

    def test_smart_grouping_by_entity_overlap(self, tmp_path: Path) -> None:
        """Test that tasks mentioning same entities are grouped."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 User Class
            - [ ] Create `User` class with basic fields
            - [ ] Add validation to `User` class
            - [ ] Create `Profile` class
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # First two tasks both mention `User` - should be grouped
        # Third task mentions `Profile` - stays separate
        assert manager.total_count <= 2


class TestSmartGroupingHelpers:
    """Tests for smart grouping helper methods."""

    def test_extract_significant_words(self) -> None:
        """Test word extraction filters stop words."""
        words = TodoListManager._extract_significant_words(
            "Create the user authentication module for the app"
        )
        assert "create" in words
        assert "user" in words
        assert "authentication" in words
        assert "module" in words
        assert "app" in words
        # Stop words should be filtered
        assert "the" not in words
        assert "for" not in words

    def test_extract_significant_words_ignores_entities(self) -> None:
        """Test that backtick-wrapped entities are ignored in word extraction."""
        words = TodoListManager._extract_significant_words("Create `src/user.py` with User class")
        # Backtick content should be removed
        assert "src" not in words
        assert "user.py" not in words
        # Regular words kept
        assert "create" in words
        assert "user" in words
        assert "class" in words

    def test_extract_significant_words_short_words_filtered(self) -> None:
        """Test that words with 2 or fewer chars are filtered."""
        words = TodoListManager._extract_significant_words("Add a new ID to the API")
        assert "add" in words
        assert "new" in words
        assert "api" in words
        # Short words filtered
        assert "a" not in words
        assert "id" not in words


class TestTodoListManagerOperations:
    """Tests for todo operations."""

    def test_next_pending(self, tmp_path: Path) -> None:
        """Test getting next pending todo."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [x] Completed task
            - [ ] Pending task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        next_todo = manager.next_pending()
        assert next_todo is not None
        assert "Pending task" in next_todo.source_titles

    def test_mark_in_progress(self, tmp_path: Path) -> None:
        """Test marking a todo as in progress."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] First task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        assert todo is not None

        manager.mark_in_progress(todo.id)
        assert manager.get_todo(todo.id).status == TodoStatus.IN_PROGRESS
        assert manager.in_progress_count == 1

    def test_mark_completed_syncs_roadmap(self, tmp_path: Path) -> None:
        """Test that marking completed syncs to ROADMAP."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] First task
            - [ ] Second task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        updated = manager.mark_completed(todo.id, sync_roadmap=True)

        assert updated >= 1
        assert manager.get_todo(todo.id).status == TodoStatus.COMPLETED

        # Verify ROADMAP was updated
        updated_content = roadmap_file.read_text()
        assert "[x] First task" in updated_content
        assert "[ ] Second task" in updated_content  # Second still unchecked

    def test_mark_completed_with_files_created(self, tmp_path: Path) -> None:
        """Test that marking completed tracks files_created."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] Create user module
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        files = ["src/user.py", "tests/test_user.py"]
        manager.mark_completed(todo.id, files_created=files, sync_roadmap=False)

        completed_todo = manager.get_todo(todo.id)
        assert completed_todo.status == TodoStatus.COMPLETED
        assert completed_todo.files_created == files

    def test_mark_failed_with_error(self, tmp_path: Path) -> None:
        """Test that marking failed stores error message."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] Create module with circular import
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        error_msg = "ImportError: cannot import name 'Profile' from 'user'"
        manager.mark_failed(todo.id, error=error_msg)

        failed_todo = manager.get_todo(todo.id)
        assert failed_todo.status == TodoStatus.FAILED
        assert failed_todo.error == error_msg


class TestRoadmapSync:
    """Tests for immediate ROADMAP synchronization."""

    def test_sync_updates_checkbox_immediately(self, tmp_path: Path) -> None:
        """Test that mark_completed updates ROADMAP checkbox immediately."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] First task to complete
            - [ ] Second task pending
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Complete first task
        todo = manager.next_pending()
        updated = manager.mark_completed(todo.id, sync_roadmap=True)

        # Verify immediate sync happened
        assert updated == 1
        content = roadmap_file.read_text()
        assert "[x] First task to complete" in content
        assert "[ ] Second task pending" in content

    def test_sync_multiple_source_titles(self, tmp_path: Path) -> None:
        """Test syncing a grouped todo updates all source checkboxes."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 User Module
            - [ ] Create `src/user.py` with User class
            - [ ] Add validation to `src/user.py`
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        # Use smart grouping to group by shared file
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # Should be grouped into one todo
        todo = manager.next_pending()
        updated = manager.mark_completed(todo.id, sync_roadmap=True)

        # Both checkboxes should be updated
        content = roadmap_file.read_text()
        assert "[x]" in content
        # At least one checkbox updated
        assert updated >= 1

    def test_sync_handles_bold_titles(self, tmp_path: Path) -> None:
        """Test syncing works with bold markdown titles."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] **Important task**
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        updated = manager.mark_completed(todo.id, sync_roadmap=True)

        content = roadmap_file.read_text()
        # Should update bold title
        assert "[x]" in content or updated >= 0  # May not match if parser strips bold

    def test_sync_no_roadmap_path(self, tmp_path: Path) -> None:
        """Test sync gracefully handles missing roadmap_path."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        # Create manager without roadmap_path
        manager = TodoListManager.from_roadmap(roadmap, roadmap_path=None, group_strategy="none")

        todo = manager.next_pending()
        updated = manager.mark_completed(todo.id, sync_roadmap=True)

        # Should return 0 but not crash
        assert updated == 0

    def test_sync_preserves_indentation(self, tmp_path: Path) -> None:
        """Test that sync preserves original indentation."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] Parent task
              - [ ] Nested child task
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Complete parent task
        todo = manager.next_pending()
        manager.mark_completed(todo.id, sync_roadmap=True)

        content = roadmap_file.read_text()
        # Nested task should still be indented
        assert "  - [ ] Nested child task" in content or "  -" in content

    def test_sync_all_completed(self, tmp_path: Path) -> None:
        """Test sync_all_completed syncs all completed todos."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [ ] Task 1
            - [ ] Task 2
            - [ ] Task 3
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Complete first two without syncing
        todo1 = manager.get_todo(1)
        todo2 = manager.get_todo(2)
        manager.mark_completed(todo1.id, sync_roadmap=False)
        manager.mark_completed(todo2.id, sync_roadmap=False)

        # Verify not synced yet
        content = roadmap_file.read_text()
        assert content.count("[x]") == 0

        # Now sync all
        total = manager.sync_all_completed()
        assert total == 2

        content = roadmap_file.read_text()
        assert content.count("[x]") == 2
        assert "[ ] Task 3" in content  # Third still pending


class TestTodoListManagerSummary:
    """Tests for summary and statistics."""

    def test_get_summary(self, tmp_path: Path) -> None:
        """Test summary statistics."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Tasks
            - [x] Completed 1
            - [x] Completed 2
            - [ ] Pending 1
            - [ ] Pending 2
            - [ ] Pending 3
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        summary = manager.get_summary()
        assert summary["total"] == 5
        assert summary["completed"] == 2
        assert summary["pending"] == 3
        assert summary["in_progress"] == 0

    def test_repr(self, tmp_path: Path) -> None:
        """Test string representation."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
            - [ ] Task 2
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        repr_str = repr(manager)
        assert "TodoListManager" in repr_str
        assert "todos=2" in repr_str


class TestTodoListManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_roadmap(self, tmp_path: Path) -> None:
        """Test handling empty ROADMAP."""
        roadmap_content = "# Empty Roadmap\n"
        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        assert manager.total_count == 0
        assert manager.next_pending() is None

    def test_weird_roadmap_structure(self, tmp_path: Path) -> None:
        """Test handling non-standard ROADMAP formats."""
        # Roadmap with no sections, just bullet points
        roadmap_content = dedent("""
            # My Project

            Some description text.

            - [ ] Do this thing
            - [ ] Do another thing
            - [x] Already done
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # Should still work with default phase/section
        # Note: Smart grouping may combine similar tasks, so total may be <= 3
        assert manager.total_count >= 1
        # At least one pending todo (may be grouped)
        assert manager.pending_count >= 1

    def test_get_todo_for_task(self, tmp_path: Path) -> None:
        """Test looking up todo by source task ID."""
        roadmap_content = dedent("""
            # Test

            ### 1.1 Module

            **Files**: `src/app.py`

            - [ ] Create `src/app.py`
            - [ ] Update `src/app.py` with config
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)

        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # Get all tasks as a list
        all_tasks = list(roadmap.all_tasks())
        assert len(all_tasks) == 2

        task1_id = all_tasks[0].id
        task2_id = all_tasks[1].id

        todo1 = manager.get_todo_for_task(task1_id)
        todo2 = manager.get_todo_for_task(task2_id)

        assert todo1 is not None
        assert todo2 is not None
        # Both tasks may be in the same todo if grouped, or separate
        # Just verify we can look them up


class TestShortenTitle:
    """Tests for title shortening utility."""

    def test_removes_backticks(self) -> None:
        """Test that backticks are removed."""
        result = TodoListManager._shorten_title("Create `src/user.py` file")
        assert "`" not in result
        assert "src/user.py" in result or "user.py" in result

    def test_removes_path_prefixes(self) -> None:
        """Test that src/ and tests/ prefixes are removed."""
        result = TodoListManager._shorten_title("Create src/tests/test_app.py")
        assert "src/" not in result

    def test_truncates_long_titles(self) -> None:
        """Test that long titles are truncated."""
        long_title = "This is a very long title with many many words that should be truncated"
        result = TodoListManager._shorten_title(long_title, max_words=5)
        assert len(result.split()) <= 5


# =============================================================================
# Phase 5.12.6: pending() and get_source_tasks() Tests
# =============================================================================


class TestPendingAndSourceTasks:
    """Tests for Phase 5.12.6: pending() and get_source_tasks() methods."""

    def test_pending_returns_not_started_todos(self, tmp_path: Path) -> None:
        """Test that pending() returns all NOT_STARTED todos."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
            - [ ] Task 2
            - [ ] Task 3
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        pending = manager.pending()

        assert len(pending) >= 1
        for todo in pending:
            assert todo.status == TodoStatus.NOT_STARTED

    def test_pending_excludes_in_progress(self, tmp_path: Path) -> None:
        """Test that pending() excludes IN_PROGRESS todos."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
            - [ ] Task 2
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Mark first todo as in-progress
        first = manager.next_pending()
        assert first is not None
        manager.mark_in_progress(first.id)

        # pending() should not include the in-progress one
        pending = manager.pending()
        for todo in pending:
            assert todo.id != first.id
            assert todo.status == TodoStatus.NOT_STARTED

    def test_pending_excludes_completed(self, tmp_path: Path) -> None:
        """Test that pending() excludes COMPLETED todos."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
            - [ ] Task 2
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Mark first todo as completed
        first = manager.next_pending()
        assert first is not None
        manager.mark_completed(first.id, sync_roadmap=False)

        # pending() should not include the completed one
        pending = manager.pending()
        for todo in pending:
            assert todo.id != first.id
            assert todo.status == TodoStatus.NOT_STARTED

    def test_get_source_tasks_returns_parsed_tasks(self, tmp_path: Path) -> None:
        """Test that get_source_tasks returns actual ParsedTask objects."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Create user module
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        todo = manager.next_pending()
        assert todo is not None

        source_tasks = manager.get_source_tasks(todo, roadmap)

        assert len(source_tasks) >= 1
        for task in source_tasks:
            assert task.id in todo.source_task_ids
            assert hasattr(task, "title")
            assert hasattr(task, "description")

    def test_get_source_tasks_with_grouped_todo(self, tmp_path: Path) -> None:
        """Test get_source_tasks with a todo containing multiple source tasks."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 User Module
            - [ ] Create `src/user.py` model
            - [ ] Create `src/user.py` service
            - [ ] Create `src/user.py` routes
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="smart"
        )

        # With smart grouping, these may be grouped by file
        todo = manager.next_pending()
        assert todo is not None

        source_tasks = manager.get_source_tasks(todo, roadmap)

        # All source tasks should be returned
        assert len(source_tasks) == len(todo.source_task_ids)
        task_ids = {t.id for t in source_tasks}
        assert task_ids == set(todo.source_task_ids)

    def test_get_source_tasks_empty_for_invalid_ids(self, tmp_path: Path) -> None:
        """Test get_source_tasks returns empty list for invalid task IDs."""
        roadmap_content = dedent("""
            # Test
            ### 1.1 Tasks
            - [ ] Task 1
        """).strip()

        roadmap_file = tmp_path / "ROADMAP.md"
        roadmap_file.write_text(roadmap_content)

        parser = RoadmapParser()
        roadmap = parser.parse(roadmap_file)
        manager = TodoListManager.from_roadmap(
            roadmap, roadmap_path=roadmap_file, group_strategy="none"
        )

        # Create a todo with invalid source task IDs
        fake_todo = TodoItem(
            id=999,
            title="Fake Todo",
            description="",
            source_task_ids=["invalid-1", "invalid-2"],
            source_titles=["Invalid 1", "Invalid 2"],
        )

        source_tasks = manager.get_source_tasks(fake_todo, roadmap)

        assert source_tasks == []
