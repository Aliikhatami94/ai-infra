"""Tests for replay module - WorkflowRecorder and replay() function."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ai_infra.replay import MemoryStorage, ReplayResult, WorkflowRecorder, WorkflowStep, replay
from ai_infra.replay.replay import delete_recording, get_recording, list_recordings
from ai_infra.replay.storage import SQLiteStorage


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_create_step(self):
        step = WorkflowStep(
            step_id=0,
            step_type="llm_call",
            timestamp=datetime.now(),
            data={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert step.step_id == 0
        assert step.step_type == "llm_call"
        assert "messages" in step.data

    def test_to_dict(self):
        step = WorkflowStep(
            step_id=1,
            step_type="tool_call",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            data={"name": "get_weather", "args": {"city": "NYC"}},
        )

        data = step.to_dict()

        assert data["step_id"] == 1
        assert data["step_type"] == "tool_call"
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["data"]["name"] == "get_weather"

    def test_from_dict(self):
        data = {
            "step_id": 2,
            "step_type": "tool_result",
            "timestamp": "2024-01-01T12:00:00",
            "data": {"name": "get_weather", "result": {"temp": 72}},
        }

        step = WorkflowStep.from_dict(data)

        assert step.step_id == 2
        assert step.step_type == "tool_result"
        assert step.data["result"]["temp"] == 72


class TestWorkflowRecorder:
    """Tests for WorkflowRecorder class."""

    def test_record_llm_call(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        step = recorder.record_llm_call(
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi there!",
            model="gpt-4",
        )

        assert step.step_id == 0
        assert step.step_type == "llm_call"
        assert step.data["messages"][0]["content"] == "Hello"
        assert step.data["response"] == "Hi there!"
        assert step.data["model"] == "gpt-4"

    def test_record_tool_call(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        step = recorder.record_tool_call(
            tool_name="get_weather",
            args={"city": "NYC"},
        )

        assert step.step_id == 0
        assert step.step_type == "tool_call"
        assert step.data["name"] == "get_weather"
        assert step.data["args"]["city"] == "NYC"

    def test_record_tool_result(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        step = recorder.record_tool_result(
            tool_name="get_weather",
            result={"temp": 72, "conditions": "sunny"},
            duration_ms=150.5,
        )

        assert step.step_id == 0
        assert step.step_type == "tool_result"
        assert step.data["name"] == "get_weather"
        assert step.data["result"]["temp"] == 72
        assert step.data["duration_ms"] == 150.5

    def test_record_agent_response(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        step = recorder.record_agent_response(
            content="The weather in NYC is 72°F and sunny.",
        )

        assert step.step_id == 0
        assert step.step_type == "agent_response"
        assert "72°F" in step.data["content"]

    def test_step_counter_increments(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        recorder.record_llm_call(messages=[])
        recorder.record_tool_call("tool1", {})
        recorder.record_tool_result("tool1", {})
        step4 = recorder.record_agent_response("Done")

        assert step4.step_id == 3

    def test_save_and_load(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_1", storage)

        recorder.record_llm_call(messages=[{"role": "user", "content": "Test"}])
        recorder.record_tool_call("my_tool", {"arg": 1})
        recorder.record_tool_result("my_tool", {"result": "ok"})
        recorder.save()

        # Load and verify
        loaded = storage.load("workflow_1")
        assert len(loaded) == 3
        assert loaded[0].step_type == "llm_call"
        assert loaded[1].step_type == "tool_call"
        assert loaded[2].step_type == "tool_result"

    def test_timeline(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        recorder.record_llm_call(messages=[])
        recorder.record_tool_call("get_weather", {"city": "NYC"})
        recorder.record_tool_result("get_weather", {"temp": 72})
        recorder.record_agent_response("It's 72°F in NYC")

        timeline = recorder.timeline()

        assert len(timeline) == 4
        assert "[Step 0: llm_call]" in timeline[0]
        assert "get_weather" in timeline[1]
        assert "get_weather" in timeline[2]
        assert "[Step 3: agent_response]" in timeline[3]

    def test_clear(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)

        recorder.record_llm_call(messages=[])
        recorder.record_tool_call("tool", {})

        assert len(recorder.steps) == 2

        recorder.clear()

        assert len(recorder.steps) == 0
        # Counter should reset
        step = recorder.record_llm_call(messages=[])
        assert step.step_id == 0


class TestMemoryStorage:
    """Tests for MemoryStorage backend."""

    def test_save_and_load(self):
        storage = MemoryStorage()
        steps = [
            WorkflowStep(0, "llm_call", datetime.now(), {}),
            WorkflowStep(1, "tool_call", datetime.now(), {"name": "tool1"}),
        ]

        storage.save("test_id", steps)
        loaded = storage.load("test_id")

        assert len(loaded) == 2
        assert loaded[0].step_id == 0
        assert loaded[1].data["name"] == "tool1"

    def test_exists(self):
        storage = MemoryStorage()

        assert storage.exists("not_found") is False

        storage.save("exists", [])
        assert storage.exists("exists") is True

    def test_delete(self):
        storage = MemoryStorage()
        storage.save("to_delete", [])

        assert storage.exists("to_delete") is True

        result = storage.delete("to_delete")
        assert result is True
        assert storage.exists("to_delete") is False

        # Delete non-existent
        result = storage.delete("not_found")
        assert result is False

    def test_list_recordings(self):
        storage = MemoryStorage()

        assert storage.list_recordings() == []

        storage.save("rec1", [])
        storage.save("rec2", [])
        storage.save("rec3", [])

        recordings = storage.list_recordings()
        assert set(recordings) == {"rec1", "rec2", "rec3"}


class TestSQLiteStorage:
    """Tests for SQLiteStorage backend."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)

            steps = [
                WorkflowStep(0, "llm_call", datetime.now(), {"msg": "test"}),
                WorkflowStep(1, "tool_call", datetime.now(), {"name": "tool1"}),
            ]

            storage.save("test_id", steps)
            loaded = storage.load("test_id")

            assert len(loaded) == 2
            assert loaded[0].data["msg"] == "test"

            storage.close()

    def test_exists_and_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)

            storage.save("rec1", [])

            assert storage.exists("rec1") is True
            assert storage.exists("not_found") is False

            storage.delete("rec1")
            assert storage.exists("rec1") is False

            storage.close()


class TestReplay:
    """Tests for replay() function."""

    def test_basic_replay(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_1", storage)

        recorder.record_llm_call(messages=[{"role": "user", "content": "Test"}])
        recorder.record_tool_call("get_data", {"id": 1})
        recorder.record_tool_result("get_data", {"value": 42})
        recorder.record_agent_response("The value is 42")
        recorder.save()

        result = replay("workflow_1", storage=storage)

        assert result.record_id == "workflow_1"
        assert len(result.steps) == 4
        assert result.output == "The value is 42"

    def test_replay_from_step(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_1", storage)

        recorder.record_llm_call(messages=[])
        recorder.record_tool_call("tool1", {})
        recorder.record_tool_result("tool1", {})
        recorder.record_tool_call("tool2", {})
        recorder.record_tool_result("tool2", {"important": True})
        recorder.save()

        # Start from step 3
        result = replay("workflow_1", from_step=3, storage=storage)

        assert result.from_step == 3
        assert len(result.steps) == 2  # Steps 3 and 4
        assert result.steps[0].step_id == 3
        assert result.steps[0].data["name"] == "tool2"

    def test_replay_with_injection(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_1", storage)

        recorder.record_llm_call(messages=[])
        recorder.record_tool_call("fetch_data", {"id": 1})
        recorder.record_tool_result("fetch_data", {"original": True})
        recorder.record_agent_response("Original response")
        recorder.save()

        # Inject fake data
        result = replay(
            "workflow_1",
            inject={"fetch_data": {"injected": True, "value": 999}},
            storage=storage,
        )

        # Find the tool result step
        tool_result = next(s for s in result.steps if s.step_type == "tool_result")

        assert tool_result.data["result"]["injected"] is True
        assert tool_result.data["result"]["value"] == 999
        assert tool_result.data.get("injected") is True
        assert tool_result.step_id in result.injected_steps

    def test_replay_not_found(self):
        storage = MemoryStorage()

        with pytest.raises(ValueError, match="Recording not found"):
            replay("nonexistent", storage=storage)

    def test_replay_timeline_shows_injections(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("workflow_1", storage)

        recorder.record_tool_call("my_tool", {})
        recorder.record_tool_result("my_tool", {"original": True})
        recorder.save()

        result = replay(
            "workflow_1",
            inject={"my_tool": {"injected": True}},
            storage=storage,
        )

        timeline = result.timeline()

        # Injected step should be marked
        assert any("[INJECTED]" in line for line in timeline)


class TestReplayResult:
    """Tests for ReplayResult class."""

    def test_output_from_agent_response(self):
        steps = [
            WorkflowStep(0, "llm_call", datetime.now(), {}),
            WorkflowStep(1, "tool_call", datetime.now(), {"name": "tool1"}),
            WorkflowStep(2, "tool_result", datetime.now(), {"name": "tool1", "result": 42}),
            WorkflowStep(3, "agent_response", datetime.now(), {"content": "Final answer"}),
        ]

        result = ReplayResult("test", steps=steps)

        assert result.output == "Final answer"

    def test_output_falls_back_to_tool_result(self):
        steps = [
            WorkflowStep(0, "tool_call", datetime.now(), {"name": "tool1"}),
            WorkflowStep(
                1, "tool_result", datetime.now(), {"name": "tool1", "result": {"data": 123}}
            ),
        ]

        result = ReplayResult("test", steps=steps)

        assert result.output == {"data": 123}

    def test_tool_calls(self):
        steps = [
            WorkflowStep(0, "llm_call", datetime.now(), {}),
            WorkflowStep(1, "tool_call", datetime.now(), {"name": "tool1", "args": {"a": 1}}),
            WorkflowStep(2, "tool_result", datetime.now(), {"name": "tool1", "result": {}}),
            WorkflowStep(3, "tool_call", datetime.now(), {"name": "tool2", "args": {"b": 2}}),
        ]

        result = ReplayResult("test", steps=steps)

        tool_calls = result.tool_calls
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool1"
        assert tool_calls[1]["name"] == "tool2"

    def test_tool_results(self):
        steps = [
            WorkflowStep(0, "tool_call", datetime.now(), {"name": "tool1"}),
            WorkflowStep(1, "tool_result", datetime.now(), {"name": "tool1", "result": "r1"}),
            WorkflowStep(2, "tool_call", datetime.now(), {"name": "tool2"}),
            WorkflowStep(3, "tool_result", datetime.now(), {"name": "tool2", "result": "r2"}),
        ]

        result = ReplayResult("test", steps=steps)

        tool_results = result.tool_results
        assert len(tool_results) == 2
        assert tool_results[0]["result"] == "r1"
        assert tool_results[1]["result"] == "r2"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_recording(self):
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)
        recorder.record_llm_call(messages=[])
        recorder.save()

        steps = get_recording("test", storage=storage)
        assert len(steps) == 1

    def test_list_recordings(self):
        storage = MemoryStorage()
        storage.save("rec1", [])
        storage.save("rec2", [])

        recordings = list_recordings(storage=storage)
        assert "rec1" in recordings
        assert "rec2" in recordings

    def test_delete_recording(self):
        storage = MemoryStorage()
        storage.save("to_delete", [])

        result = delete_recording("to_delete", storage=storage)
        assert result is True

        result = delete_recording("not_found", storage=storage)
        assert result is False
