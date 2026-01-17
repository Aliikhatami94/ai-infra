"""Unit tests for shell snapshot persistence (Phase 16.2)."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_infra.executor.shell.persistence import (
    COMPRESSION_THRESHOLD,
    DEFAULT_SNAPSHOT_DIR,
    _compress_json,
    _decompress_json,
    _should_compress,
    cleanup_old_snapshots,
    delete_snapshot,
    ensure_snapshot_dir,
    generate_snapshot_filename,
    get_latest_snapshot,
    get_snapshot_dir,
    list_snapshots,
    load_snapshot,
    parse_snapshot_filename,
    save_snapshot,
)
from ai_infra.executor.shell.snapshot import ShellSnapshot, ShellType

# =============================================================================
# Filename Generation Tests (Phase 16.2.4)
# =============================================================================


class TestGenerateSnapshotFilename:
    """Tests for snapshot filename generation (Phase 16.2.4)."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_snapshot_filename(ShellType.BASH)
        assert filename.startswith("snapshot-bash-")
        assert filename.endswith(".json")

    def test_filename_with_shell_type(self):
        """Test filename includes shell type."""
        assert "bash" in generate_snapshot_filename(ShellType.BASH)
        assert "zsh" in generate_snapshot_filename(ShellType.ZSH)
        assert "fish" in generate_snapshot_filename(ShellType.FISH)

    def test_filename_with_timestamp(self):
        """Test filename includes provided timestamp."""
        ts = datetime(2026, 1, 15, 10, 30, 45)
        filename = generate_snapshot_filename(ShellType.BASH, ts)
        assert "20260115-103045" in filename

    def test_filename_with_suffix(self):
        """Test filename includes suffix."""
        filename = generate_snapshot_filename(ShellType.BASH, suffix="-pre")
        assert "-pre.json" in filename

    def test_filename_uniqueness(self):
        """Test filenames are unique."""
        filenames = {generate_snapshot_filename(ShellType.BASH) for _ in range(100)}
        assert len(filenames) == 100


class TestParseSnapshotFilename:
    """Tests for parsing snapshot filenames."""

    def test_parse_valid_filename(self):
        """Test parsing a valid filename."""
        result = parse_snapshot_filename("snapshot-bash-20260115-103045-abc12345.json")
        assert result["valid"] is True
        assert result["shell_type"] == ShellType.BASH
        assert result["timestamp"] == datetime(2026, 1, 15, 10, 30, 45)
        assert result["random_id"] == "abc12345"

    def test_parse_compressed_filename(self):
        """Test parsing compressed filename."""
        result = parse_snapshot_filename("snapshot-zsh-20260115-103045-abc12345.json.gz")
        assert result["valid"] is True
        assert result["shell_type"] == ShellType.ZSH

    def test_parse_invalid_filename(self):
        """Test parsing invalid filename."""
        result = parse_snapshot_filename("invalid-file.json")
        assert result["valid"] is False

    def test_parse_with_suffix(self):
        """Test parsing filename with suffix."""
        result = parse_snapshot_filename("snapshot-bash-20260115-103045-abc12345-pre.json")
        assert result["valid"] is True
        assert result["suffix"] == "pre"


# =============================================================================
# Snapshot Directory Tests (Phase 16.2.3)
# =============================================================================


class TestSnapshotDirectory:
    """Tests for snapshot directory management (Phase 16.2.3)."""

    def test_get_snapshot_dir_default(self):
        """Test default snapshot directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
                result = get_snapshot_dir()
                assert str(result).endswith(DEFAULT_SNAPSHOT_DIR)

    def test_get_snapshot_dir_with_workspace(self):
        """Test snapshot directory with workspace."""
        result = get_snapshot_dir("/home/user/project")
        assert str(result) == f"/home/user/project/{DEFAULT_SNAPSHOT_DIR}"

    def test_ensure_snapshot_dir_creates(self):
        """Test ensure_snapshot_dir creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_snapshot_dir(tmpdir)
            assert result.exists()
            assert result.is_dir()

    def test_ensure_snapshot_dir_idempotent(self):
        """Test ensure_snapshot_dir is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = ensure_snapshot_dir(tmpdir)
            result2 = ensure_snapshot_dir(tmpdir)
            assert result1 == result2
            assert result1.exists()


# =============================================================================
# Compression Tests (Phase 16.2.5)
# =============================================================================


class TestCompression:
    """Tests for snapshot compression (Phase 16.2.5)."""

    def test_should_compress_small_data(self):
        """Test small data is not compressed."""
        data = {"env_vars": {"A": "1"}, "functions": {}}
        assert _should_compress(data) is False

    def test_should_compress_large_functions(self):
        """Test large functions trigger compression."""
        # Create data larger than threshold
        large_func = "x" * (COMPRESSION_THRESHOLD + 1000)
        data = {"functions": {"big_func": large_func}}
        assert _should_compress(data) is True

    def test_compress_decompress_roundtrip(self):
        """Test compression/decompression roundtrip."""
        data = {
            "env_vars": {"PATH": "/usr/bin"},
            "functions": {"test": "echo hello"},
        }
        compressed = _compress_json(data)
        decompressed = _decompress_json(compressed)
        assert decompressed == data

    def test_compressed_is_smaller(self):
        """Test compressed data is smaller for large data."""
        # Create compressible data (repeated patterns)
        large_data = {"functions": {"f": "echo " * 10000}}
        json_size = len(json.dumps(large_data))
        compressed_size = len(_compress_json(large_data))
        assert compressed_size < json_size


# =============================================================================
# Save Snapshot Tests (Phase 16.2.1)
# =============================================================================


class TestSaveSnapshot:
    """Tests for saving snapshots (Phase 16.2.1)."""

    def test_save_snapshot_basic(self):
        """Test basic snapshot save."""
        snapshot = ShellSnapshot(
            env_vars={"TEST": "value"},
            aliases={"ll": "ls -l"},
            shell_type=ShellType.BASH,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, workspace_dir=tmpdir)

            assert path.exists()
            assert path.name.startswith("snapshot-bash-")
            assert path.name.endswith(".json")

    def test_save_snapshot_explicit_path(self):
        """Test save to explicit path."""
        snapshot = ShellSnapshot(shell_type=ShellType.ZSH)

        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_path = Path(tmpdir) / "my-snapshot.json"
            result = save_snapshot(snapshot, path=explicit_path)

            assert result.exists()
            assert result.name == "my-snapshot.json"

    def test_save_snapshot_compressed(self):
        """Test forced compression."""
        snapshot = ShellSnapshot(shell_type=ShellType.BASH)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, workspace_dir=tmpdir, compress=True)

            assert path.exists()
            assert str(path).endswith(".gz")

    def test_save_snapshot_creates_directory(self):
        """Test save creates parent directories."""
        snapshot = ShellSnapshot(shell_type=ShellType.BASH)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "deep" / "nested" / "snapshot.json"
            path = save_snapshot(snapshot, path=nested_path)

            assert path.exists()
            assert path.parent.exists()

    def test_save_snapshot_content(self):
        """Test saved content is correct."""
        snapshot = ShellSnapshot(
            env_vars={"KEY": "value"},
            aliases={"a": "alias"},
            functions={"f": "function"},
            working_dir="/home/user",
            shell_type=ShellType.BASH,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, workspace_dir=tmpdir, compress=False)

            data = json.loads(path.read_text())
            assert data["env_vars"] == {"KEY": "value"}
            assert data["aliases"] == {"a": "alias"}
            assert data["functions"] == {"f": "function"}
            assert data["working_dir"] == "/home/user"
            assert data["shell_type"] == "bash"


# =============================================================================
# Load Snapshot Tests (Phase 16.2.2)
# =============================================================================


class TestLoadSnapshot:
    """Tests for loading snapshots (Phase 16.2.2)."""

    def test_load_snapshot_basic(self):
        """Test basic snapshot load."""
        snapshot = ShellSnapshot(
            env_vars={"TEST": "value"},
            aliases={"ll": "ls -l"},
            shell_type=ShellType.BASH,
            working_dir="/home/user",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, workspace_dir=tmpdir)
            loaded = load_snapshot(path)

            assert loaded.env_vars == snapshot.env_vars
            assert loaded.aliases == snapshot.aliases
            assert loaded.shell_type == snapshot.shell_type
            assert loaded.working_dir == snapshot.working_dir

    def test_load_snapshot_compressed(self):
        """Test loading compressed snapshot."""
        snapshot = ShellSnapshot(
            env_vars={"COMPRESSED": "yes"},
            shell_type=ShellType.ZSH,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, workspace_dir=tmpdir, compress=True)
            loaded = load_snapshot(path)

            assert loaded.env_vars == {"COMPRESSED": "yes"}
            assert loaded.shell_type == ShellType.ZSH

    def test_load_snapshot_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_snapshot("/nonexistent/path/snapshot.json")

    def test_load_snapshot_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text("not valid json {{{")

            with pytest.raises(ValueError, match="Invalid snapshot JSON"):
                load_snapshot(bad_file)

    def test_load_snapshot_invalid_compressed(self):
        """Test loading invalid compressed file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json.gz"
            bad_file.write_bytes(b"not gzip data")

            with pytest.raises(ValueError, match="Invalid compressed snapshot"):
                load_snapshot(bad_file)


# =============================================================================
# List Snapshots Tests
# =============================================================================


class TestListSnapshots:
    """Tests for listing snapshots."""

    def test_list_empty_directory(self):
        """Test listing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_snapshots(tmpdir)
            assert result == []

    def test_list_snapshots_basic(self):
        """Test listing snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some snapshots
            for _ in range(3):
                snapshot = ShellSnapshot(shell_type=ShellType.BASH)
                save_snapshot(snapshot, workspace_dir=tmpdir)

            result = list_snapshots(tmpdir)
            assert len(result) == 3

    def test_list_snapshots_filter_by_shell(self):
        """Test filtering by shell type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create bash and zsh snapshots
            save_snapshot(ShellSnapshot(shell_type=ShellType.BASH), workspace_dir=tmpdir)
            save_snapshot(ShellSnapshot(shell_type=ShellType.BASH), workspace_dir=tmpdir)
            save_snapshot(ShellSnapshot(shell_type=ShellType.ZSH), workspace_dir=tmpdir)

            bash_only = list_snapshots(tmpdir, shell_type=ShellType.BASH)
            assert len(bash_only) == 2

            zsh_only = list_snapshots(tmpdir, shell_type=ShellType.ZSH)
            assert len(zsh_only) == 1

    def test_list_snapshots_limit(self):
        """Test limiting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(5):
                snapshot = ShellSnapshot(shell_type=ShellType.BASH)
                save_snapshot(snapshot, workspace_dir=tmpdir)

            result = list_snapshots(tmpdir, limit=2)
            assert len(result) == 2

    def test_list_snapshots_sorted_by_time(self):
        """Test results are sorted newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts1 = datetime(2026, 1, 1, 10, 0, 0)
            ts2 = datetime(2026, 1, 2, 10, 0, 0)
            ts3 = datetime(2026, 1, 3, 10, 0, 0)

            save_snapshot(
                ShellSnapshot(shell_type=ShellType.BASH, captured_at=ts1),
                workspace_dir=tmpdir,
            )
            save_snapshot(
                ShellSnapshot(shell_type=ShellType.BASH, captured_at=ts3),
                workspace_dir=tmpdir,
            )
            save_snapshot(
                ShellSnapshot(shell_type=ShellType.BASH, captured_at=ts2),
                workspace_dir=tmpdir,
            )

            result = list_snapshots(tmpdir)
            timestamps = [r["timestamp"] for r in result]
            assert timestamps == sorted(timestamps, reverse=True)


# =============================================================================
# Get Latest Snapshot Tests
# =============================================================================


class TestGetLatestSnapshot:
    """Tests for getting latest snapshot."""

    def test_get_latest_none_exists(self):
        """Test returns None when no snapshots exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_latest_snapshot(tmpdir)
            assert result is None

    def test_get_latest_returns_newest(self):
        """Test returns the newest snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_old = datetime(2026, 1, 1, 10, 0, 0)
            ts_new = datetime(2026, 1, 15, 10, 0, 0)

            save_snapshot(
                ShellSnapshot(
                    env_vars={"OLD": "yes"},
                    shell_type=ShellType.BASH,
                    captured_at=ts_old,
                ),
                workspace_dir=tmpdir,
            )
            save_snapshot(
                ShellSnapshot(
                    env_vars={"NEW": "yes"},
                    shell_type=ShellType.BASH,
                    captured_at=ts_new,
                ),
                workspace_dir=tmpdir,
            )

            latest = get_latest_snapshot(tmpdir)
            assert latest is not None
            assert "NEW" in latest.env_vars


# =============================================================================
# Delete Snapshot Tests
# =============================================================================


class TestDeleteSnapshot:
    """Tests for deleting snapshots."""

    def test_delete_existing(self):
        """Test deleting existing snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = ShellSnapshot(shell_type=ShellType.BASH)
            path = save_snapshot(snapshot, workspace_dir=tmpdir)

            assert path.exists()
            result = delete_snapshot(path)
            assert result is True
            assert not path.exists()

    def test_delete_nonexistent(self):
        """Test deleting non-existent file returns False."""
        result = delete_snapshot("/nonexistent/file.json")
        assert result is False


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanupOldSnapshots:
    """Tests for cleaning up old snapshots."""

    def test_cleanup_keeps_recent(self):
        """Test cleanup keeps the specified number of recent snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 snapshots
            for i in range(5):
                ts = datetime(2026, 1, i + 1, 10, 0, 0)
                save_snapshot(
                    ShellSnapshot(shell_type=ShellType.BASH, captured_at=ts),
                    workspace_dir=tmpdir,
                )

            # Keep only 2
            deleted = cleanup_old_snapshots(tmpdir, keep_count=2)
            assert deleted == 3

            remaining = list_snapshots(tmpdir)
            assert len(remaining) == 2

    def test_cleanup_by_shell_type(self):
        """Test cleanup only affects specified shell type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create bash and zsh snapshots
            for _ in range(3):
                save_snapshot(ShellSnapshot(shell_type=ShellType.BASH), workspace_dir=tmpdir)
            for _ in range(3):
                save_snapshot(ShellSnapshot(shell_type=ShellType.ZSH), workspace_dir=tmpdir)

            # Cleanup only bash, keep 1
            deleted = cleanup_old_snapshots(tmpdir, keep_count=1, shell_type=ShellType.BASH)
            assert deleted == 2

            # Bash should have 1, zsh should still have 3
            bash_remaining = list_snapshots(tmpdir, shell_type=ShellType.BASH)
            zsh_remaining = list_snapshots(tmpdir, shell_type=ShellType.ZSH)
            assert len(bash_remaining) == 1
            assert len(zsh_remaining) == 3

    def test_cleanup_nothing_to_delete(self):
        """Test cleanup when nothing to delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 snapshots, keep 5
            for _ in range(2):
                save_snapshot(ShellSnapshot(shell_type=ShellType.BASH), workspace_dir=tmpdir)

            deleted = cleanup_old_snapshots(tmpdir, keep_count=5)
            assert deleted == 0
