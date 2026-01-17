"""Shell snapshot persistence (Phase 16.2).

This module provides functionality to save and load shell snapshots
to/from disk with optional compression for large function definitions.
"""

from __future__ import annotations

import gzip
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

from .snapshot import ShellSnapshot, ShellType

logger = get_logger("executor.shell")


# =============================================================================
# Constants
# =============================================================================


# Default directory for shell snapshots (relative to workspace)
DEFAULT_SNAPSHOT_DIR = ".executor/shell-snapshots"

# Compression threshold in bytes (compress if larger)
COMPRESSION_THRESHOLD = 10_000  # 10KB

# File extension for compressed snapshots
COMPRESSED_EXTENSION = ".json.gz"
UNCOMPRESSED_EXTENSION = ".json"


# =============================================================================
# Snapshot Naming (Phase 16.2.4)
# =============================================================================


def generate_snapshot_filename(
    shell_type: ShellType,
    timestamp: datetime | None = None,
    suffix: str = "",
) -> str:
    """Generate a unique snapshot filename (Phase 16.2.4).

    Format: snapshot-{shell}-{timestamp}-{random}.json

    Args:
        shell_type: Type of shell (bash, zsh, fish).
        timestamp: Timestamp for the snapshot (default: now).
        suffix: Optional suffix before extension (e.g., "-pre", "-post").

    Returns:
        Unique filename string.
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)  # 8 hex chars
    shell_name = shell_type.value

    if suffix:
        return f"snapshot-{shell_name}-{ts_str}-{random_suffix}{suffix}.json"
    return f"snapshot-{shell_name}-{ts_str}-{random_suffix}.json"


def parse_snapshot_filename(filename: str) -> dict[str, Any]:
    """Parse components from a snapshot filename.

    Args:
        filename: Snapshot filename.

    Returns:
        Dictionary with shell_type, timestamp, random_id, suffix.
    """
    # Remove extension
    base = filename
    for ext in (COMPRESSED_EXTENSION, UNCOMPRESSED_EXTENSION):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    parts = base.split("-")
    if len(parts) < 4 or parts[0] != "snapshot":
        return {"valid": False, "filename": filename}

    try:
        shell_type = ShellType(parts[1])
        # Timestamp is parts[2] and parts[3] combined
        ts_str = f"{parts[2]}-{parts[3]}"
        timestamp = datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
        random_id = parts[4] if len(parts) > 4 else ""
        suffix = "-".join(parts[5:]) if len(parts) > 5 else ""

        return {
            "valid": True,
            "filename": filename,
            "shell_type": shell_type,
            "timestamp": timestamp,
            "random_id": random_id,
            "suffix": suffix,
        }
    except (ValueError, IndexError):
        return {"valid": False, "filename": filename}


# =============================================================================
# Snapshot Directory Management (Phase 16.2.3)
# =============================================================================


def get_snapshot_dir(workspace_dir: str | Path | None = None) -> Path:
    """Get the snapshot directory path (Phase 16.2.3).

    Args:
        workspace_dir: Workspace directory (default: cwd).

    Returns:
        Path to the snapshots directory.
    """
    if workspace_dir:
        base = Path(workspace_dir)
    else:
        base = Path.cwd()

    return base / DEFAULT_SNAPSHOT_DIR


def ensure_snapshot_dir(workspace_dir: str | Path | None = None) -> Path:
    """Ensure the snapshot directory exists.

    Args:
        workspace_dir: Workspace directory (default: cwd).

    Returns:
        Path to the created/existing snapshots directory.
    """
    snapshot_dir = get_snapshot_dir(workspace_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


# =============================================================================
# Compression (Phase 16.2.5)
# =============================================================================


def _should_compress(data: dict[str, Any]) -> bool:
    """Determine if snapshot should be compressed (Phase 16.2.5).

    Compression is applied when the JSON representation exceeds
    the threshold, typically due to large function definitions.

    Args:
        data: Snapshot data dictionary.

    Returns:
        True if compression should be used.
    """
    # Estimate size by checking functions (usually the largest part)
    functions_size = sum(len(v) for v in data.get("functions", {}).values())
    if functions_size > COMPRESSION_THRESHOLD:
        return True

    # Check total size
    try:
        json_str = json.dumps(data)
        return len(json_str) > COMPRESSION_THRESHOLD
    except (TypeError, ValueError):
        return False


def _compress_json(data: dict[str, Any]) -> bytes:
    """Compress JSON data using gzip.

    Args:
        data: Data to compress.

    Returns:
        Gzip-compressed bytes.
    """
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    return gzip.compress(json_bytes, compresslevel=6)


def _decompress_json(compressed: bytes) -> dict[str, Any]:
    """Decompress gzip JSON data.

    Args:
        compressed: Gzip-compressed bytes.

    Returns:
        Decompressed JSON data.
    """
    json_bytes = gzip.decompress(compressed)
    return json.loads(json_bytes.decode("utf-8"))


# =============================================================================
# Save Snapshot (Phase 16.2.1)
# =============================================================================


def save_snapshot(
    snapshot: ShellSnapshot,
    path: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    compress: bool | None = None,
) -> Path:
    """Save a shell snapshot to disk (Phase 16.2.1).

    Args:
        snapshot: ShellSnapshot to save.
        path: Explicit file path (if None, generates filename).
        workspace_dir: Workspace directory for default location.
        compress: Force compression (None = auto-detect).

    Returns:
        Path to the saved snapshot file.

    Raises:
        IOError: If the file cannot be written.
    """
    data = snapshot.to_dict()

    # Determine if we should compress
    should_compress = compress if compress is not None else _should_compress(data)

    # Determine file path
    if path:
        file_path = Path(path)
        # Ensure correct extension
        if should_compress and not str(file_path).endswith(COMPRESSED_EXTENSION):
            file_path = Path(str(file_path) + ".gz")
    else:
        # Generate filename in snapshot directory
        snapshot_dir = ensure_snapshot_dir(workspace_dir)
        filename = generate_snapshot_filename(snapshot.shell_type, snapshot.captured_at)
        if should_compress:
            filename = filename.replace(UNCOMPRESSED_EXTENSION, COMPRESSED_EXTENSION)
        file_path = snapshot_dir / filename

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    if should_compress:
        compressed_data = _compress_json(data)
        file_path.write_bytes(compressed_data)
        logger.debug(f"Saved compressed snapshot to {file_path} ({len(compressed_data)} bytes)")
    else:
        json_str = json.dumps(data, indent=2)
        file_path.write_text(json_str, encoding="utf-8")
        logger.debug(f"Saved snapshot to {file_path} ({len(json_str)} bytes)")

    logger.info(f"Snapshot saved: {file_path.name}")
    return file_path


# =============================================================================
# Load Snapshot (Phase 16.2.2)
# =============================================================================


def load_snapshot(path: str | Path) -> ShellSnapshot:
    """Load a shell snapshot from disk (Phase 16.2.2).

    Args:
        path: Path to the snapshot file.

    Returns:
        Loaded ShellSnapshot.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a valid snapshot.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {file_path}")

    # Determine if compressed
    is_compressed = str(file_path).endswith(COMPRESSED_EXTENSION) or str(file_path).endswith(".gz")

    try:
        if is_compressed:
            compressed_data = file_path.read_bytes()
            data = _decompress_json(compressed_data)
            logger.debug(f"Loaded compressed snapshot from {file_path}")
        else:
            json_str = file_path.read_text(encoding="utf-8")
            data = json.loads(json_str)
            logger.debug(f"Loaded snapshot from {file_path}")

        snapshot = ShellSnapshot.from_dict(data)
        logger.info(f"Snapshot loaded: {snapshot.summary()}")
        return snapshot

    except gzip.BadGzipFile as e:
        raise ValueError(f"Invalid compressed snapshot: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid snapshot JSON: {e}") from e
    except KeyError as e:
        raise ValueError(f"Missing required field in snapshot: {e}") from e


# =============================================================================
# List Snapshots
# =============================================================================


def list_snapshots(
    workspace_dir: str | Path | None = None,
    shell_type: ShellType | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List available snapshots in the workspace.

    Args:
        workspace_dir: Workspace directory (default: cwd).
        shell_type: Filter by shell type.
        limit: Maximum number of snapshots to return (newest first).

    Returns:
        List of snapshot metadata dictionaries.
    """
    snapshot_dir = get_snapshot_dir(workspace_dir)

    if not snapshot_dir.exists():
        return []

    snapshots = []
    for file_path in snapshot_dir.iterdir():
        if not file_path.is_file():
            continue

        # Must be a snapshot file
        if not (
            file_path.name.endswith(UNCOMPRESSED_EXTENSION)
            or file_path.name.endswith(COMPRESSED_EXTENSION)
        ):
            continue

        if not file_path.name.startswith("snapshot-"):
            continue

        # Parse filename
        info = parse_snapshot_filename(file_path.name)
        if not info.get("valid"):
            continue

        # Filter by shell type
        if shell_type and info.get("shell_type") != shell_type:
            continue

        # Add file info
        info["path"] = str(file_path)
        info["size"] = file_path.stat().st_size
        info["compressed"] = str(file_path).endswith(COMPRESSED_EXTENSION)

        snapshots.append(info)

    # Sort by timestamp (newest first)
    snapshots.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)

    if limit:
        snapshots = snapshots[:limit]

    return snapshots


def get_latest_snapshot(
    workspace_dir: str | Path | None = None,
    shell_type: ShellType | None = None,
) -> ShellSnapshot | None:
    """Get the most recent snapshot.

    Args:
        workspace_dir: Workspace directory (default: cwd).
        shell_type: Filter by shell type.

    Returns:
        Most recent ShellSnapshot or None if no snapshots exist.
    """
    snapshots = list_snapshots(workspace_dir, shell_type, limit=1)
    if not snapshots:
        return None

    return load_snapshot(snapshots[0]["path"])


def delete_snapshot(path: str | Path) -> bool:
    """Delete a snapshot file.

    Args:
        path: Path to the snapshot file.

    Returns:
        True if deleted, False if file didn't exist.
    """
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted snapshot: {file_path.name}")
        return True
    return False


def cleanup_old_snapshots(
    workspace_dir: str | Path | None = None,
    keep_count: int = 10,
    shell_type: ShellType | None = None,
) -> int:
    """Delete old snapshots, keeping the most recent ones.

    Args:
        workspace_dir: Workspace directory (default: cwd).
        keep_count: Number of snapshots to keep per shell type.
        shell_type: Only cleanup specific shell type (None = all).

    Returns:
        Number of snapshots deleted.
    """
    deleted = 0

    if shell_type:
        shell_types = [shell_type]
    else:
        shell_types = [ShellType.BASH, ShellType.ZSH, ShellType.FISH]

    for st in shell_types:
        snapshots = list_snapshots(workspace_dir, st)
        # Keep the newest, delete the rest
        to_delete = snapshots[keep_count:]
        for snap_info in to_delete:
            if delete_snapshot(snap_info["path"]):
                deleted += 1

    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old snapshot(s)")

    return deleted
