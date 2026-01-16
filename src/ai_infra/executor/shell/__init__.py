"""Shell environment management for Executor.

This package provides:
- Shell state capture (env vars, aliases, functions, options)
- Snapshot persistence
- Environment restoration

Phase 16 of EXECUTOR_5.md
"""

from __future__ import annotations

from ai_infra.executor.shell.persistence import (
    COMPRESSED_EXTENSION,
    COMPRESSION_THRESHOLD,
    DEFAULT_SNAPSHOT_DIR,
    UNCOMPRESSED_EXTENSION,
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
from ai_infra.executor.shell.restoration import (
    CAUTION_ENV_VARS,
    PROTECTED_ENV_VARS,
    RestorationResult,
    SnapshotDiff,
    diff_snapshots,
    generate_restore_script,
    is_caution_env_var,
    is_protected_env_var,
    restore_aliases,
    restore_env_vars,
    restore_functions,
    restore_shell_state,
    restore_working_dir,
    validate_env_var_value,
)
from ai_infra.executor.shell.snapshot import (
    ShellSnapshot,
    ShellType,
    capture_aliases,
    capture_env_vars,
    capture_functions,
    capture_shell_options,
    capture_shell_state,
    detect_shell,
)

__all__ = [
    # Data models
    "ShellSnapshot",
    "ShellType",
    "RestorationResult",
    "SnapshotDiff",
    # Capture functions
    "capture_aliases",
    "capture_env_vars",
    "capture_functions",
    "capture_shell_options",
    "capture_shell_state",
    # Detection
    "detect_shell",
    # Persistence (Phase 16.2)
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "get_latest_snapshot",
    "delete_snapshot",
    "cleanup_old_snapshots",
    "generate_snapshot_filename",
    "parse_snapshot_filename",
    "get_snapshot_dir",
    "ensure_snapshot_dir",
    # Restoration (Phase 16.3)
    "restore_env_vars",
    "restore_aliases",
    "restore_functions",
    "restore_working_dir",
    "restore_shell_state",
    "generate_restore_script",
    "diff_snapshots",
    # Safety checks
    "is_protected_env_var",
    "is_caution_env_var",
    "validate_env_var_value",
    # Constants
    "DEFAULT_SNAPSHOT_DIR",
    "COMPRESSION_THRESHOLD",
    "COMPRESSED_EXTENSION",
    "UNCOMPRESSED_EXTENSION",
    "PROTECTED_ENV_VARS",
    "CAUTION_ENV_VARS",
]
