"""Outcome extraction from agent responses.

This module provides utilities to extract structured TaskOutcome data
from agent execution responses. It parses agent output to identify:
- Files created, modified, or deleted
- Key decisions made by the agent
- A summary of what was accomplished

Part of Phase 5.8.3: Execution Memory Architecture.

Example:
    ```python
    from pathlib import Path
    from ai_infra.executor import extract_outcome

    # Extract outcome from agent response (without LLM)
    result = await extract_outcome(
        agent_response="I will create src/utils.py with the formatName function...",
        workspace_root=Path("/path/to/project"),
        task_id="1.1",
        task_title="Create utils module",
    )

    print(result.files)  # {Path("src/utils.py"): FileAction.CREATED}
    print(result.summary)  # "Create utils module"
    print(result.key_decisions)  # ["create src/utils.py with the formatName function"]

    # With LLM for better extraction
    from ai_infra import LLM
    result = await extract_outcome(
        agent_response="...",
        workspace_root=Path("/path/to/project"),
        task_id="1.1",
        task_title="Create utils module",
        llm=LLM(),
    )
    ```
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.run_memory import FileAction

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extracting outcome data from agent response.

    Attributes:
        files: Map of file paths to the action performed (created/modified/deleted).
        key_decisions: List of key decisions extracted from agent reasoning.
        summary: One-line summary of what was accomplished.
        raw_file_refs: Raw file references found (before path resolution).
    """

    files: dict[Path, FileAction] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)
    summary: str = ""
    raw_file_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "files": {str(p): a.value for p, a in self.files.items()},
            "key_decisions": self.key_decisions,
            "summary": self.summary,
            "raw_file_refs": self.raw_file_refs,
        }


# =============================================================================
# File Operation Extraction
# =============================================================================

# Patterns for detecting file operations in agent responses
FILE_OPERATION_PATTERNS = [
    # Tool call patterns (write_file, edit_file, etc.)
    (r'write_file\s*\(\s*["\']([^"\']+)["\']', FileAction.CREATED),
    (r'edit_file\s*\(\s*["\']([^"\']+)["\']', FileAction.MODIFIED),
    (r'create_file\s*\(\s*["\']([^"\']+)["\']', FileAction.CREATED),
    (r'delete_file\s*\(\s*["\']([^"\']+)["\']', FileAction.DELETED),
    (r'remove_file\s*\(\s*["\']([^"\']+)["\']', FileAction.DELETED),
    # JSON-style tool calls
    (r'"tool":\s*"write_file"[^}]*"path":\s*"([^"]+)"', FileAction.CREATED),
    (r'"tool":\s*"edit_file"[^}]*"path":\s*"([^"]+)"', FileAction.MODIFIED),
    (r'"name":\s*"write_file"[^}]*"path":\s*"([^"]+)"', FileAction.CREATED),
    (r'"name":\s*"edit_file"[^}]*"path":\s*"([^"]+)"', FileAction.MODIFIED),
    # Natural language patterns
    (
        r'(?:created?|wrote?|written)\s+(?:the\s+)?(?:file\s+)?[`"\']([^`"\']+\.(?:py|js|ts|rs|go|java|md|json|yaml|yml|toml))[`"\']',
        FileAction.CREATED,
    ),
    (
        r'(?:modified|updated|changed|edited)\s+(?:the\s+)?(?:file\s+)?[`"\']([^`"\']+\.(?:py|js|ts|rs|go|java|md|json|yaml|yml|toml))[`"\']',
        FileAction.MODIFIED,
    ),
    (
        r'(?:deleted?|removed?)\s+(?:the\s+)?(?:file\s+)?[`"\']([^`"\']+\.(?:py|js|ts|rs|go|java|md|json|yaml|yml|toml))[`"\']',
        FileAction.DELETED,
    ),
]

# Patterns for file references without explicit action (assume created if new)
FILE_REFERENCE_PATTERNS = [
    r'[`"\']([^`"\']+\.(?:py|js|ts|jsx|tsx|rs|go|java|md|json|yaml|yml|toml))[`"\']',
    r"(?:in|to|file)\s+([a-zA-Z0-9_/.-]+\.(?:py|js|ts|rs|go|java))",
]


def extract_file_operations(
    response: str,
    workspace_root: Path,
) -> dict[Path, FileAction]:
    """Extract file operations from agent response.

    Parses the agent response looking for:
    - Tool calls (write_file, edit_file, etc.)
    - Natural language descriptions of file operations
    - File path references

    Args:
        response: The agent's response text.
        workspace_root: Root path for resolving relative paths.

    Returns:
        Dictionary mapping file paths to their action (CREATED/MODIFIED/DELETED).
    """
    files: dict[Path, FileAction] = {}

    # Extract from explicit operation patterns
    for pattern, action in FILE_OPERATION_PATTERNS:
        for match in re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE):
            raw_path = match.group(1).strip()
            if raw_path:
                resolved = _resolve_path(raw_path, workspace_root)
                if resolved:
                    # Only update if not already set or if new action is DELETED
                    # DELETED takes precedence as it's the final state
                    if resolved not in files or action == FileAction.DELETED:
                        files[resolved] = action

    return files


def extract_file_references(response: str) -> list[str]:
    """Extract all file path references from agent response.

    Returns raw file paths found in the response, without resolving
    or determining action. Useful for tracking what files were mentioned.

    Args:
        response: The agent's response text.

    Returns:
        List of unique file path strings found.
    """
    refs: set[str] = set()

    for pattern in FILE_REFERENCE_PATTERNS:
        for match in re.finditer(pattern, response, re.IGNORECASE):
            ref = match.group(1).strip()
            if ref and _is_valid_file_ref(ref):
                refs.add(ref)

    return sorted(refs)


def _resolve_path(raw_path: str, root: Path) -> Path | None:
    """Resolve a raw path string to an absolute Path.

    Args:
        raw_path: The raw path string from agent response.
        root: Workspace root for resolving relative paths.

    Returns:
        Resolved Path or None if invalid.
    """
    if not raw_path or not _is_valid_file_ref(raw_path):
        return None

    # Clean the path
    cleaned = raw_path.strip().strip("'\"`")

    # Handle absolute paths
    if cleaned.startswith("/"):
        return Path(cleaned)

    # Relative path - resolve from workspace root
    return root / cleaned


def _is_valid_file_ref(ref: str) -> bool:
    """Check if a string looks like a valid file reference.

    Args:
        ref: The string to check.

    Returns:
        True if it looks like a file path.
    """
    if not ref or len(ref) < 3:
        return False

    # Must have an extension or be a known file
    if "." not in ref and ref not in ("Makefile", "Dockerfile", "README"):
        return False

    # Exclude URLs
    if ref.startswith(("http://", "https://", "ftp://")):
        return False

    # Exclude common false positives
    false_positives = {
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "et.",
        "al.",
        "0.0",
        "1.0",
        "2.0",
        "0.1",
        "1.1",
        "2.1",
    }
    if ref.lower() in false_positives:
        return False

    return True


# =============================================================================
# Decision Extraction
# =============================================================================

# Patterns for extracting key decisions from agent reasoning
DECISION_PATTERNS = [
    # "I will/need to/should" patterns
    (r"I (?:will|'ll|need to|should|must|am going to) (.+?)(?:\.|$)", 1),
    # "First/Then/Finally" patterns
    (r"(?:First|Then|Next|Finally|Now)[,:]?\s+(?:I (?:will|'ll|need to))?\s*(.+?)(?:\.|$)", 1),
    # "The issue/problem/fix is" patterns
    (r"The (?:issue|problem|error|fix|solution) (?:is|was|seems to be) (.+?)(?:\.|$)", 1),
    # "Let me/Let's" patterns
    (r"Let(?:'s| me) (.+?)(?:\.|$)", 1),
    # "I decided to" patterns
    (r"I (?:decided|chose|opted) to (.+?)(?:\.|$)", 1),
    # "Because/Since" reasoning
    (r"(?:Because|Since) (.+?), I", 1),
]


def extract_decisions_simple(response: str, max_decisions: int = 3) -> list[str]:
    """Extract key decisions from agent response using regex.

    Looks for common reasoning patterns in agent responses to identify
    key decisions and intentions.

    Args:
        response: The agent's response text.
        max_decisions: Maximum number of decisions to extract.

    Returns:
        List of decision strings (cleaned and truncated).
    """
    decisions: list[str] = []
    seen: set[str] = set()

    for pattern, group in DECISION_PATTERNS:
        for match in re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE):
            decision = match.group(group).strip()
            decision = _clean_decision(decision)

            # Skip if too short, too long, or duplicate
            if len(decision) < 10 or len(decision) > 150:
                continue

            normalized = decision.lower()
            if normalized in seen:
                continue
            seen.add(normalized)

            decisions.append(decision[:100])  # Truncate to 100 chars

            if len(decisions) >= max_decisions:
                return decisions

    return decisions


def _clean_decision(decision: str) -> str:
    """Clean a decision string.

    Args:
        decision: Raw decision string.

    Returns:
        Cleaned decision string.
    """
    # Remove leading/trailing whitespace and punctuation
    decision = decision.strip().strip(".,;:")

    # Remove markdown formatting
    decision = re.sub(r"\*\*(.+?)\*\*", r"\1", decision)
    decision = re.sub(r"`(.+?)`", r"\1", decision)

    # Normalize whitespace
    decision = " ".join(decision.split())

    return decision


# =============================================================================
# LLM-Based Extraction
# =============================================================================

EXTRACTION_PROMPT = """Analyze this agent execution response and extract:

1. A one-line summary of what was accomplished (max 80 characters)
2. Up to 3 key decisions the agent made

Task being executed: {task_title}

Agent Response:
{response}

Respond ONLY with valid JSON in this exact format:
{{"summary": "Brief summary here", "decisions": ["decision 1", "decision 2"]}}

JSON Response:"""


async def extract_with_llm(
    response: str,
    task_title: str,
    llm: Any,
    max_response_chars: int = 3000,
) -> tuple[str, list[str]]:
    """Use LLM to extract summary and decisions from agent response.

    Args:
        response: The agent's response text.
        task_title: Title of the task being executed.
        llm: LLM instance with async chat capability.
        max_response_chars: Maximum chars of response to send to LLM.

    Returns:
        Tuple of (summary, list of decisions).
    """
    # Truncate response to avoid excessive token usage
    truncated = response[:max_response_chars]
    if len(response) > max_response_chars:
        truncated += "\n... [truncated]"

    prompt = EXTRACTION_PROMPT.format(
        task_title=task_title,
        response=truncated,
    )

    try:
        # Call LLM
        if hasattr(llm, "achat"):
            result = await llm.achat(prompt)
        elif hasattr(llm, "chat"):
            result = llm.chat(prompt)
        else:
            logger.warning("LLM has no chat method, falling back to simple extraction")
            return task_title[:80], extract_decisions_simple(response)

        # Extract content from response
        content = result.content if hasattr(result, "content") else str(result)

        # Parse JSON
        parsed = _parse_extraction_json(content)
        if parsed:
            summary = parsed.get("summary", task_title)[:80]
            decisions = parsed.get("decisions", [])[:3]
            return summary, [str(d)[:100] for d in decisions]

        # Fallback if parsing fails
        logger.warning("Failed to parse LLM extraction response, using simple extraction")
        return task_title[:80], extract_decisions_simple(response)

    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}, using simple extraction")
        return task_title[:80], extract_decisions_simple(response)


def _parse_extraction_json(content: str) -> dict[str, Any] | None:
    """Parse JSON from LLM response.

    Handles common LLM response formats including:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after

    Args:
        content: LLM response content.

    Returns:
        Parsed dictionary or None if parsing fails.
    """
    # Try direct parsing first
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", content, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    json_match = re.search(r"\{[^{}]*\"summary\"[^{}]*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Main Extraction Function
# =============================================================================


async def extract_outcome(
    agent_response: str,
    workspace_root: Path,
    task_id: str,
    task_title: str,
    llm: Any | None = None,
) -> ExtractionResult:
    """Extract TaskOutcome data from agent execution response.

    Parses the agent's response to extract:
    - Files created, modified, or deleted
    - Key decisions made by the agent
    - A summary of what was accomplished

    Uses regex-based extraction by default. If an LLM is provided,
    uses it for higher-quality summary and decision extraction.

    Args:
        agent_response: The agent's full response text.
        workspace_root: Root path for resolving relative file paths.
        task_id: The task identifier (e.g., "1.1").
        task_title: The task title/description.
        llm: Optional LLM instance for enhanced extraction.

    Returns:
        ExtractionResult with files, decisions, and summary.

    Example:
        ```python
        result = await extract_outcome(
            agent_response="I will create src/utils.py...",
            workspace_root=Path("/project"),
            task_id="1.1",
            task_title="Create utils module",
        )
        ```
    """
    # Extract file operations
    files = extract_file_operations(agent_response, workspace_root)

    # Extract raw file references
    raw_refs = extract_file_references(agent_response)

    # Extract summary and decisions
    if llm:
        summary, decisions = await extract_with_llm(
            response=agent_response,
            task_title=task_title,
            llm=llm,
        )
    else:
        # Simple extraction without LLM
        summary = task_title[:80]
        decisions = extract_decisions_simple(agent_response)

    return ExtractionResult(
        files=files,
        key_decisions=decisions,
        summary=summary,
        raw_file_refs=raw_refs,
    )


def extract_outcome_sync(
    agent_response: str,
    workspace_root: Path,
    task_id: str,
    task_title: str,
) -> ExtractionResult:
    """Synchronous version of extract_outcome (no LLM support).

    For use in synchronous contexts where LLM extraction is not needed.

    Args:
        agent_response: The agent's full response text.
        workspace_root: Root path for resolving relative file paths.
        task_id: The task identifier (e.g., "1.1").
        task_title: The task title/description.

    Returns:
        ExtractionResult with files, decisions, and summary.
    """
    files = extract_file_operations(agent_response, workspace_root)
    raw_refs = extract_file_references(agent_response)
    decisions = extract_decisions_simple(agent_response)

    return ExtractionResult(
        files=files,
        key_decisions=decisions,
        summary=task_title[:80],
        raw_file_refs=raw_refs,
    )
