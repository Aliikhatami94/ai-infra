"""Build context node for executor graph.

Phase 1.2.2: Builds execution context and prompt for current task.
Phase 1.5: Uses few-shot templates for improved LLM output quality.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.prompts import get_few_shot_section, infer_template_type
from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.context import ProjectContext
    from ai_infra.executor.project_memory import ProjectMemory
    from ai_infra.executor.run_memory import RunMemory

logger = get_logger("executor.nodes.context")


async def build_context_node(
    state: ExecutorGraphState,
    *,
    project_context: ProjectContext | None = None,
    run_memory: RunMemory | None = None,
    project_memory: ProjectMemory | None = None,
    max_context_tokens: int = 50000,
    workspace_root: Path | str | None = None,
    # Phase 8.2: Skills injection (EXECUTOR_3.md)
    skills_db: Any | None = None,  # SkillsDatabase | None
    max_skills: int = 3,
) -> ExecutorGraphState:
    """Build execution context and prompt for current task.

    This node:
    1. Gets current_task from state
    2. Builds project context (file structure, relevant code)
    3. Includes run memory (previous task outcomes)
    4. Includes project memory (cross-run insights) - Phase 2.1.2
    5. Injects matching skills from SkillsDatabase - Phase 8.2
    6. Creates the final prompt for agent execution

    Args:
        state: Current graph state with current_task.
        project_context: Optional ProjectContext for semantic search.
        run_memory: Optional RunMemory for task-to-task context.
        project_memory: Optional ProjectMemory for cross-run insights (Phase 2.1.2).
        max_context_tokens: Maximum tokens for context.
        workspace_root: Root directory for workspace boundary constraints.
        skills_db: Optional SkillsDatabase for injecting learned patterns (Phase 8.2).
        max_skills: Maximum number of skills to inject (Phase 8.2, default: 3).

    Returns:
        Updated state with context, prompt, and skills_context.
    """
    current_task = state.get("current_task")

    # Determine workspace root from parameter or state
    if workspace_root is None:
        roadmap_path = state.get("roadmap_path")
        if roadmap_path:
            workspace_root = Path(roadmap_path).parent
        elif project_context is not None:
            workspace_root = project_context.root

    if current_task is None:
        logger.warning("No current task to build context for")
        return {
            **state,
            "context": "",
            "prompt": "",
            "error": {
                "error_type": "context",
                "message": "No current task to build context for",
                "node": "build_context",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    try:
        context_parts: list[str] = []

        # 1. Task description
        task_section = _build_task_section(current_task)
        context_parts.append(task_section)

        # Phase 2.4.2: Include execution plan if available
        task_plan = state.get("task_plan")
        if task_plan:
            plan_section = _build_plan_section(task_plan)
            if plan_section:
                context_parts.append(plan_section)
                logger.debug(f"Added execution plan for task [{current_task.id}]")

        # 2. Project structure (if available)
        if project_context is not None:
            try:
                structure = await asyncio.wait_for(
                    project_context.build_structure(),
                    timeout=NodeTimeouts.BUILD_CONTEXT / 2,
                )
                context_parts.append(f"## Project Structure\n\n```\n{structure}\n```")
            except TimeoutError:
                logger.warning("Timeout building project structure")
            except Exception as e:
                logger.warning(f"Failed to build project structure: {e}")

        # 3. Relevant code from file hints (enhanced with plan's likely_files)
        # Phase 2.4.2: Combine file hints with plan's likely_files
        file_hints = list(current_task.file_hints) if current_task.file_hints else []
        if task_plan and task_plan.get("likely_files"):
            for f in task_plan["likely_files"]:
                if f not in file_hints:
                    file_hints.append(f)
            logger.debug(f"Enhanced file hints with plan: {file_hints}")

        if file_hints and project_context is not None:
            try:
                relevant_code = await _get_relevant_code(
                    project_context,
                    file_hints,
                    max_tokens=max_context_tokens // 3,
                )
                if relevant_code:
                    context_parts.append(f"## Relevant Code\n\n{relevant_code}")
            except Exception as e:
                logger.warning(f"Failed to get relevant code: {e}")

        # 4. Run memory (previous task outcomes)
        # Phase 2.1.1: Use RunMemory object's get_context() method
        # Phase 9.2: Use ContextSummarizer for smart relevance filtering
        if run_memory is not None:
            try:
                from ai_infra.executor.context_carryover import ContextSummarizer

                memory_budget = max_context_tokens // 4  # Reserve ~25% for memory

                # Phase 9.2: Use ContextSummarizer if there are outcomes
                if run_memory.outcomes:
                    # Convert outcomes to task dicts for summarizer
                    previous_tasks = [
                        {
                            "title": o.title,
                            "files": [str(p) for p in o.files.keys()],
                            "summary": o.summary or f"Completed: {o.title}",
                            "key_decisions": o.key_decisions,
                        }
                        for o in run_memory.outcomes
                    ]

                    summarizer = ContextSummarizer(
                        max_tokens=memory_budget,
                        max_tasks=5,  # Allow more tasks than default
                        relevance_threshold=0.05,  # Lower threshold for within-run context
                    )

                    summarized_context = summarizer.summarize_for_task(
                        previous_tasks,
                        current_task,
                        max_tokens=memory_budget,
                    )

                    if summarized_context:
                        context_parts.append(summarized_context)
                        logger.debug(
                            f"Added smart run memory context for task [{current_task.id}]: "
                            f"{len(summarized_context)} chars, {len(run_memory.outcomes)} outcomes"
                        )
                    else:
                        # Fallback to standard context if summarizer returns nothing
                        memory_context = run_memory.get_context(
                            current_task_id=str(current_task.id),
                            max_tokens=memory_budget,
                        )
                        if memory_context:
                            context_parts.append(memory_context)
                            logger.debug(
                                f"Added run memory context (fallback) for task [{current_task.id}]"
                            )
                else:
                    # No outcomes yet, get_context handles empty case
                    memory_context = run_memory.get_context(
                        current_task_id=str(current_task.id),
                        max_tokens=memory_budget,
                    )
                    if memory_context:
                        context_parts.append(memory_context)
            except Exception as e:
                logger.warning(f"Failed to get run memory context: {e}")
        else:
            # Fallback: Use state dict if RunMemory object not available
            run_memory_dict = state.get("run_memory", {})
            if run_memory_dict:
                memory_section = _build_memory_section(run_memory_dict)
                if memory_section:
                    context_parts.append(memory_section)

        # 5. Project memory (cross-run insights)
        # Phase 2.1.2: Include project context from previous runs
        if project_memory is not None:
            try:
                # Get current task title for relevance filtering
                task_title = getattr(current_task, "title", "")
                project_context_str = project_memory.get_context(
                    task_title=task_title,
                    max_tokens=max_context_tokens // 8,  # Reserve ~12.5% for project memory
                )
                if project_context_str:
                    context_parts.append(project_context_str)
                    logger.debug(
                        f"Added project memory context for task [{current_task.id}]: "
                        f"{len(project_context_str)} chars"
                    )
            except Exception as e:
                logger.warning(f"Failed to get project memory context: {e}")

        # 6. Skills context (learned patterns)
        # Phase 8.2: Inject matching skills from SkillsDatabase
        skills_context: list[dict[str, Any]] = []
        if skills_db is not None:
            try:
                from ai_infra.executor.skills.models import SkillContext

                # Build skill context from task information
                task_title = getattr(current_task, "title", "")
                task_description = getattr(current_task, "description", "") or ""
                file_hints = list(getattr(current_task, "file_hints", []) or [])

                # Extract keywords from title and description
                task_keywords = _extract_keywords(task_title, task_description)

                # Infer language from file hints
                language = _infer_language(file_hints)

                skill_ctx = SkillContext(
                    language=language,
                    framework=state.get("project_framework"),
                    task_keywords=task_keywords,
                    task_title=task_title,
                    task_description=task_description,
                    file_hints=file_hints,
                )

                # Find matching skills
                matching_skills = skills_db.find_matching(skill_ctx, limit=max_skills)

                if matching_skills:
                    skills_context = [
                        {
                            "title": s.title,
                            "pattern": s.pattern,
                            "rationale": s.rationale,
                            "confidence": getattr(s, "confidence", 0.5),
                            "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                        }
                        for s in matching_skills
                    ]
                    logger.info(
                        f"Injecting {len(matching_skills)} skills into context for task [{current_task.id}]",
                        extra={"skills": [s.title for s in matching_skills]},
                    )
            except Exception as e:
                logger.warning(f"Failed to inject skills context: {e}")

        # Combine context
        full_context = "\n\n---\n\n".join(context_parts)

        # Build prompt with workspace boundary constraints and skills context
        # Phase 8.2: Pass skills_context to _build_prompt
        prompt = _build_prompt(current_task, full_context, workspace_root, skills_context)

        logger.info(
            f"Built context for task [{current_task.id}]: "
            f"{len(full_context)} chars, {len(prompt)} chars prompt"
            + (f", {len(skills_context)} skills" if skills_context else "")
        )

        # Debug: Log the task description being used
        logger.debug(
            f"Task [{current_task.id}] description: {current_task.description[:200] if current_task.description else 'None'}..."
        )

        return {
            **state,
            "context": full_context,
            "prompt": prompt,
            "skills_context": skills_context,  # Phase 8.2: Include in state
            "error": None,
        }

    except TimeoutError:
        logger.error("Timeout building context")
        return {
            **state,
            "context": "",
            "prompt": "",
            "error": {
                "error_type": "timeout",
                "message": f"Timeout building context after {NodeTimeouts.BUILD_CONTEXT}s",
                "node": "build_context",
                "task_id": str(current_task.id) if current_task else None,
                "recoverable": True,
                "stack_trace": None,
            },
        }
    except Exception as e:
        logger.exception(f"Failed to build context: {e}")
        return {
            **state,
            "context": "",
            "prompt": "",
            "error": {
                "error_type": "context",
                "message": str(e),
                "node": "build_context",
                "task_id": str(current_task.id) if current_task else None,
                "recoverable": True,
                "stack_trace": None,
            },
        }


def _build_task_section(task: Any) -> str:
    """Build the task description section."""
    lines = [
        "## Current Task",
        "",
        f"**ID**: {task.id}",
        f"**Title**: {task.title}",
    ]

    if task.description:
        lines.extend(["", f"**Description**: {task.description}"])

    if task.file_hints:
        hints = ", ".join(f"`{f}`" for f in task.file_hints[:5])
        if len(task.file_hints) > 5:
            hints += f" (+{len(task.file_hints) - 5} more)"
        lines.extend(["", f"**File Hints**: {hints}"])

    return "\n".join(lines)


def _build_plan_section(task_plan: dict[str, Any]) -> str:
    """Build the execution plan section.

    Phase 2.4.2: Formats the task_plan dict into a readable context section.

    Args:
        task_plan: Plan dict with likely_files, dependencies, risks, approach, complexity.

    Returns:
        Formatted plan section or empty string if plan is empty.
    """
    if not task_plan:
        return ""

    lines = ["## Execution Plan", ""]

    # Approach
    approach = task_plan.get("approach", "")
    if approach:
        lines.extend([f"**Approach**: {approach}", ""])

    # Complexity
    complexity = task_plan.get("complexity", "medium")
    lines.append(f"**Complexity**: {complexity.upper()}")

    # Likely files
    likely_files = task_plan.get("likely_files", [])
    if likely_files:
        lines.extend(["", "**Files to Modify**:"])
        for f in likely_files[:10]:  # Limit display
            lines.append(f"- `{f}`")

    # Dependencies
    dependencies = task_plan.get("dependencies", [])
    if dependencies:
        lines.extend(["", "**Dependencies**:"])
        for dep in dependencies[:5]:
            lines.append(f"- {dep}")

    # Risks
    risks = task_plan.get("risks", [])
    if risks:
        lines.extend(["", "**Risks to Consider**:"])
        for risk in risks[:5]:
            lines.append(f"- {risk}")

    return "\n".join(lines)


async def _get_relevant_code(
    project_context: ProjectContext,
    file_hints: list[str],
    max_tokens: int,
) -> str:
    """Get relevant code from file hints."""
    code_sections: list[str] = []
    tokens_used = 0

    for hint in file_hints[:5]:  # Limit to 5 files
        try:
            # Try to read the file directly if it exists
            file_path = project_context.root / hint
            if file_path.exists() and file_path.is_file():
                content = file_path.read_text(encoding="utf-8")
                # Rough token estimate: 4 chars per token
                estimated_tokens = len(content) // 4
                if tokens_used + estimated_tokens > max_tokens:
                    break
                code_sections.append(f"### {hint}\n\n```\n{content}\n```")
                tokens_used += estimated_tokens
        except Exception as e:
            logger.debug(f"Could not read {hint}: {e}")

    return "\n\n".join(code_sections)


def _build_memory_section(run_memory: dict[str, Any]) -> str:
    """Build the run memory section."""
    if not run_memory:
        return ""

    lines = ["## Previous Tasks This Run"]

    completed = run_memory.get("completed_tasks", [])
    for task_info in completed[-5:]:  # Last 5 tasks
        task_id = task_info.get("id", "?")
        title = task_info.get("title", "Unknown")
        lines.append(f"- [{task_id}] {title}")

    return "\n".join(lines)


def _extract_keywords(title: str, description: str) -> list[str]:
    """Extract keywords from task title and description.

    Phase 8.2: Used for skill matching.

    Args:
        title: Task title.
        description: Task description.

    Returns:
        List of extracted keywords.
    """
    import re

    # Combine title and description
    text = f"{title} {description}".lower()

    # Remove punctuation and split into words
    words = re.findall(r"\b[a-z][a-z0-9_]*\b", text)

    # Filter out common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "he",
        "she",
        "his",
        "her",
        "if",
        "then",
        "else",
        "when",
        "where",
        "which",
        "who",
        "what",
        "how",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "into",
        "over",
        "after",
        "before",
        "above",
        "below",
        "between",
        "through",
        "during",
        "add",
        "create",
        "implement",
        "update",
        "fix",
        "make",
        "new",
        "use",
    }

    # Extract meaningful keywords (min length 2, not stop words)
    keywords = [w for w in words if len(w) >= 2 and w not in stop_words]

    # Return unique keywords, preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords[:20]  # Limit to 20 keywords


def _infer_language(file_hints: list[str]) -> str:
    """Infer programming language from file hints.

    Phase 8.2: Used for skill matching.

    Args:
        file_hints: List of file paths.

    Returns:
        Inferred language (default: "python").
    """
    if not file_hints:
        return "python"

    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".php": "php",
        ".sql": "sql",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".md": "markdown",
    }

    for hint in file_hints:
        for ext, lang in extension_map.items():
            if hint.endswith(ext):
                return lang

    return "python"


def _build_skills_section(skills_context: list[dict[str, Any]]) -> str:
    """Build the skills section for the prompt.

    Phase 8.2: Formats learned skills/patterns for injection into prompt.

    Args:
        skills_context: List of skill dicts with title, pattern, rationale, confidence.

    Returns:
        Formatted skills section string.
    """
    if not skills_context:
        return ""

    lines = [
        "",
        "---",
        "",
        "## Relevant Patterns from Past Experience",
        "",
        "The following patterns have been learned from successful past tasks. "
        "Consider them when implementing the current task.",
        "",
    ]

    for skill in skills_context:
        title = skill.get("title", "Unnamed Pattern")
        confidence = skill.get("confidence", 0.5)
        pattern = skill.get("pattern", "")
        rationale = skill.get("rationale", "")
        skill_type = skill.get("type", "pattern")

        lines.append(f"### {title} (confidence: {confidence:.0%})")
        lines.append("")

        if pattern:
            lines.append("**Pattern:**")
            lines.append("```")
            lines.append(pattern.strip())
            lines.append("```")
            lines.append("")

        if rationale:
            lines.append(f"**Why this works:** {rationale}")
            lines.append("")

        if skill_type == "anti_pattern":
            lines.append("**Note:** This is an anti-pattern - avoid this approach.")
            lines.append("")

    lines.append("")

    return "\n".join(lines)


def _build_workspace_boundary_section(workspace_root: Path | str | None) -> str:
    """Build the workspace boundary constraints section.

    This matches the legacy executor's workspace boundary instructions
    to ensure consistent behavior between graph and legacy modes.
    """
    if workspace_root is None:
        workspace_root = "."

    return f"""## CRITICAL: Workspace Boundary

**Workspace root**: `{workspace_root}`

**ALL file operations MUST stay within this workspace directory.**
You CANNOT access files outside this directory.

**When using file tools:**
- `ls`: Always specify a path relative to workspace root (e.g., `.` or `src/`)
- `read`: Use workspace-relative paths (e.g., `src/user.py`)
- `write`: Use workspace-relative paths (e.g., `src/user.py`)
- `grep`: ALWAYS specify a path parameter (e.g., `path='.'` or `path='src/'`)
- `glob`: Use workspace-relative patterns (e.g., `src/**/*.py`)

**NEVER:**
- Access system directories (`/usr`, `/etc`, `/var`, `/bin`, `/sbin`)
- Use absolute paths outside the workspace
- Call `grep` without specifying a `path` parameter

**Before importing or referencing any file:**
1. Check if it exists using the `ls` or `read` tool
2. If importing from a module, verify the module file exists first
3. If a required file doesn't exist, create it first"""


def _build_prompt(
    task: Any,
    context: str,
    workspace_root: Path | str | None = None,
    skills_context: list[dict[str, Any]] | None = None,
) -> str:
    """Build the final prompt for agent execution.

    Phase 1.5: Uses few-shot templates based on task type for improved
    LLM output quality, especially with weaker models.
    Phase 8.2: Includes learned skills/patterns from SkillsDatabase.

    Args:
        task: The current task to execute.
        context: Built context including project structure and relevant code.
        workspace_root: Root directory for workspace boundary constraints.
        skills_context: Optional list of matching skills to inject (Phase 8.2).

    Returns:
        Complete prompt for the agent with few-shot examples and skills.
    """
    workspace_section = _build_workspace_boundary_section(workspace_root)

    # Phase 1.5: Get few-shot section based on task type
    few_shot_section = get_few_shot_section(task)

    # Infer template type for logging
    file_hints = getattr(task, "file_hints", None)
    template_type = infer_template_type(
        task.title,
        getattr(task, "description", ""),
        file_hints,
    )
    logger.debug(f"Using template type '{template_type.value}' for task [{task.id}]")

    # Phase 8.2: Build skills section if available
    skills_section = _build_skills_section(skills_context) if skills_context else ""

    return f"""You are an autonomous development agent executing a task from a ROADMAP.

{context}

---

{workspace_section}
{skills_section}
---

## Few-Shot Example

The following example shows the expected code style and quality:

{few_shot_section}

---

## Instructions

Execute ONLY the current task described above. Do NOT look ahead to future tasks.

**IMPORTANT**: You must complete ONLY this specific task, nothing more:
- Do NOT implement features from other tasks in the roadmap
- Do NOT anticipate what future tasks might need
- If the task says to create basic functions, create ONLY basic functions
- Other tasks will handle additional features - do not do their work

Steps:
1. Read the current task requirements carefully
2. Implement EXACTLY what is asked - no more, no less
3. Use RELATIVE paths for all file operations (e.g., `src/file.py` not `/src/file.py`)
4. Ensure the code is correct and follows project conventions
5. Follow the example's style (type hints, docstrings, error handling)
6. Report what files you modified

Complete the task: **{task.title}**
"""
