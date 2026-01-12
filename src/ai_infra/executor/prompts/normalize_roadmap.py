"""Prompt for LLM-based ROADMAP normalization.

This module provides the prompt template for extracting todos from any ROADMAP
format. The LLM can identify tasks regardless of whether they use checkboxes,
emojis, bullets, or prose.

Phase 5.13: LLM-Based ROADMAP Normalization
"""

# System prompt for the normalization task
NORMALIZE_ROADMAP_SYSTEM_PROMPT = """You are a task extraction expert. Your job is to analyze ROADMAP files and extract all actionable tasks into a structured JSON format.

You can recognize tasks in ANY format:
- Markdown checkboxes: [ ], [x], [X]
- Emojis: 🚀, ✅, ⏳, 📋
- Bullets: -, *, +, numbered lists
- Prose: "Next we need to...", "TODO:", "FIXME:"
- Section headers that are actually tasks

You are precise, thorough, and always output valid JSON."""

# Main prompt template - expects {roadmap_content} to be filled in
NORMALIZE_ROADMAP_PROMPT = """Analyze this ROADMAP file and extract ALL actionable tasks.

## Input ROADMAP:
```markdown
{roadmap_content}
```

## Instructions:
1. Identify ALL tasks/todos regardless of format (checkboxes, emojis, bullets, prose)
2. Determine if each task is pending or already completed
3. Extract file paths mentioned (e.g., `src/config.py`, "the config file")
4. Identify dependencies between tasks (if task B requires task A, note it)
5. Flatten subtasks into separate todos with incremented IDs
6. Preserve the original line number and text for each task

## Status Detection:
- **pending**: [ ], ⏳, 🚀, "TODO", unchecked, no completion indicator
- **completed**: [x], [X], ✅, ✔, "done", "completed", strikethrough
- **skipped**: ⏭, "skipped", "N/A", explicitly marked as not needed

## Output Format:
Return ONLY valid JSON with this exact structure:
```json
{{
  "todos": [
    {{
      "id": 1,
      "title": "Short actionable title (3-7 words)",
      "description": "Optional longer description with context",
      "status": "pending",
      "file_hints": ["path/to/file.py"],
      "dependencies": [],
      "source_line": 5,
      "source_text": "Original line from ROADMAP"
    }}
  ]
}}
```

## Rules:
- Every actionable item becomes a separate todo
- Use sequential IDs starting from 1
- Keep titles concise and action-oriented
- Extract file paths from backticks, quotes, or context
- Dependencies are IDs of tasks that must complete first
- Include source_line (1-based) and source_text for sync-back
- Ignore non-actionable content (headers, descriptions, notes)
- If a parent task has subtasks, create separate todos for each subtask

## Example Transformations:
- "🚀 Build the config" → {{"title": "Build config module", "status": "pending"}}
- "[x] Create database schema" → {{"title": "Create database schema", "status": "completed"}}
- "- [ ] Implement `src/auth.py`" → {{"title": "Implement auth module", "file_hints": ["src/auth.py"]}}

Now extract all todos from the ROADMAP above. Output ONLY the JSON object, no explanation:"""


# Compact prompt for smaller ROADMAPs (reduces token usage)
NORMALIZE_ROADMAP_PROMPT_COMPACT = """Extract todos from this ROADMAP as JSON.

ROADMAP:
{roadmap_content}

Output JSON with this structure:
{{"todos": [{{"id": 1, "title": "...", "description": "...", "status": "pending|completed|skipped", "file_hints": [], "dependencies": [], "source_line": N, "source_text": "..."}}]}}

Rules:
- Every task becomes a todo (checkboxes, emojis, bullets, prose)
- [x]/✅ = completed, [ ]/🚀 = pending
- Extract file paths from backticks
- Use sequential IDs from 1

JSON only:"""
