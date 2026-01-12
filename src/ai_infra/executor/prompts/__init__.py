"""Prompts for the Executor module.

This package contains prompt templates used by the Executor for LLM-based
operations like ROADMAP normalization, task extraction, and decision making.

Phase 1.5: Added few-shot templates for code generation to improve LLM output
quality with weaker models.
"""

from ai_infra.executor.prompts.normalize_roadmap import (
    NORMALIZE_ROADMAP_PROMPT,
    NORMALIZE_ROADMAP_SYSTEM_PROMPT,
)
from ai_infra.executor.prompts.templates import (
    CONFIG_TEMPLATE,
    DOCUMENTATION_TEMPLATE,
    GENERIC_TEMPLATE,
    PYTHON_CODE_TEMPLATE,
    SCRIPT_TEMPLATE,
    TEMPLATES,
    TEST_CODE_TEMPLATE,
    TemplateType,
    format_template,
    get_few_shot_section,
    get_template,
    infer_template_type,
)

__all__ = [
    # Roadmap normalization
    "NORMALIZE_ROADMAP_PROMPT",
    "NORMALIZE_ROADMAP_SYSTEM_PROMPT",
    # Phase 1.5: Few-shot templates
    "CONFIG_TEMPLATE",
    "DOCUMENTATION_TEMPLATE",
    "GENERIC_TEMPLATE",
    "PYTHON_CODE_TEMPLATE",
    "SCRIPT_TEMPLATE",
    "TEMPLATES",
    "TEST_CODE_TEMPLATE",
    "TemplateType",
    "format_template",
    "get_few_shot_section",
    "get_template",
    "infer_template_type",
]
