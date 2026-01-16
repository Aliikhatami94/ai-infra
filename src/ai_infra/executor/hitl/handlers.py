"""HITL action handlers for processing human actions.

Phase 4.1.2-4.1.5 of EXECUTOR_1.md: Handler implementations.

This module provides handlers for each HITL action type:
- EditHandler: Processes edit requests to modify proposals
- SuggestHandler: Incorporates hints into the plan
- ExplainHandler: Provides reasoning explanations
- RollbackHandler: Restores previous checkpoint states
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_infra.executor.hitl.actions import (
    HITLAction,
    HITLActionType,
    HITLProposal,
    HITLResponse,
)
from ai_infra.llm import LLM

if TYPE_CHECKING:
    from ai_infra.executor.agents.base import ExecutionContext
    from ai_infra.executor.checkpoint import CommitInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Base Handler
# =============================================================================


class HITLActionHandler(ABC):
    """Abstract base class for HITL action handlers.

    All handlers must implement the handle method to process their
    specific action type.

    Attributes:
        action_type: The action type this handler processes.
    """

    action_type: HITLActionType

    @abstractmethod
    async def handle(
        self,
        action: HITLAction,
        proposal: HITLProposal,
        context: ExecutionContext,
    ) -> HITLResponse:
        """Handle the HITL action.

        Args:
            action: The human action to process.
            proposal: The current proposal being considered.
            context: Execution context.

        Returns:
            HITLResponse indicating how the action was processed.
        """
        ...


# =============================================================================
# Edit Handler (Phase 4.1.2)
# =============================================================================


@dataclass
class EditHandler(HITLActionHandler):
    """Handle edit requests to modify proposals.

    Phase 4.1.2: Parses edit instructions and revises the proposal.

    The EditHandler uses an LLM to understand the edit request and
    generate a revised plan that incorporates the changes.

    Attributes:
        llm: LLM instance for processing edits.
        action_type: Always HITLActionType.EDIT.

    Example:
        ```python
        handler = EditHandler(llm=my_llm)

        action = HITLAction(
            type=HITLActionType.EDIT,
            content="Use pytest-asyncio instead of unittest",
        )

        response = await handler.handle(action, proposal, context)
        # response.revised_plan contains the updated approach
        ```
    """

    llm: LLM
    action_type: HITLActionType = HITLActionType.EDIT

    async def handle(
        self,
        action: HITLAction,
        proposal: HITLProposal,
        context: ExecutionContext,
    ) -> HITLResponse:
        """Process an edit request.

        Uses LLM to understand the edit instruction and revise the plan.

        Args:
            action: Edit action with content containing the edit instruction.
            proposal: Current proposal to be modified.
            context: Execution context with relevant files.

        Returns:
            HITLResponse with revised plan or error.
        """
        if not action.content:
            return HITLResponse(
                understood=False,
                error="Edit action requires content specifying the changes",
                next_step="Please provide edit instructions",
            )

        edit_prompt = f"""You are revising a plan based on user feedback.

CURRENT PROPOSAL:
{proposal.description}

PLANNED ACTIONS:
{chr(10).join(f"- {a}" for a in proposal.planned_actions)}

FILES TO MODIFY:
{chr(10).join(f"- {f}" for f in proposal.files_affected)}

ORIGINAL RATIONALE:
{proposal.rationale}

USER'S EDIT REQUEST:
{action.content}

Based on the user's edit request, provide a revised plan.
Be specific about what changes will be made differently.
Keep the same format but incorporate the user's feedback."""

        try:
            result = await self.llm.acomplete(edit_prompt)
            revised_plan = result.content if hasattr(result, "content") else str(result)

            logger.info(
                "Processed edit request",
                extra={
                    "task_id": proposal.task_id,
                    "edit_instruction": action.content[:100],
                },
            )

            return HITLResponse(
                understood=True,
                revised_plan=revised_plan,
                next_step=f"Proceeding with revised plan for task {proposal.task_id}",
                metadata={"original_instruction": action.content},
            )

        except Exception as e:
            logger.exception("Failed to process edit request")
            return HITLResponse(
                understood=False,
                error=f"Failed to process edit: {e!s}",
                next_step="Using original plan",
            )


# =============================================================================
# Suggest Handler (Phase 4.1.3)
# =============================================================================


@dataclass
class SuggestHandler(HITLActionHandler):
    """Handle suggestion hints from the user.

    Phase 4.1.3: Incorporates hints without requiring full specification.

    The SuggestHandler is for when the user wants to nudge the agent
    in a direction without specifying exactly what to do.

    Attributes:
        llm: LLM instance for processing suggestions.
        action_type: Always HITLActionType.SUGGEST.

    Example:
        ```python
        handler = SuggestHandler(llm=my_llm)

        action = HITLAction(
            type=HITLActionType.SUGGEST,
            content="Consider using caching for performance",
        )

        response = await handler.handle(action, proposal, context)
        ```
    """

    llm: LLM
    action_type: HITLActionType = HITLActionType.SUGGEST

    async def handle(
        self,
        action: HITLAction,
        proposal: HITLProposal,
        context: ExecutionContext,
    ) -> HITLResponse:
        """Process a suggestion hint.

        Uses LLM to incorporate the suggestion into the current plan
        while maintaining the overall approach.

        Args:
            action: Suggest action with content containing the hint.
            proposal: Current proposal to enhance.
            context: Execution context.

        Returns:
            HITLResponse with revised plan incorporating suggestion.
        """
        if not action.content:
            return HITLResponse(
                understood=False,
                error="Suggest action requires content with the suggestion",
                next_step="Please provide a suggestion",
            )

        suggest_prompt = f"""You are improving a plan based on a user's suggestion.
The suggestion is a hint - incorporate it thoughtfully without changing the core approach.

CURRENT PROPOSAL:
{proposal.description}

PLANNED ACTIONS:
{chr(10).join(f"- {a}" for a in proposal.planned_actions)}

USER'S SUGGESTION:
{action.content}

PROJECT CONTEXT:
{context.summary[:500] if context.summary else "No additional context"}

Incorporate the suggestion into the plan.
Keep the same overall structure but enhance it with the user's insight.
Explain how the suggestion will be incorporated."""

        try:
            result = await self.llm.acomplete(suggest_prompt)
            revised_plan = result.content if hasattr(result, "content") else str(result)

            logger.info(
                "Incorporated suggestion",
                extra={
                    "task_id": proposal.task_id,
                    "suggestion": action.content[:100],
                },
            )

            return HITLResponse(
                understood=True,
                revised_plan=revised_plan,
                next_step=f"Proceeding with enhanced plan for task {proposal.task_id}",
                metadata={"suggestion": action.content},
            )

        except Exception as e:
            logger.exception("Failed to incorporate suggestion")
            return HITLResponse(
                understood=False,
                error=f"Failed to incorporate suggestion: {e!s}",
                next_step="Using original plan",
            )


# =============================================================================
# Explain Handler (Phase 4.1.4)
# =============================================================================


@dataclass
class ExplainHandler(HITLActionHandler):
    """Handle explanation requests.

    Phase 4.1.4: Provides reasoning behind agent decisions.

    The ExplainHandler answers user questions about why the agent
    chose a particular approach, what alternatives were considered,
    and the reasoning behind specific decisions.

    Attributes:
        llm: LLM instance for generating explanations.
        action_type: Always HITLActionType.EXPLAIN.

    Example:
        ```python
        handler = ExplainHandler(llm=my_llm)

        action = HITLAction(
            type=HITLActionType.EXPLAIN,
            content="Why are you using SQLite instead of PostgreSQL?",
        )

        response = await handler.handle(action, proposal, context)
        # response.explanation contains the reasoning
        ```
    """

    llm: LLM
    action_type: HITLActionType = HITLActionType.EXPLAIN

    async def handle(
        self,
        action: HITLAction,
        proposal: HITLProposal,
        context: ExecutionContext,
    ) -> HITLResponse:
        """Generate an explanation for the current proposal.

        Uses LLM to explain the reasoning behind the proposal,
        answering the user's specific question if provided.

        Args:
            action: Explain action with optional question in content.
            proposal: Current proposal to explain.
            context: Execution context with relevant files.

        Returns:
            HITLResponse with explanation.
        """
        question = action.content or "Why did you choose this approach?"

        explain_prompt = f"""You are explaining your reasoning for a proposed plan.

PROPOSAL:
{proposal.description}

PLANNED ACTIONS:
{chr(10).join(f"- {a}" for a in proposal.planned_actions)}

RATIONALE:
{proposal.rationale}

ALTERNATIVES CONSIDERED:
{chr(10).join(f"- {a}" for a in proposal.alternatives_considered) or "None explicitly listed"}

FILES TO MODIFY:
{chr(10).join(f"- {f}" for f in proposal.files_affected)}

PROJECT CONTEXT:
{context.summary[:500] if context.summary else "No additional context"}

RELEVANT FILES:
{chr(10).join(context.relevant_files[:5]) if context.relevant_files else "None listed"}

USER'S QUESTION:
{question}

Explain your reasoning clearly and concisely.
Reference specific context that informed your decision.
If the user is asking about alternatives, explain why they were not chosen."""

        try:
            result = await self.llm.acomplete(explain_prompt)
            explanation = result.content if hasattr(result, "content") else str(result)

            logger.info(
                "Generated explanation",
                extra={
                    "task_id": proposal.task_id,
                    "question": question[:100],
                },
            )

            return HITLResponse(
                understood=True,
                revised_plan=None,  # Explanation doesn't change the plan
                explanation=explanation,
                next_step="Waiting for user decision after explanation",
                metadata={"question": question},
            )

        except Exception as e:
            logger.exception("Failed to generate explanation")
            return HITLResponse(
                understood=False,
                error=f"Failed to generate explanation: {e!s}",
                next_step="Waiting for user decision",
            )


# =============================================================================
# Rollback Handler (Phase 4.1.5)
# =============================================================================


@dataclass
class RollbackHandler(HITLActionHandler):
    """Handle rollback requests to previous checkpoints.

    Phase 4.1.5: Restores previous state and allows retry.

    The RollbackHandler finds the specified checkpoint and restores
    the workspace to that state, allowing the user to try a different
    approach.

    Attributes:
        checkpoint_manager: Manager for git checkpoints.
        action_type: Always HITLActionType.ROLLBACK.

    Example:
        ```python
        handler = RollbackHandler(checkpoint_manager=manager)

        action = HITLAction(
            type=HITLActionType.ROLLBACK,
            target="task_1.2.1",  # Rollback to before this task
        )

        response = await handler.handle(action, proposal, context)
        ```
    """

    checkpoint_manager: Any  # CheckpointManager
    action_type: HITLActionType = HITLActionType.ROLLBACK

    async def handle(
        self,
        action: HITLAction,
        proposal: HITLProposal,
        context: ExecutionContext,
    ) -> HITLResponse:
        """Rollback to a previous checkpoint.

        Finds the target checkpoint and restores the workspace state.

        Args:
            action: Rollback action with optional target checkpoint.
            proposal: Current proposal (used for context).
            context: Execution context with workspace path.

        Returns:
            HITLResponse indicating rollback success or failure.
        """
        target = action.target or "last"

        try:
            # Find the checkpoint
            if target == "last":
                checkpoint = await self._get_last_checkpoint(context)
            else:
                checkpoint = await self._get_checkpoint_by_task(target, context)

            if not checkpoint:
                return HITLResponse(
                    understood=False,
                    error=f"No checkpoint found for target: {target}",
                    next_step="Cannot rollback, continuing from current state",
                )

            # Perform the rollback
            await self._perform_rollback(checkpoint, context)

            logger.info(
                "Rolled back to checkpoint",
                extra={
                    "target": target,
                    "checkpoint_sha": checkpoint.short_sha
                    if hasattr(checkpoint, "short_sha")
                    else str(checkpoint),
                },
            )

            return HITLResponse(
                understood=True,
                revised_plan="Rolled back and will retry with fresh approach",
                explanation=f"Restored to checkpoint: {target}",
                next_step="Retrying from checkpoint, ready for new approach",
                metadata={
                    "rollback_target": target,
                    "checkpoint": str(checkpoint),
                },
            )

        except Exception as e:
            logger.exception("Failed to rollback")
            return HITLResponse(
                understood=False,
                error=f"Rollback failed: {e!s}",
                next_step="Continuing from current state",
            )

    async def _get_last_checkpoint(
        self,
        context: ExecutionContext,
    ) -> CommitInfo | None:
        """Get the most recent checkpoint.

        Args:
            context: Execution context with workspace path.

        Returns:
            The last checkpoint or None.
        """
        if not self.checkpoint_manager:
            return None

        try:
            # Get recent commits
            commits = self.checkpoint_manager.get_executor_commits(limit=1)
            return commits[0] if commits else None
        except Exception as e:
            logger.warning(f"Failed to get last checkpoint: {e}")
            return None

    async def _get_checkpoint_by_task(
        self,
        task_id: str,
        context: ExecutionContext,
    ) -> CommitInfo | None:
        """Get the checkpoint for a specific task.

        Args:
            task_id: Task ID to find checkpoint for.
            context: Execution context.

        Returns:
            The checkpoint for the task or None.
        """
        if not self.checkpoint_manager:
            return None

        try:
            # Find commits for this task
            commits = self.checkpoint_manager.get_commits_for_task(task_id)
            # Return the first (oldest) commit for this task
            return commits[0] if commits else None
        except Exception as e:
            logger.warning(f"Failed to get checkpoint for task {task_id}: {e}")
            return None

    async def _perform_rollback(
        self,
        checkpoint: CommitInfo,
        context: ExecutionContext,
    ) -> None:
        """Perform the actual rollback.

        Args:
            checkpoint: The checkpoint to rollback to.
            context: Execution context.

        Raises:
            Exception: If rollback fails.
        """
        if not self.checkpoint_manager:
            msg = "No checkpoint manager available"
            raise ValueError(msg)

        # Use the checkpoint manager's rollback functionality
        # This typically does a git reset or checkout
        parent_sha = checkpoint.sha + "^"  # Parent of the checkpoint
        self.checkpoint_manager.rollback_to(parent_sha)


# =============================================================================
# Handler Registry
# =============================================================================


class HITLHandlerRegistry:
    """Registry for HITL action handlers.

    Provides a central place to get the appropriate handler for an action.

    Example:
        ```python
        registry = HITLHandlerRegistry()
        registry.register(HITLActionType.EDIT, EditHandler(llm=my_llm))
        registry.register(HITLActionType.SUGGEST, SuggestHandler(llm=my_llm))

        handler = registry.get_handler(action.type)
        if handler:
            response = await handler.handle(action, proposal, context)
        ```
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: dict[HITLActionType, HITLActionHandler] = {}

    def register(
        self,
        action_type: HITLActionType,
        handler: HITLActionHandler,
    ) -> None:
        """Register a handler for an action type.

        Args:
            action_type: The action type to handle.
            handler: The handler instance.
        """
        self._handlers[action_type] = handler
        logger.debug(f"Registered handler for {action_type.value}")

    def get_handler(self, action_type: HITLActionType) -> HITLActionHandler | None:
        """Get the handler for an action type.

        Args:
            action_type: The action type to get handler for.

        Returns:
            The handler or None if not registered.
        """
        return self._handlers.get(action_type)

    def has_handler(self, action_type: HITLActionType) -> bool:
        """Check if a handler is registered for an action type.

        Args:
            action_type: The action type to check.

        Returns:
            True if a handler is registered.
        """
        return action_type in self._handlers

    @classmethod
    def create_default(
        cls,
        llm: LLM,
        checkpoint_manager: Any | None = None,
    ) -> HITLHandlerRegistry:
        """Create a registry with default handlers.

        Args:
            llm: LLM instance for handlers that need it.
            checkpoint_manager: Optional checkpoint manager for rollback.

        Returns:
            Registry with all default handlers registered.
        """
        registry = cls()

        # Register LLM-based handlers
        registry.register(HITLActionType.EDIT, EditHandler(llm=llm))
        registry.register(HITLActionType.SUGGEST, SuggestHandler(llm=llm))
        registry.register(HITLActionType.EXPLAIN, ExplainHandler(llm=llm))

        # Register checkpoint-based handlers
        if checkpoint_manager:
            registry.register(
                HITLActionType.ROLLBACK,
                RollbackHandler(checkpoint_manager=checkpoint_manager),
            )

        logger.info(f"Created HITL handler registry with {len(registry._handlers)} handlers")
        return registry
