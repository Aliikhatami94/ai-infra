"""HITL (Human-in-the-Loop) models for approval workflows.

This module provides structured models for approval requests and responses,
designed to work with both console scripts and web applications.

Example (Console):
    ```python
    agent = Agent(
        tools=[dangerous_tool],
        require_approval=True,  # All tools need approval
    )
    result = agent.run("Do something dangerous")
    # Console prompt: "Approve [tool_name](args)? [y/n]"
    ```

Example (Web App):
    ```python
    async def web_approval(request: ApprovalRequest) -> ApprovalResponse:
        await websocket.send_json(request.model_dump())
        response = await websocket.receive_json(timeout=300)
        return ApprovalResponse(**response)

    agent = Agent(
        tools=[dangerous_tool],
        approval_handler=web_approval,
    )
    ```
"""

from __future__ import annotations

import ast
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ApprovalRequest(BaseModel):
    """Request for human approval before tool execution.

    Attributes:
        id: Unique request ID for tracking
        tool_name: Name of the tool being called
        args: Arguments to the tool
        context: Optional conversation context
        timestamp: When approval was requested
        timeout: Seconds before auto-reject (0 = no timeout)
        metadata: Custom metadata for the request
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    args: dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeout: int = 300  # 5 minutes default
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_console_prompt(self) -> str:
        """Format the request as a console prompt."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"Approve {self.tool_name}({args_str})?"


class ApprovalResponse(BaseModel):
    """Response to an approval request.

    Attributes:
        approved: Whether the request was approved
        modified_args: Optional modified arguments (if approved with changes)
        reason: Optional reason for the decision
        approver: Optional identifier of who approved
        timestamp: When the response was given
    """

    approved: bool
    modified_args: dict[str, Any] | None = None
    reason: str | None = None
    approver: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def approve(
        cls,
        *,
        modified_args: dict[str, Any] | None = None,
        reason: str | None = None,
        approver: str | None = None,
    ) -> ApprovalResponse:
        """Create an approval response."""
        return cls(
            approved=True,
            modified_args=modified_args,
            reason=reason,
            approver=approver,
        )

    @classmethod
    def reject(
        cls,
        *,
        reason: str | None = None,
        approver: str | None = None,
    ) -> ApprovalResponse:
        """Create a rejection response."""
        return cls(
            approved=False,
            reason=reason,
            approver=approver,
        )


class OutputReviewRequest(BaseModel):
    """Request for human review of model output.

    Attributes:
        id: Unique request ID
        output: The model output to review
        context: Conversation context
        timestamp: When review was requested
        metadata: Custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    output: str
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputReviewResponse(BaseModel):
    """Response to an output review request.

    Attributes:
        action: What to do with the output (pass, modify, block)
        replacement: Replacement text if action is modify/block
        reason: Optional reason for the decision
        reviewer: Optional identifier of who reviewed
    """

    action: Literal["pass", "modify", "block"] = "pass"
    replacement: str | None = None
    reason: str | None = None
    reviewer: str | None = None

    @classmethod
    def allow(cls) -> OutputReviewResponse:
        """Allow the output unchanged."""
        return cls(action="pass")

    @classmethod
    def modify(cls, replacement: str, *, reason: str | None = None) -> OutputReviewResponse:
        """Modify the output."""
        return cls(action="modify", replacement=replacement, reason=reason)

    @classmethod
    def block(
        cls,
        replacement: str = "[Content blocked by reviewer]",
        *,
        reason: str | None = None,
    ) -> OutputReviewResponse:
        """Block the output."""
        return cls(action="block", replacement=replacement, reason=reason)


# Type aliases for handlers
ApprovalHandler = Callable[[ApprovalRequest], ApprovalResponse]
AsyncApprovalHandler = Callable[
    [ApprovalRequest], "Awaitable[ApprovalResponse]"
]  # Returns awaitable
OutputReviewer = Callable[[OutputReviewRequest], OutputReviewResponse]
AsyncOutputReviewer = Callable[
    [OutputReviewRequest], "Awaitable[OutputReviewResponse]"
]  # Returns awaitable


def console_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
    """Default console-based approval handler.

    Prompts user in terminal with y/n choice.
    """
    prompt = request.to_console_prompt()
    print("\nTool Approval Required")
    print(f"   {prompt}")

    try:
        while True:
            ans = input("   [y]es / [n]o / [m]odify args: ").strip().lower()
            if ans in ("y", "yes"):
                return ApprovalResponse.approve(approver="console")
            elif ans in ("n", "no"):
                return ApprovalResponse.reject(reason="Rejected by user", approver="console")
            elif ans in ("m", "modify"):
                print(f"   Current args: {request.args}")
                # Simple modification: let user enter new args as Python dict literal
                # Using ast.literal_eval for safety - only allows dict/list/str/int/float/bool/None
                try:
                    new_args_str = input("   New args (Python dict literal): ").strip()
                    new_args = ast.literal_eval(new_args_str)  # Safe: only literals
                    if isinstance(new_args, dict):
                        return ApprovalResponse.approve(
                            modified_args=new_args,
                            reason="Modified by user",
                            approver="console",
                        )
                    print("   Invalid: must be a dict")
                except (ValueError, SyntaxError) as e:
                    print(f"   Error parsing args: {e}")
                    print("   Hint: Only dict literals allowed, e.g. {'key': 'value', 'count': 42}")
            else:
                print("   Please enter y, n, or m")
    except EOFError:
        return ApprovalResponse.reject(reason="EOF - no input available", approver="console")
    except KeyboardInterrupt:
        return ApprovalResponse.reject(reason="Cancelled by user", approver="console")


def auto_approve_handler(request: ApprovalRequest) -> ApprovalResponse:
    """Handler that auto-approves all requests (for testing/development)."""
    return ApprovalResponse.approve(reason="Auto-approved", approver="auto")


def auto_reject_handler(request: ApprovalRequest) -> ApprovalResponse:
    """Handler that auto-rejects all requests (for testing)."""
    return ApprovalResponse.reject(reason="Auto-rejected", approver="auto")


# Selective approval helper
def create_selective_handler(
    tools_requiring_approval: list[str],
    handler: ApprovalHandler = console_approval_handler,
) -> ApprovalHandler:
    """Create a handler that only prompts for specific tools.

    Args:
        tools_requiring_approval: List of tool names that require approval
        handler: Handler to use for tools requiring approval

    Returns:
        A handler that auto-approves tools not in the list

    Example:
        ```python
        approval_handler = create_selective_handler(
            ["dangerous_tool", "delete_file"],
            handler=console_approval_handler,
        )
        agent = Agent(tools=[...], approval_handler=approval_handler)
        ```
    """

    def selective_handler(request: ApprovalRequest) -> ApprovalResponse:
        if request.tool_name in tools_requiring_approval:
            return handler(request)
        return ApprovalResponse.approve(reason="Tool not in approval list", approver="auto")

    return selective_handler


# =============================================================================
# Multi-Level Approval
# =============================================================================


class ApprovalRule(BaseModel):
    """Rule specifying approval requirements for a tool.

    This allows fine-grained control over which tools need approval,
    who can approve them, and whether all approvers must approve.

    Attributes:
        required: Whether approval is required at all
        approvers: List of approver identities who can approve (None = any)
        require_all: If True, all approvers must approve. If False, any one can approve.
        priority: Priority level for ordering (higher = more important)
        timeout: Custom timeout for this rule (overrides default)
        message: Custom message to show when requesting approval

    Example:
        ```python
        from ai_infra.llm.tools import ApprovalRule

        # Simple: any approver
        ApprovalRule(required=True)

        # Specific approvers
        ApprovalRule(required=True, approvers=["admin", "security_team"])

        # Require ALL approvers (multi-signature)
        ApprovalRule(
            required=True,
            approvers=["admin", "legal", "security"],
            require_all=True,
        )

        # High-priority with custom message
        ApprovalRule(
            required=True,
            approvers=["cto"],
            priority=100,
            message="This action requires CTO approval",
        )
        ```
    """

    required: bool = True
    approvers: list[str] | None = None  # None = any approver allowed
    require_all: bool = False  # If True, all approvers must approve
    priority: int = 0  # Higher = more important
    timeout: int | None = None  # Override default timeout
    message: str | None = None  # Custom approval message

    @classmethod
    def no_approval(cls) -> ApprovalRule:
        """Create a rule that requires no approval."""
        return cls(required=False)

    @classmethod
    def any_approver(cls, *, timeout: int | None = None) -> ApprovalRule:
        """Create a rule that any approver can satisfy."""
        return cls(required=True, approvers=None, timeout=timeout)

    @classmethod
    def specific_approvers(
        cls,
        approvers: list[str],
        *,
        require_all: bool = False,
        timeout: int | None = None,
    ) -> ApprovalRule:
        """Create a rule with specific approvers."""
        return cls(
            required=True,
            approvers=approvers,
            require_all=require_all,
            timeout=timeout,
        )

    def is_valid_approver(self, approver: str) -> bool:
        """Check if an approver is valid for this rule."""
        if self.approvers is None:
            return True  # Any approver allowed
        return approver in self.approvers


class MultiApprovalRequest(BaseModel):
    """Request for multi-level approval.

    Extends ApprovalRequest with tracking for multiple approvers.

    Attributes:
        rule: The approval rule being applied
        required_approvers: List of approvers still needed
        received_approvals: Approvals already received
        status: Current status of the multi-approval
    """

    # Base request fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    args: dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Multi-approval specific
    rule: ApprovalRule = Field(default_factory=ApprovalRule)
    required_approvers: list[str] = Field(default_factory=list)
    received_approvals: list[ApprovalResponse] = Field(default_factory=list)
    status: Literal["pending", "approved", "rejected", "partial"] = "pending"

    def to_approval_request(self) -> ApprovalRequest:
        """Convert to base ApprovalRequest for handlers."""
        return ApprovalRequest(
            id=self.id,
            tool_name=self.tool_name,
            args=self.args,
            context=self.context,
            timestamp=self.timestamp,
            timeout=self.rule.timeout or 300,
            metadata={
                **self.metadata,
                "approval_rule": self.rule.model_dump(),
                "required_approvers": self.required_approvers,
                "received_count": len(self.received_approvals),
            },
        )

    def add_approval(self, response: ApprovalResponse) -> MultiApprovalRequest:
        """Add an approval and update status.

        Returns:
            Updated MultiApprovalRequest (creates a new instance)
        """
        new_approvals = [*self.received_approvals, response]

        # If rejected, immediately set status
        if not response.approved:
            return self.model_copy(
                update={
                    "received_approvals": new_approvals,
                    "status": "rejected",
                }
            )

        # Check if we have enough approvals
        if self.rule.require_all:
            # Need all specified approvers
            if self.rule.approvers:
                approved_by = {r.approver for r in new_approvals if r.approved}
                if set(self.rule.approvers) <= approved_by:
                    new_status = "approved"
                else:
                    new_status = "partial"
            else:
                new_status = "approved"  # No specific approvers required
        else:
            # Any one approval is enough
            new_status = "approved"

        return self.model_copy(
            update={
                "received_approvals": new_approvals,
                "status": new_status,
            }
        )

    @property
    def is_complete(self) -> bool:
        """Check if approval process is complete."""
        return self.status in ("approved", "rejected")

    @property
    def is_approved(self) -> bool:
        """Check if fully approved."""
        return self.status == "approved"

    def get_pending_approvers(self) -> list[str]:
        """Get list of approvers who haven't yet approved."""
        if not self.rule.approvers:
            return []
        approved_by = {r.approver for r in self.received_approvals if r.approved}
        return [a for a in self.rule.approvers if a not in approved_by]

    def to_final_response(self) -> ApprovalResponse:
        """Convert to final ApprovalResponse after completion."""
        if not self.is_complete:
            raise ValueError("Cannot convert incomplete approval to response")

        if self.status == "approved":
            # Use the last modified_args if any
            modified = None
            for r in reversed(self.received_approvals):
                if r.modified_args:
                    modified = r.modified_args
                    break

            approvers = [r.approver for r in self.received_approvals if r.approved and r.approver]
            return ApprovalResponse.approve(
                modified_args=modified,
                reason=f"Approved by: {', '.join(approvers)}" if approvers else "Approved",
                approver=approvers[0] if approvers else None,
            )
        else:
            # Find rejection reason
            for r in self.received_approvals:
                if not r.approved:
                    return ApprovalResponse.reject(
                        reason=r.reason or "Rejected",
                        approver=r.approver,
                    )
            return ApprovalResponse.reject(reason="Rejected")


def create_rule_based_handler(
    rules: dict[str, ApprovalRule],
    handler: ApprovalHandler = console_approval_handler,
    default_rule: ApprovalRule | None = None,
) -> ApprovalHandler:
    """Create a handler that applies different rules per tool.

    Args:
        rules: Dict mapping tool names to ApprovalRule
        handler: Base handler to use for tools requiring approval
        default_rule: Rule to use for tools not in rules dict (default: no approval)

    Returns:
        A handler that applies appropriate rules

    Example:
        ```python
        from ai_infra.llm.tools import ApprovalRule, create_rule_based_handler

        rules = {
            "read_file": ApprovalRule.no_approval(),
            "delete_file": ApprovalRule.specific_approvers(["admin"]),
            "transfer_money": ApprovalRule.specific_approvers(
                ["cfo", "ceo"], require_all=True
            ),
        }

        handler = create_rule_based_handler(rules)
        agent = Agent(tools=[...], approval_handler=handler, require_approval=True)
        ```
    """

    def rule_based_handler(request: ApprovalRequest) -> ApprovalResponse:
        rule = rules.get(request.tool_name, default_rule or ApprovalRule.no_approval())

        if not rule.required:
            return ApprovalResponse.approve(
                reason="No approval required for this tool",
                approver="auto",
            )

        # For simple rules (no multi-approver), use the handler directly
        if not rule.require_all or not rule.approvers:
            response = handler(request)

            # Validate approver if specified
            if rule.approvers and response.approved:
                if response.approver and not rule.is_valid_approver(response.approver):
                    return ApprovalResponse.reject(
                        reason=f"Approver '{response.approver}' not authorized. "
                        f"Must be one of: {rule.approvers}",
                    )

            return response

        # For require_all rules, we need multiple approvals
        # In sync handler, we collect all approvals sequentially
        multi_request = MultiApprovalRequest(
            id=request.id,
            tool_name=request.tool_name,
            args=request.args,
            context=request.context,
            timestamp=request.timestamp,
            metadata=request.metadata,
            rule=rule,
            required_approvers=rule.approvers or [],
        )

        # Collect approvals from each required approver
        for approver in rule.approvers:
            print(f"\nApproval needed from: {approver}")
            sub_request = multi_request.to_approval_request()
            sub_request.metadata["current_approver"] = approver

            response = handler(sub_request)
            response = response.model_copy(update={"approver": approver})

            multi_request = multi_request.add_approval(response)

            if multi_request.status == "rejected":
                break

        return multi_request.to_final_response()

    return rule_based_handler
