#!/usr/bin/env python
"""Human-in-the-Loop (HITL) Approval Example.

This example demonstrates:
- Requiring human approval before tool execution
- Console-based approval handlers
- Custom async approval handlers (for web apps)
- Per-tool approval control
- Approval with argument modification

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio

from ai_infra import Agent
from ai_infra.llm.tools import (
    ApprovalRequest,
    ApprovalResponse,
    console_approval_handler,
    create_selective_handler,
)

# =============================================================================
# Sample Tools (Some Dangerous, Some Safe)
# =============================================================================


def get_balance(account_id: str) -> str:
    """Get account balance (safe operation).

    Args:
        account_id: The account identifier.

    Returns:
        Account balance information.
    """
    # Simulated balances
    balances = {
        "checking": "$5,234.56",
        "savings": "$12,500.00",
        "investment": "$45,678.90",
    }
    return balances.get(account_id, f"Account {account_id} not found")


def transfer_money(from_account: str, to_account: str, amount: float) -> str:
    """Transfer money between accounts (DANGEROUS - requires approval).

    Args:
        from_account: Source account ID.
        to_account: Destination account ID.
        amount: Amount to transfer.

    Returns:
        Confirmation of transfer.
    """
    return f"[OK] Transferred ${amount:.2f} from {from_account} to {to_account}"


def delete_account(account_id: str, confirm: bool = False) -> str:
    """Delete an account permanently (VERY DANGEROUS - requires approval).

    Args:
        account_id: Account to delete.
        confirm: Must be True to proceed.

    Returns:
        Confirmation or error message.
    """
    if not confirm:
        return "Error: confirm must be True to delete account"
    return f"[!] Account {account_id} has been permanently deleted"


def send_notification(user_id: str, message: str) -> str:
    """Send a notification to a user (safe operation).

    Args:
        user_id: User to notify.
        message: Notification message.

    Returns:
        Confirmation.
    """
    return f"[OK] Notification sent to user {user_id}"


# =============================================================================
# Basic Approval Examples
# =============================================================================


def all_tools_require_approval():
    """Require approval for ALL tool calls."""
    print("=" * 60)
    print("All Tools Require Approval")
    print("=" * 60)

    agent = Agent(
        tools=[get_balance, transfer_money, delete_account],
        require_approval=True,  # ALL tools need approval
        # Uses built-in console_approval_handler by default
    )

    print("\nNote: You will be prompted to approve each tool call.")
    print("Type 'y' to approve, 'n' to reject, or 'q' to quit.\n")

    # This will prompt for approval before calling get_balance
    result = agent.run("What's my checking account balance?")
    print(f"\nResult: {result}")


def selective_tool_approval():
    """Require approval only for specific dangerous tools."""
    print("\n" + "=" * 60)
    print("Selective Tool Approval")
    print("=" * 60)

    # Only these tools require approval
    dangerous_tools = ["transfer_money", "delete_account"]

    agent = Agent(
        tools=[get_balance, transfer_money, delete_account, send_notification],
        require_approval=dangerous_tools,  # Only specified tools need approval
    )

    print(f"\nDangerous tools requiring approval: {dangerous_tools}")
    print("Safe tools like 'get_balance' will execute immediately.\n")

    # get_balance executes without approval
    result = agent.run("Check my savings balance")
    print(f"Balance check (no approval needed): {result}")

    # transfer_money requires approval
    print("\n--- Now requesting a transfer (will need approval) ---")
    result = agent.run("Transfer $100 from checking to savings")
    print(f"Transfer result: {result}")


def dynamic_approval_function():
    """Use a function to dynamically decide if approval is needed."""
    print("\n" + "=" * 60)
    print("Dynamic Approval Function")
    print("=" * 60)

    def needs_approval(tool_name: str, args: dict) -> bool:
        """Dynamically decide if approval is needed based on tool and args."""
        # Always approve safe tools
        if tool_name == "get_balance":
            return False

        # Require approval for large transfers
        if tool_name == "transfer_money":
            amount = args.get("amount", 0)
            return amount > 100  # Only need approval for transfers > $100

        # Always require approval for deletions
        if tool_name == "delete_account":
            return True

        return False

    agent = Agent(
        tools=[get_balance, transfer_money, delete_account],
        require_approval=needs_approval,  # Function for dynamic approval
    )

    print("Rules:")
    print("  - get_balance: Never needs approval")
    print("  - transfer_money: Only if amount > $100")
    print("  - delete_account: Always needs approval")

    # Small transfer - no approval
    result = agent.run("Transfer $50 from checking to savings")
    print(f"\nSmall transfer ($50): {result}")

    # Large transfer - needs approval
    print("\n--- Large transfer will need approval ---")
    result = agent.run("Transfer $500 from checking to savings")
    print(f"Large transfer ($500): {result}")


# =============================================================================
# Custom Approval Handlers
# =============================================================================


def custom_sync_handler():
    """Use a custom synchronous approval handler."""
    print("\n" + "=" * 60)
    print("Custom Sync Approval Handler")
    print("=" * 60)

    def my_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
        """Custom approval handler with logging."""
        print("\n APPROVAL REQUIRED")
        print(f"   Tool: {request.tool_name}")
        print(f"   Args: {request.args}")
        print(f"   Request ID: {request.id}")

        # Could integrate with Slack, Discord, or any notification system here

        # For demo, auto-approve with modification
        if request.tool_name == "transfer_money":
            # Modify the amount to a safer value
            modified_args = dict(request.args)
            original = modified_args.get("amount", 0)
            modified_args["amount"] = min(original, 100)  # Cap at $100

            print(f"   [!] Modified amount from ${original} to ${modified_args['amount']}")

            return ApprovalResponse(
                approved=True,
                modified_args=modified_args,
                reason="Amount capped at $100 for security",
            )

        return ApprovalResponse(approved=True, reason="Auto-approved")

    agent = Agent(
        tools=[transfer_money],
        require_approval=True,
        approval_handler=my_approval_handler,
    )

    result = agent.run("Transfer $5000 from checking to savings")
    print(f"\nResult: {result}")


async def custom_async_handler():
    """Use a custom async approval handler (for web apps)."""
    print("\n" + "=" * 60)
    print("Custom Async Approval Handler (Web App Pattern)")
    print("=" * 60)

    # Simulated pending approvals (in real app, this would be a database)
    pending_approvals: dict[str, ApprovalRequest] = {}
    approval_responses: dict[str, ApprovalResponse] = {}

    async def web_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
        """
        Async approval handler for web applications.

        In a real app, this would:
        1. Store the request in a database
        2. Send a notification (WebSocket, email, Slack, etc.)
        3. Wait for the response (or timeout)
        """
        print("\n WEB APPROVAL REQUEST")
        print(f"   Request ID: {request.id}")
        print(f"   Tool: {request.tool_name}")
        print(f"   Args: {request.args}")

        # Simulate storing request
        pending_approvals[request.id] = request

        # Simulate sending WebSocket notification
        print("    Sending to frontend via WebSocket...")
        await asyncio.sleep(0.5)  # Simulate network delay

        # Simulate receiving approval (in real app, would wait for user response)
        print("    Received approval from user via WebSocket")

        # Simulate approval
        approval_responses[request.id] = ApprovalResponse(
            approved=True,
            reason="Approved by admin user",
            approver="admin@example.com",
        )

        return approval_responses[request.id]

    agent = Agent(
        tools=[transfer_money, delete_account],
        require_approval=True,
        approval_handler=web_approval_handler,  # Async handler
    )

    result = await agent.arun("Transfer $200 from checking to savings")
    print(f"\nResult: {result}")


def selective_handler_example():
    """Use create_selective_handler for per-tool handlers."""
    print("\n" + "=" * 60)
    print("Selective Handler (Per-Tool Handlers)")
    print("=" * 60)

    def handle_transfer(request: ApprovalRequest) -> ApprovalResponse:
        """Custom handler for transfers - always approve small amounts."""
        amount = request.args.get("amount", 0)
        if amount <= 100:
            return ApprovalResponse(approved=True, reason="Small amount auto-approved")
        return console_approval_handler(request)  # Manual approval for large

    def handle_delete(request: ApprovalRequest) -> ApprovalResponse:
        """Custom handler for deletes - always require manual approval."""
        print("\n[!]  DELETE OPERATION REQUIRES MANUAL APPROVAL [!]")
        return console_approval_handler(request)

    # Create handler that routes to different handlers per tool
    handler = create_selective_handler(
        {
            "transfer_money": handle_transfer,
            "delete_account": handle_delete,
        }
    )

    agent = Agent(
        tools=[get_balance, transfer_money, delete_account],
        require_approval=["transfer_money", "delete_account"],
        approval_handler=handler,
    )

    # Small transfer - auto-approved
    result = agent.run("Transfer $50 from checking to savings")
    print(f"\nSmall transfer: {result}")

    # Large transfer - needs manual approval
    print("\n--- Large transfer needs manual approval ---")
    result = agent.run("Transfer $500 from checking to savings")
    print(f"Large transfer: {result}")


# =============================================================================
# Rejection and Modification Examples
# =============================================================================


def approval_rejection_example():
    """Handle approval rejections gracefully."""
    print("\n" + "=" * 60)
    print("Handling Approval Rejections")
    print("=" * 60)

    def strict_handler(request: ApprovalRequest) -> ApprovalResponse:
        """Reject all deletion operations."""
        if request.tool_name == "delete_account":
            return ApprovalResponse(
                approved=False,
                reason="Account deletion is disabled for security",
            )
        return ApprovalResponse(approved=True)

    agent = Agent(
        tools=[delete_account, get_balance],
        require_approval=True,
        approval_handler=strict_handler,
    )

    # Deletion will be rejected
    result = agent.run("Delete the checking account")
    print(f"\nResult after rejection: {result}")


def argument_modification_example():
    """Modify tool arguments during approval."""
    print("\n" + "=" * 60)
    print("Argument Modification During Approval")
    print("=" * 60)

    def sanitizing_handler(request: ApprovalRequest) -> ApprovalResponse:
        """Sanitize and modify arguments before approval."""
        if request.tool_name == "send_notification":
            # Sanitize the message
            original_message = request.args.get("message", "")
            sanitized = original_message.replace("password", "****")

            if original_message != sanitized:
                print("\n[!] Sanitized message content")
                return ApprovalResponse(
                    approved=True,
                    modified_args={
                        **request.args,
                        "message": sanitized,
                    },
                    reason="Sensitive content was sanitized",
                )

        return ApprovalResponse(approved=True)

    agent = Agent(
        tools=[send_notification],
        require_approval=True,
        approval_handler=sanitizing_handler,
    )

    result = agent.run("Send a notification to user123 saying 'Your password reset code is 12345'")
    print(f"\nResult (sanitized): {result}")


if __name__ == "__main__":
    print("HITL Examples - Some require interactive input\n")

    # Non-interactive examples
    dynamic_approval_function()
    custom_sync_handler()
    asyncio.run(custom_async_handler())
    approval_rejection_example()
    argument_modification_example()

    # Interactive examples (uncomment to try)
    # all_tools_require_approval()
    # selective_tool_approval()
    # selective_handler_example()
