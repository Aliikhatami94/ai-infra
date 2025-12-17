"""Tests for approval-based HITL (Human-in-the-Loop)."""

from datetime import datetime

import pytest

from ai_infra.llm.tools.approval import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalRule,
    MultiApprovalRequest,
    OutputReviewRequest,
    OutputReviewResponse,
    auto_approve_handler,
    auto_reject_handler,
    create_rule_based_handler,
    create_selective_handler,
)
from ai_infra.llm.tools.hitl import ApprovalConfig, wrap_tool_for_approval


class TestApprovalRequest:
    """Test ApprovalRequest model."""

    def test_default_values(self):
        req = ApprovalRequest(tool_name="my_tool", args={"x": 1})
        assert req.tool_name == "my_tool"
        assert req.args == {"x": 1}
        assert req.id is not None  # Auto-generated
        assert req.timeout == 300  # Default 5 minutes
        assert req.context == {}
        assert req.metadata == {}
        assert isinstance(req.timestamp, datetime)

    def test_custom_values(self):
        req = ApprovalRequest(
            tool_name="my_tool",
            args={"x": 1},
            timeout=60,
            context={"conversation": "hello"},
            metadata={"user_id": "123"},
        )
        assert req.timeout == 60
        assert req.context == {"conversation": "hello"}
        assert req.metadata == {"user_id": "123"}

    def test_to_console_prompt(self):
        req = ApprovalRequest(tool_name="delete_file", args={"path": "/tmp/test.txt"})
        prompt = req.to_console_prompt()
        assert "delete_file" in prompt
        assert "/tmp/test.txt" in prompt


class TestApprovalResponse:
    """Test ApprovalResponse model."""

    def test_approve_factory(self):
        resp = ApprovalResponse.approve(reason="Looks good")
        assert resp.approved is True
        assert resp.reason == "Looks good"
        assert resp.modified_args is None

    def test_reject_factory(self):
        resp = ApprovalResponse.reject(reason="Too dangerous")
        assert resp.approved is False
        assert resp.reason == "Too dangerous"

    def test_approve_with_modified_args(self):
        resp = ApprovalResponse.approve(
            modified_args={"amount": 100},
            reason="Reduced amount",
        )
        assert resp.approved is True
        assert resp.modified_args == {"amount": 100}


class TestOutputReviewModels:
    """Test output review models."""

    def test_output_review_request(self):
        req = OutputReviewRequest(output="Hello world!")
        assert req.output == "Hello world!"
        assert req.id is not None

    def test_output_review_response_allow(self):
        resp = OutputReviewResponse.allow()
        assert resp.action == "pass"

    def test_output_review_response_modify(self):
        resp = OutputReviewResponse.modify("Modified text", reason="PII removed")
        assert resp.action == "modify"
        assert resp.replacement == "Modified text"
        assert resp.reason == "PII removed"

    def test_output_review_response_block(self):
        resp = OutputReviewResponse.block(reason="Policy violation")
        assert resp.action == "block"
        assert resp.replacement == "[Content blocked by reviewer]"


class TestBuiltInHandlers:
    """Test built-in approval handlers."""

    def test_auto_approve_handler(self):
        req = ApprovalRequest(tool_name="test", args={})
        resp = auto_approve_handler(req)
        assert resp.approved is True
        assert resp.approver == "auto"

    def test_auto_reject_handler(self):
        req = ApprovalRequest(tool_name="test", args={})
        resp = auto_reject_handler(req)
        assert resp.approved is False
        assert resp.approver == "auto"

    def test_selective_handler(self):
        handler = create_selective_handler(
            tools_requiring_approval=["dangerous_tool"],
            handler=auto_reject_handler,  # Use auto_reject for testing
        )

        # Tool not in list -> auto-approve
        req1 = ApprovalRequest(tool_name="safe_tool", args={})
        resp1 = handler(req1)
        assert resp1.approved is True

        # Tool in list -> uses provided handler (auto_reject)
        req2 = ApprovalRequest(tool_name="dangerous_tool", args={})
        resp2 = handler(req2)
        assert resp2.approved is False


class TestApprovalConfig:
    """Test ApprovalConfig dataclass."""

    def test_default_no_approval(self):
        config = ApprovalConfig()
        assert config.require_approval is False
        assert config.needs_approval("any_tool") is False

    def test_require_all_approval(self):
        config = ApprovalConfig(require_approval=True, auto_approve=True)
        assert config.needs_approval("any_tool") is False  # auto_approve overrides

    def test_require_specific_tools(self):
        config = ApprovalConfig(require_approval=["tool_a", "tool_b"])
        assert config.needs_approval("tool_a") is True
        assert config.needs_approval("tool_b") is True
        assert config.needs_approval("tool_c") is False

    def test_auto_sets_console_handler(self):
        config = ApprovalConfig(require_approval=True)
        # Should have set console_approval_handler
        assert config.approval_handler is not None

    @pytest.mark.asyncio
    async def test_request_approval_auto_approve(self):
        config = ApprovalConfig(auto_approve=True)
        req = ApprovalRequest(tool_name="test", args={})
        resp = await config.request_approval(req)
        assert resp.approved is True
        assert resp.approver == "auto"

    @pytest.mark.asyncio
    async def test_request_approval_with_handler(self):
        config = ApprovalConfig(
            require_approval=True,
            approval_handler=auto_reject_handler,
        )
        req = ApprovalRequest(tool_name="test", args={})
        resp = await config.request_approval(req)
        assert resp.approved is False

    @pytest.mark.asyncio
    async def test_request_approval_with_async_handler(self):
        async def async_handler(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse.approve(reason="Async approved")

        config = ApprovalConfig(
            require_approval=True,
            approval_handler_async=async_handler,
        )
        req = ApprovalRequest(tool_name="test", args={})
        resp = await config.request_approval(req)
        assert resp.approved is True
        assert resp.reason == "Async approved"

    def test_callable_approval_condition(self):
        """Test that require_approval can be a callable."""

        # Approve based on tool name and args
        def condition(tool_name: str, args: dict) -> bool:
            # Only require approval for dangerous operations
            if tool_name == "delete_file":
                return True
            if tool_name == "transfer_money" and args.get("amount", 0) > 1000:
                return True
            return False

        config = ApprovalConfig(require_approval=condition)

        # delete_file always needs approval
        assert config.needs_approval("delete_file") is True
        assert config.needs_approval("delete_file", {"path": "/tmp"}) is True

        # transfer_money depends on amount
        assert config.needs_approval("transfer_money", {"amount": 500}) is False
        assert config.needs_approval("transfer_money", {"amount": 2000}) is True

        # Other tools don't need approval
        assert config.needs_approval("read_file") is False

    def test_callable_with_empty_args(self):
        """Test callable condition with no args provided."""

        def condition(tool_name: str, args: dict) -> bool:
            return tool_name.startswith("dangerous_")

        config = ApprovalConfig(require_approval=condition)
        # Should work without args
        assert config.needs_approval("dangerous_action") is True
        assert config.needs_approval("safe_action") is False


class TestWrapToolForApproval:
    """Test wrap_tool_for_approval function."""

    def test_no_config_returns_original(self):
        def my_tool(x: int) -> str:
            """A tool."""
            return str(x)

        result = wrap_tool_for_approval(my_tool, None)
        assert result is my_tool

    @pytest.mark.asyncio
    async def test_tool_not_requiring_approval(self):
        def my_tool(x: int) -> str:
            """A tool that doesn't need approval."""
            return f"Result: {x}"

        config = ApprovalConfig(require_approval=["other_tool"])  # Not this tool
        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        assert result == "Result: 5"

    @pytest.mark.asyncio
    async def test_tool_approved(self):
        def my_tool(x: int) -> str:
            """A tool that needs approval."""
            return f"Result: {x}"

        config = ApprovalConfig(
            require_approval=True,
            approval_handler=auto_approve_handler,
        )
        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        assert result == "Result: 5"

    @pytest.mark.asyncio
    async def test_tool_rejected(self):
        def my_tool(x: int) -> str:
            """A tool that needs approval."""
            return f"Result: {x}"

        config = ApprovalConfig(
            require_approval=True,
            approval_handler=auto_reject_handler,
        )
        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        assert "[Tool call rejected:" in result

    @pytest.mark.asyncio
    async def test_tool_with_modified_args(self):
        def my_tool(x: int) -> str:
            """A tool that can have args modified."""
            return f"Result: {x}"

        def modifying_handler(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse.approve(
                modified_args={"x": 100},
                reason="Modified x",
            )

        config = ApprovalConfig(
            require_approval=True,
            approval_handler=modifying_handler,
        )
        wrapped = wrap_tool_for_approval(my_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        # Should use modified args (x=100) instead of original (x=5)
        assert result == "Result: 100"

    @pytest.mark.asyncio
    async def test_tool_with_callable_condition(self):
        """Test wrapping a tool with callable approval condition."""

        def my_tool(amount: int) -> str:
            """Transfer money."""
            return f"Transferred: {amount}"

        # Require approval only for large amounts
        def condition(tool_name: str, args: dict) -> bool:
            return args.get("amount", 0) > 100

        config = ApprovalConfig(
            require_approval=condition,
            approval_handler=auto_reject_handler,  # Reject if approval required
        )
        wrapped = wrap_tool_for_approval(my_tool, config)

        # Small amount - no approval needed, executes
        result = await wrapped.ainvoke({"amount": 50})
        assert result == "Transferred: 50"

        # Large amount - approval needed, rejected
        result = await wrapped.ainvoke({"amount": 200})
        assert "[Tool call rejected:" in result


class TestApprovalRule:
    """Test ApprovalRule model."""

    def test_default_rule(self):
        rule = ApprovalRule()
        assert rule.required is True
        assert rule.approvers is None
        assert rule.require_all is False

    def test_no_approval_factory(self):
        rule = ApprovalRule.no_approval()
        assert rule.required is False

    def test_any_approver_factory(self):
        rule = ApprovalRule.any_approver(timeout=60)
        assert rule.required is True
        assert rule.approvers is None
        assert rule.timeout == 60

    def test_specific_approvers_factory(self):
        rule = ApprovalRule.specific_approvers(["admin", "manager"])
        assert rule.required is True
        assert rule.approvers == ["admin", "manager"]
        assert rule.require_all is False

    def test_specific_approvers_require_all(self):
        rule = ApprovalRule.specific_approvers(
            ["admin", "legal", "cfo"],
            require_all=True,
        )
        assert rule.require_all is True
        assert len(rule.approvers) == 3

    def test_is_valid_approver(self):
        rule = ApprovalRule.specific_approvers(["admin", "manager"])
        assert rule.is_valid_approver("admin") is True
        assert rule.is_valid_approver("manager") is True
        assert rule.is_valid_approver("user") is False

    def test_is_valid_approver_any(self):
        rule = ApprovalRule.any_approver()
        assert rule.is_valid_approver("anyone") is True


class TestMultiApprovalRequest:
    """Test MultiApprovalRequest model."""

    def test_creation(self):
        req = MultiApprovalRequest(
            tool_name="delete_all",
            args={"confirm": True},
            rule=ApprovalRule.specific_approvers(["admin", "cto"], require_all=True),
        )
        assert req.status == "pending"
        assert req.is_complete is False
        assert req.get_pending_approvers() == ["admin", "cto"]

    def test_add_approval(self):
        req = MultiApprovalRequest(
            tool_name="delete_all",
            args={},
            rule=ApprovalRule.specific_approvers(["admin", "cto"], require_all=True),
        )

        # First approval
        approval1 = ApprovalResponse.approve(approver="admin")
        req = req.add_approval(approval1)
        assert req.status == "partial"
        assert req.is_complete is False
        assert req.get_pending_approvers() == ["cto"]

        # Second approval
        approval2 = ApprovalResponse.approve(approver="cto")
        req = req.add_approval(approval2)
        assert req.status == "approved"
        assert req.is_complete is True
        assert req.is_approved is True

    def test_add_rejection(self):
        req = MultiApprovalRequest(
            tool_name="delete_all",
            args={},
            rule=ApprovalRule.specific_approvers(["admin", "cto"], require_all=True),
        )

        # First approval
        approval1 = ApprovalResponse.approve(approver="admin")
        req = req.add_approval(approval1)

        # Second rejection
        rejection = ApprovalResponse.reject(reason="Too risky", approver="cto")
        req = req.add_approval(rejection)
        assert req.status == "rejected"
        assert req.is_complete is True
        assert req.is_approved is False

    def test_to_final_response_approved(self):
        req = MultiApprovalRequest(
            tool_name="transfer",
            args={"amount": 1000},
            rule=ApprovalRule.any_approver(),
        )
        approval = ApprovalResponse.approve(approver="manager")
        req = req.add_approval(approval)

        final = req.to_final_response()
        assert final.approved is True
        assert "manager" in (final.reason or "")

    def test_to_final_response_rejected(self):
        req = MultiApprovalRequest(
            tool_name="delete",
            args={},
            rule=ApprovalRule.any_approver(),
        )
        rejection = ApprovalResponse.reject(reason="Not safe", approver="admin")
        req = req.add_approval(rejection)

        final = req.to_final_response()
        assert final.approved is False
        assert final.reason == "Not safe"


class TestRuleBasedHandler:
    """Test create_rule_based_handler."""

    def test_no_approval_rule(self):
        rules = {
            "read_file": ApprovalRule.no_approval(),
            "delete_file": ApprovalRule.any_approver(),
        }
        handler = create_rule_based_handler(rules)

        req = ApprovalRequest(tool_name="read_file", args={})
        resp = handler(req)
        assert resp.approved is True
        assert "No approval required" in resp.reason

    def test_any_approver_rule(self):
        rules = {
            "delete_file": ApprovalRule.any_approver(),
        }
        handler = create_rule_based_handler(rules, handler=auto_approve_handler)

        req = ApprovalRequest(tool_name="delete_file", args={})
        resp = handler(req)
        assert resp.approved is True

    def test_default_rule_no_approval(self):
        # No rules, default = no approval
        handler = create_rule_based_handler({})

        req = ApprovalRequest(tool_name="unknown_tool", args={})
        resp = handler(req)
        assert resp.approved is True  # Default: no approval required

    def test_specific_approver_validation(self):
        rules = {
            "admin_action": ApprovalRule.specific_approvers(["admin"]),
        }

        # Handler that claims to be "user" not "admin"
        def user_handler(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse.approve(approver="user")

        handler = create_rule_based_handler(rules, handler=user_handler)

        req = ApprovalRequest(tool_name="admin_action", args={})
        resp = handler(req)
        # Should be rejected because "user" is not in approved list
        assert resp.approved is False
        assert "not authorized" in resp.reason
