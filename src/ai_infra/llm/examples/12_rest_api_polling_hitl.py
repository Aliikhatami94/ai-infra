"""12_rest_api_polling_hitl: REST API HITL with polling example.

This example demonstrates how to integrate Human-in-the-Loop (HITL) approval
with a REST API using polling for approval requests.

This pattern is useful when:
- WebSockets are not available
- You need a simpler integration
- You're building mobile apps or CLI tools that poll for updates

Usage:
    # Install dependencies
    pip install fastapi uvicorn

    # Run the server
    python -m ai_infra.llm.examples.12_rest_api_polling_hitl

    # Or with uvicorn
    uvicorn ai_infra.llm.examples.12_rest_api_polling_hitl:app --reload

Key features demonstrated:
- Async polling-based approval handler
- Session management with pause/resume workflow
- REST API endpoints for approval workflow
- Frontend polling JavaScript client
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ai_infra.llm import Agent
from ai_infra.llm.session import memory

# Try to import FastAPI and related dependencies
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError:
    raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")


# ============================================================================
# Models
# ============================================================================


class TaskStatus(str, Enum):
    """Status of an agent task."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class CreateTaskRequest(BaseModel):
    """Request to create a new agent task."""

    message: str = Field(..., description="The user message to process")


class ApprovalRequest(BaseModel):
    """An approval request waiting for user decision."""

    id: str
    task_id: str
    tool_name: str
    tool_args: dict[str, Any]
    context: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime
    expires_at: datetime


class ApprovalDecision(BaseModel):
    """User's decision on an approval request."""

    approved: bool
    reason: str | None = None
    modified_args: dict[str, Any] | None = None


class TaskResponse(BaseModel):
    """Response for a task."""

    id: str
    status: TaskStatus
    message: str | None = None
    result: str | None = None
    pending_approvals: list[str] = []
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Application State
# ============================================================================

app = FastAPI(title="HITL Polling Demo")

# Store tasks by ID
tasks: dict[str, TaskResponse] = {}

# Store pending approval requests
pending_approvals: dict[str, ApprovalRequest] = {}

# Store completion events for tasks
task_events: dict[str, asyncio.Event] = {}

# Store approval decisions
approval_decisions: dict[str, ApprovalDecision | None] = {}


# ============================================================================
# Example Tools
# ============================================================================


def delete_file(filename: str) -> str:
    """Delete a file from the system (simulated).

    Args:
        filename: Name of the file to delete

    Returns:
        Confirmation message
    """
    return f"Successfully deleted file: {filename}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (simulated).

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body

    Returns:
        Confirmation message
    """
    return f"Email sent to {to} with subject: {subject}"


def make_payment(amount: float, recipient: str) -> str:
    """Make a payment (simulated).

    Args:
        amount: Amount to pay
        recipient: Payment recipient

    Returns:
        Confirmation message
    """
    return f"Payment of ${amount:.2f} sent to {recipient}"


# ============================================================================
# Background Task Runner
# ============================================================================


async def run_agent_task(task_id: str, message: str):
    """Run an agent task in the background.

    Uses pause/resume workflow for HITL approvals.
    """
    # Create agent with pause_before for dangerous tools
    agent = Agent(
        tools=[delete_file, send_email, make_payment],
        session=memory(),
        pause_before=["delete_file", "send_email", "make_payment"],
    )

    session_id = task_id

    try:
        # Update task status
        tasks[task_id].status = TaskStatus.RUNNING
        tasks[task_id].updated_at = datetime.now()

        # Run the agent
        result = await agent.arun(message, session_id=session_id)

        if isinstance(result, str):
            tasks[task_id].status = TaskStatus.COMPLETED
            tasks[task_id].updated_at = datetime.now()
            tasks[task_id].result = result
            return

        from ai_infra.llm.session import SessionResult

        # Check if paused (awaiting approval)
        while isinstance(result, SessionResult) and result.paused:
            # Task is paused - create approval request
            tasks[task_id].status = TaskStatus.AWAITING_APPROVAL
            tasks[task_id].updated_at = datetime.now()

            pending_action = result.pending_action
            approval_id = str(uuid.uuid4())

            # Create approval request
            # pending_action fields may be None in edge cases
            tool_name = pending_action.tool_name if pending_action else "unknown"
            tool_args = pending_action.args if pending_action else {}
            context_str = (
                str(pending_action.context) if pending_action and pending_action.context else ""
            )
            approval = ApprovalRequest(
                id=approval_id,
                task_id=task_id,
                tool_name=tool_name or "unknown",
                tool_args=tool_args,
                context={"messages": context_str},
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=5),
            )
            pending_approvals[approval_id] = approval
            tasks[task_id].pending_approvals.append(approval_id)

            # Wait for decision
            while approval.status == ApprovalStatus.PENDING:
                await asyncio.sleep(0.5)

                # Check if expired
                if datetime.now() > approval.expires_at:
                    approval.status = ApprovalStatus.EXPIRED
                    break

                # Check if decision made
                decision = approval_decisions.get(approval_id)
                if decision is not None:
                    approval.status = (
                        ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
                    )
                    break

            # Get the decision
            decision = approval_decisions.get(approval_id)
            approved = decision.approved if decision else False

            # Update task status
            tasks[task_id].status = TaskStatus.RUNNING
            tasks[task_id].updated_at = datetime.now()

            # Resume the agent
            result = await agent.aresume(
                session_id=session_id,
                approved=approved,
                modified_args=decision.modified_args if decision else None,
                reason=decision.reason if decision else "Expired or rejected",
            )

        # Task completed
        response_text = result.content if isinstance(result, SessionResult) else str(result)
        tasks[task_id].status = TaskStatus.COMPLETED
        tasks[task_id].result = response_text
        tasks[task_id].updated_at = datetime.now()

    except Exception as e:
        tasks[task_id].status = TaskStatus.FAILED
        tasks[task_id].result = f"Error: {e!s}"
        tasks[task_id].updated_at = datetime.now()

    finally:
        # Signal completion
        event = task_events.get(task_id)
        if event:
            event.set()


# ============================================================================
# API Routes
# ============================================================================


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
):
    """Create a new agent task.

    The task runs in the background. Poll GET /api/tasks/{task_id} for status.
    """
    task_id = str(uuid.uuid4())
    now = datetime.now()

    task = TaskResponse(
        id=task_id,
        status=TaskStatus.PENDING,
        message=request.message,
        created_at=now,
        updated_at=now,
    )
    tasks[task_id] = task
    task_events[task_id] = asyncio.Event()

    # Run agent in background
    background_tasks.add_task(run_agent_task, task_id, request.message)

    return task


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get the status of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/tasks/{task_id}/approvals", response_model=list[ApprovalRequest])
async def get_task_approvals(task_id: str):
    """Get pending approval requests for a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return [
        pending_approvals[aid]
        for aid in tasks[task_id].pending_approvals
        if aid in pending_approvals and pending_approvals[aid].status == ApprovalStatus.PENDING
    ]


@app.get("/api/approvals", response_model=list[ApprovalRequest])
async def get_pending_approvals():
    """Get all pending approval requests."""
    return [
        approval
        for approval in pending_approvals.values()
        if approval.status == ApprovalStatus.PENDING
    ]


@app.post("/api/approvals/{approval_id}", response_model=ApprovalRequest)
async def submit_approval(approval_id: str, decision: ApprovalDecision):
    """Submit an approval decision."""
    if approval_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")

    approval = pending_approvals[approval_id]
    if approval.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=400, detail="Approval already processed")

    # Store decision
    approval_decisions[approval_id] = decision

    # Update status (will be picked up by the background task)
    approval.status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED

    return approval


# ============================================================================
# Frontend
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML page."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HITL Polling Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
        .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
        .status { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
        .status-pending { background: #fff3e0; }
        .status-running { background: #e3f2fd; }
        .status-awaiting_approval { background: #ffebee; }
        .status-completed { background: #e8f5e9; }
        .status-failed { background: #ffebee; }
        .approval-card { background: #fff8e1; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; border: none; border-radius: 3px; }
        .approve-btn { background: #4caf50; color: white; }
        .reject-btn { background: #f44336; color: white; }
        .submit-btn { background: #2196f3; color: white; }
        input[type="text"] { width: 70%; padding: 10px; margin-right: 10px; }
        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
        #approvals { margin-top: 20px; }
        h2 { margin-top: 30px; }
    </style>
</head>
<body>
    <h1>🤖 HITL Polling Demo</h1>
    <p>This demo uses REST API polling for Human-in-the-Loop approvals.</p>

    <h2>Create Task</h2>
    <form id="task-form">
        <input type="text" id="message" placeholder="Ask the agent something..." autocomplete="off">
        <button type="submit" class="submit-btn">Create Task</button>
    </form>

    <h2>Active Tasks</h2>
    <div id="tasks"></div>

    <h2>Pending Approvals</h2>
    <div id="approvals"></div>

    <script>
        let pollInterval;

        // Create task
        document.getElementById('task-form').onsubmit = async (e) => {
            e.preventDefault();
            const message = document.getElementById('message').value.trim();
            if (!message) return;

            try {
                const response = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const task = await response.json();
                document.getElementById('message').value = '';
                updateUI();
            } catch (error) {
                console.error('Error creating task:', error);
            }
        };

        // Submit approval decision
        async function submitApproval(approvalId, approved) {
            try {
                await fetch(`/api/approvals/${approvalId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        approved,
                        reason: approved ? 'User approved' : 'User rejected'
                    })
                });
                updateUI();
            } catch (error) {
                console.error('Error submitting approval:', error);
            }
        }

        // Update UI
        async function updateUI() {
            // Fetch all pending approvals
            const approvalsResponse = await fetch('/api/approvals');
            const approvals = await approvalsResponse.json();

            // Render approvals
            const approvalsDiv = document.getElementById('approvals');
            if (approvals.length === 0) {
                approvalsDiv.innerHTML = '<p>No pending approvals</p>';
            } else {
                approvalsDiv.innerHTML = approvals.map(a => `
                    <div class="card approval-card">
                        <strong>🔐 Approval Required</strong><br>
                        <p>Task: ${a.task_id.substring(0, 8)}...</p>
                        <p>Tool: <code>${a.tool_name}</code></p>
                        <pre>${JSON.stringify(a.tool_args, null, 2)}</pre>
                        <button class="approve-btn" onclick="submitApproval('${a.id}', true)">✓ Approve</button>
                        <button class="reject-btn" onclick="submitApproval('${a.id}', false)">✗ Reject</button>
                    </div>
                `).join('');
            }

            // Re-render tasks (fetch each task)
            // Note: In production you'd have a /api/tasks endpoint that returns all tasks
        }

        // Poll for updates
        function startPolling() {
            pollInterval = setInterval(updateUI, 2000);
        }

        // Initial load
        updateUI();
        startPolling();
    </script>
</body>
</html>
"""


# ============================================================================
# Run the server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting HITL Polling Demo at http://localhost:8000")
    print("Open in your browser and try creating tasks.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
