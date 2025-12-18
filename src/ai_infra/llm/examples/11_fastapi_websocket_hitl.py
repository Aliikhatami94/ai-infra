"""11_fastapi_websocket_hitl: FastAPI WebSocket HITL example.

This example demonstrates how to integrate Human-in-the-Loop (HITL) approval
with a FastAPI web application using WebSockets for real-time communication.

Usage:
    # Install dependencies
    pip install fastapi uvicorn websockets

    # Run the server
    python -m ai_infra.llm.examples.11_fastapi_websocket_hitl

    # Or with uvicorn
    uvicorn ai_infra.llm.examples.11_fastapi_websocket_hitl:app --reload

Then open http://localhost:8000 in your browser.

Key features demonstrated:
- Async WebSocket approval handler
- Real-time tool call approval/rejection
- Session management with pause/resume
- Frontend JavaScript client
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from ai_infra.llm import Agent, ApprovalRequest, ApprovalResponse
from ai_infra.llm.session import memory

# Try to import FastAPI and related dependencies
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:
    raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn websockets")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Application State
# ============================================================================

app = FastAPI(title="HITL WebSocket Demo")

# Store active WebSocket connections by session ID
active_connections: dict[str, WebSocket] = {}

# Store pending approval requests
pending_approvals: dict[str, asyncio.Future[ApprovalResponse]] = {}


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
    # Simulated - in production this would actually delete files
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
    # Simulated - in production this would send email
    return f"Email sent to {to} with subject: {subject}"


def make_payment(amount: float, recipient: str) -> str:
    """Make a payment (simulated).

    Args:
        amount: Amount to pay
        recipient: Payment recipient

    Returns:
        Confirmation message
    """
    # Simulated - in production this would process payment
    return f"Payment of ${amount:.2f} sent to {recipient}"


# ============================================================================
# WebSocket HITL Handler
# ============================================================================


async def websocket_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
    """Handle tool approval requests via WebSocket.

    This handler:
    1. Sends the approval request to the connected WebSocket client
    2. Waits for the user's response
    3. Returns the approval decision

    Args:
        request: The approval request from the agent

    Returns:
        The approval response from the user
    """
    session_id = request.metadata.get("session_id", "default")

    # Get the WebSocket for this session
    websocket = active_connections.get(session_id)
    if not websocket:
        logger.warning(f"No WebSocket connection for session {session_id}, auto-rejecting")
        return ApprovalResponse(
            approved=False,
            reason="No active WebSocket connection",
        )

    # Create a future to wait for the response
    approval_id = str(uuid.uuid4())
    future: asyncio.Future[ApprovalResponse] = asyncio.Future()
    pending_approvals[approval_id] = future

    try:
        # Send approval request to frontend
        await websocket.send_json(
            {
                "type": "approval_required",
                "approval_id": approval_id,
                "tool_name": request.tool_name,
                "tool_args": request.args,
                "context": request.context,
                "request_id": request.id,
            }
        )

        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(future, timeout=request.timeout)
            return response
        except TimeoutError:
            logger.warning(f"Approval timeout for {request.tool_name}")
            return ApprovalResponse(
                approved=False,
                reason="Approval request timed out",
            )
    finally:
        # Clean up
        pending_approvals.pop(approval_id, None)


# ============================================================================
# Routes
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML page."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HITL WebSocket Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; height: 300px; overflow-y: auto; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        .system { background: #fff3e0; font-style: italic; }
        .approval-dialog { background: #ffebee; padding: 20px; border-radius: 5px; margin: 10px 0; }
        .approval-dialog button { margin: 5px; padding: 10px 20px; cursor: pointer; }
        .approve-btn { background: #4caf50; color: white; border: none; border-radius: 3px; }
        .reject-btn { background: #f44336; color: white; border: none; border-radius: 3px; }
        input[type="text"] { width: 70%; padding: 10px; }
        button[type="submit"] { padding: 10px 20px; }
        #status { padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .connected { background: #e8f5e9; }
        .disconnected { background: #ffebee; }
    </style>
</head>
<body>
    <h1>🤖 HITL WebSocket Demo</h1>
    <p>Try asking the agent to perform sensitive operations like deleting files, sending emails, or making payments.</p>

    <div id="status" class="disconnected">Disconnected</div>

    <div class="chat-container" id="chat"></div>

    <form id="chat-form">
        <input type="text" id="message" placeholder="Ask the agent something..." autocomplete="off">
        <button type="submit">Send</button>
    </form>

    <h3>Example prompts:</h3>
    <ul>
        <li>"Delete the file report.pdf"</li>
        <li>"Send an email to bob@example.com about the meeting"</li>
        <li>"Make a payment of $100 to Alice"</li>
    </ul>

    <script>
        const sessionId = 'session-' + Math.random().toString(36).substr(2, 9);
        let ws;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected - Reconnecting...';
                document.getElementById('status').className = 'disconnected';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            const chat = document.getElementById('chat');

            if (data.type === 'approval_required') {
                // Show approval dialog
                const dialog = document.createElement('div');
                dialog.className = 'approval-dialog';
                dialog.id = 'approval-' + data.approval_id;
                dialog.innerHTML = `
                    <strong>🔐 Approval Required</strong><br><br>
                    The agent wants to call: <code>${data.tool_name}</code><br>
                    <pre>${JSON.stringify(data.tool_args, null, 2)}</pre>
                    <button class="approve-btn" onclick="respond('${data.approval_id}', true)">✓ Approve</button>
                    <button class="reject-btn" onclick="respond('${data.approval_id}', false)">✗ Reject</button>
                `;
                chat.appendChild(dialog);
                chat.scrollTop = chat.scrollHeight;
            } else if (data.type === 'assistant') {
                addMessage('assistant', data.content);
            } else if (data.type === 'system') {
                addMessage('system', data.content);
            } else if (data.type === 'approval_result') {
                const dialog = document.getElementById('approval-' + data.approval_id);
                if (dialog) {
                    dialog.innerHTML = `<strong>${data.approved ? '✓ Approved' : '✗ Rejected'}</strong>: ${data.tool_name}`;
                    dialog.style.background = data.approved ? '#e8f5e9' : '#ffebee';
                }
            }
        }

        function addMessage(type, content) {
            const chat = document.getElementById('chat');
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.textContent = content;
            chat.appendChild(msg);
            chat.scrollTop = chat.scrollHeight;
        }

        function respond(approvalId, approved) {
            ws.send(JSON.stringify({
                type: 'approval_response',
                approval_id: approvalId,
                approved: approved,
                reason: approved ? 'User approved' : 'User rejected'
            }));
        }

        document.getElementById('chat-form').onsubmit = (e) => {
            e.preventDefault();
            const input = document.getElementById('message');
            const message = input.value.trim();
            if (message && ws.readyState === WebSocket.OPEN) {
                addMessage('user', message);
                ws.send(JSON.stringify({
                    type: 'user_message',
                    content: message
                }));
                input.value = '';
            }
        };

        connect();
    </script>
</body>
</html>
"""


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connections for HITL communication."""
    await websocket.accept()
    active_connections[session_id] = websocket

    logger.info(f"WebSocket connected: {session_id}")

    # Create agent with WebSocket approval handler
    agent = Agent(
        tools=[delete_file, send_email, make_payment],
        session=memory(),
        require_approval=True,  # All tools require approval
        approval_handler=websocket_approval_handler,
    )

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "user_message":
                # Handle user message
                content = data["content"]

                # Send system message about processing
                await websocket.send_json(
                    {
                        "type": "system",
                        "content": "Processing your request...",
                    }
                )

                try:
                    # Run the agent with the user message
                    # Pass session_id in metadata for the approval handler
                    result = await agent.arun(
                        content,
                        session_id=session_id,
                    )

                    # Extract response text
                    if hasattr(result, "content"):
                        response_text = result.content
                    elif hasattr(result, "paused") and result.paused:
                        response_text = "Waiting for approval..."
                    else:
                        response_text = str(result)

                    # Send response
                    await websocket.send_json(
                        {
                            "type": "assistant",
                            "content": response_text,
                        }
                    )

                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    await websocket.send_json(
                        {
                            "type": "system",
                            "content": f"Error: {e!s}",
                        }
                    )

            elif data["type"] == "approval_response":
                # Handle approval response from user
                approval_id = data["approval_id"]
                approved = data.get("approved", False)
                reason = data.get("reason", "")
                modified_args = data.get("modified_args")

                # Resolve the pending future
                future = pending_approvals.get(approval_id)
                if future and not future.done():
                    response = ApprovalResponse(
                        approved=approved,
                        reason=reason,
                        modified_args=modified_args,
                    )
                    future.set_result(response)

                    # Send confirmation to client
                    await websocket.send_json(
                        {
                            "type": "approval_result",
                            "approval_id": approval_id,
                            "approved": approved,
                            "tool_name": data.get("tool_name", "unknown"),
                        }
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        # Clean up
        active_connections.pop(session_id, None)
        # Cancel any pending approvals
        for approval_id, future in list(pending_approvals.items()):
            if not future.done():
                future.set_result(
                    ApprovalResponse(
                        approved=False,
                        reason="WebSocket disconnected",
                    )
                )


# ============================================================================
# Run the server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting HITL WebSocket Demo at http://localhost:8000")
    print("Open in your browser and try asking the agent to perform sensitive operations.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
