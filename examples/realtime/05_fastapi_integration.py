#!/usr/bin/env python
"""FastAPI WebSocket Integration for Realtime Voice Example.

This example demonstrates:
- WebSocket endpoint for voice streaming
- Bidirectional audio communication
- FastAPI integration patterns
- Browser-to-server audio streaming
- Session management in web apps

Build real-time voice applications with FastAPI.
"""

import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ai_infra.llm.realtime import (
    RealtimeConfig,
    RealtimeVoice,
    ToolCallRequest,
    ToolDefinition,
)

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Realtime Voice API",
    description="WebSocket API for real-time voice conversations",
)


# =============================================================================
# Example 1: Basic WebSocket Endpoint
# =============================================================================


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Basic WebSocket endpoint for voice streaming."""
    await websocket.accept()

    # Check if any provider is configured
    if not RealtimeVoice.configured_providers():
        await websocket.send_json(
            {
                "type": "error",
                "message": "No voice provider configured",
            }
        )
        await websocket.close()
        return

    # Create voice instance
    voice = RealtimeVoice()

    # Handle transcripts
    @voice.on_transcript
    async def on_transcript(text: str, is_final: bool):
        await websocket.send_json(
            {
                "type": "transcript",
                "text": text,
                "is_final": is_final,
            }
        )

    # Handle audio output
    @voice.on_audio
    async def on_audio(audio: bytes):
        # Send as binary WebSocket message
        await websocket.send_bytes(audio)

    # Handle errors
    @voice.on_error
    async def on_error(error):
        await websocket.send_json(
            {
                "type": "error",
                "message": str(error),
            }
        )

    try:
        async with voice.connect() as session:
            # Notify client that connection is ready
            await websocket.send_json(
                {
                    "type": "connected",
                    "provider": voice.provider_name,
                }
            )

            # Process incoming messages
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # Handle binary audio data
                if "bytes" in message:
                    await session.send_audio(message["bytes"])

                # Handle JSON commands
                elif "text" in message:
                    data = json.loads(message["text"])

                    if data.get("type") == "text":
                        # Send text (TTS)
                        await session.send_text(data["text"])

                    elif data.get("type") == "interrupt":
                        # Interrupt AI response
                        await session.interrupt()

                    elif data.get("type") == "commit":
                        # Commit audio (manual VAD)
                        await session.commit_audio()

    except WebSocketDisconnect:
        pass  # Client disconnected
    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "message": str(e),
            }
        )
        raise


# =============================================================================
# Example 2: WebSocket with Configuration
# =============================================================================


class VoiceSettings(BaseModel):
    """Voice session settings."""

    provider: str | None = None
    voice: str | None = None
    model: str | None = None
    system_prompt: str | None = None


@app.websocket("/ws/voice/configured")
async def configured_voice_websocket(
    websocket: WebSocket,
    provider: str | None = None,
    voice: str | None = None,
    model: str | None = None,
):
    """WebSocket endpoint with query parameter configuration."""
    await websocket.accept()

    # Build config from query params
    config = RealtimeConfig(
        model=model,
        voice=voice,
    )

    # Create voice with config
    voice_client = RealtimeVoice(
        provider=provider,
        config=config,
    )

    @voice_client.on_transcript
    async def on_transcript(text: str, is_final: bool):
        await websocket.send_json(
            {
                "type": "transcript",
                "text": text,
                "is_final": is_final,
            }
        )

    @voice_client.on_audio
    async def on_audio(audio: bytes):
        await websocket.send_bytes(audio)

    try:
        async with voice_client.connect() as session:
            await websocket.send_json(
                {
                    "type": "connected",
                    "provider": voice_client.provider_name,
                    "config": {
                        "model": model,
                        "voice": voice,
                    },
                }
            )

            while True:
                message = await websocket.receive()
                if "bytes" in message:
                    await session.send_audio(message["bytes"])
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "text":
                        await session.send_text(data["text"])

    except WebSocketDisconnect:
        pass


# =============================================================================
# Example 3: WebSocket with Tools
# =============================================================================


# Tool implementations
def get_weather(location: str) -> dict:
    """Simulated weather API."""
    return {
        "location": location,
        "temperature": 22,
        "condition": "Sunny",
    }


def get_time() -> dict:
    """Get current time."""
    from datetime import datetime

    return {
        "time": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }


TOOL_HANDLERS = {
    "get_weather": get_weather,
    "get_time": get_time,
}


@app.websocket("/ws/voice/assistant")
async def voice_assistant_websocket(websocket: WebSocket):
    """Voice assistant with tool calling."""
    await websocket.accept()

    # Define tools
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="get_time",
            description="Get current time",
            parameters={"type": "object", "properties": {}},
        ),
    ]

    config = RealtimeConfig(
        tools=tools,
        system_prompt="You are a helpful voice assistant.",
    )

    voice = RealtimeVoice(config=config)

    @voice.on_transcript
    async def on_transcript(text: str, is_final: bool):
        await websocket.send_json(
            {
                "type": "transcript",
                "text": text,
                "is_final": is_final,
            }
        )

    @voice.on_audio
    async def on_audio(audio: bytes):
        await websocket.send_bytes(audio)

    @voice.on_tool_call
    async def on_tool_call(request: ToolCallRequest):
        # Notify client about tool call
        await websocket.send_json(
            {
                "type": "tool_call",
                "name": request.name,
                "arguments": request.arguments,
            }
        )

        # Execute tool
        handler = TOOL_HANDLERS.get(request.name)
        if handler:
            result = handler(**request.arguments)
            await websocket.send_json(
                {
                    "type": "tool_result",
                    "name": request.name,
                    "result": result,
                }
            )
            return result
        return {"error": f"Unknown tool: {request.name}"}

    try:
        async with voice.connect() as session:
            await websocket.send_json(
                {
                    "type": "connected",
                    "tools": [t.name for t in tools],
                }
            )

            while True:
                message = await websocket.receive()
                if "bytes" in message:
                    await session.send_audio(message["bytes"])
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "text":
                        await session.send_text(data["text"])

    except WebSocketDisconnect:
        pass


# =============================================================================
# Example 4: Session Management
# =============================================================================

# Track active sessions
active_sessions: dict[str, "VoiceSessionManager"] = {}


class VoiceSessionManager:
    """Manage voice session lifecycle."""

    def __init__(self, session_id: str, config: RealtimeConfig | None = None):
        self.session_id = session_id
        self.voice = RealtimeVoice(config=config)
        self.websocket: WebSocket | None = None
        self.session = None

    async def connect(self, websocket: WebSocket):
        """Connect WebSocket to voice session."""
        self.websocket = websocket

        @self.voice.on_transcript
        async def on_transcript(text: str, is_final: bool):
            if self.websocket:
                await self.websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "is_final": is_final,
                    }
                )

        @self.voice.on_audio
        async def on_audio(audio: bytes):
            if self.websocket:
                await self.websocket.send_bytes(audio)

        self.session = await self.voice.connect().__aenter__()

    async def send_audio(self, audio: bytes):
        """Send audio to voice session."""
        if self.session:
            await self.session.send_audio(audio)

    async def send_text(self, text: str):
        """Send text to voice session."""
        if self.session:
            await self.session.send_text(text)

    async def disconnect(self):
        """Disconnect and cleanup."""
        if self.session:
            await self.voice.connect().__aexit__(None, None, None)
        self.websocket = None
        self.session = None


@app.post("/api/voice/sessions")
async def create_session():
    """Create a new voice session."""
    import uuid

    session_id = str(uuid.uuid4())

    manager = VoiceSessionManager(session_id)
    active_sessions[session_id] = manager

    return {"session_id": session_id}


@app.delete("/api/voice/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a voice session."""
    if session_id in active_sessions:
        await active_sessions[session_id].disconnect()
        del active_sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(404, "Session not found")


@app.websocket("/ws/voice/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """Connect to an existing voice session."""
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    manager = active_sessions[session_id]

    try:
        await manager.connect(websocket)

        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
            }
        )

        while True:
            message = await websocket.receive()
            if "bytes" in message:
                await manager.send_audio(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "text":
                    await manager.send_text(data["text"])

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect()


# =============================================================================
# Example 5: REST Endpoints for Discovery
# =============================================================================


@app.get("/api/voice/providers")
async def get_providers():
    """Get available voice providers."""
    return {
        "available": RealtimeVoice.available_providers(),
        "configured": RealtimeVoice.configured_providers(),
    }


@app.get("/api/voice/models")
async def get_models(provider: str | None = None):
    """Get available models."""
    return {
        "models": RealtimeVoice.list_models(provider=provider),
    }


@app.get("/api/voice/voices")
async def get_voices(provider: str | None = None):
    """Get available voices."""
    return {
        "voices": RealtimeVoice.list_voices(provider=provider),
    }


# =============================================================================
# Example 6: HTML Test Client
# =============================================================================


@app.get("/")
async def get_test_page():
    """Serve a simple test page for the WebSocket API."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Realtime Voice Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            #status { padding: 10px; background: #f0f0f0; margin: 10px 0; }
            #transcript { padding: 10px; border: 1px solid #ccc; min-height: 100px; }
        </style>
    </head>
    <body>
        <h1>Realtime Voice Test</h1>

        <div id="status">Status: Disconnected</div>

        <div class="controls">
            <button id="connect">Connect</button>
            <button id="disconnect" disabled>Disconnect</button>
            <button id="startMic" disabled>Start Microphone</button>
            <button id="stopMic" disabled>Stop Microphone</button>
        </div>

        <h3>Transcript</h3>
        <div id="transcript"></div>

        <h3>Send Text</h3>
        <input type="text" id="textInput" placeholder="Type a message...">
        <button id="sendText" disabled>Send</button>

        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioContext = null;

            const status = document.getElementById('status');
            const transcript = document.getElementById('transcript');

            document.getElementById('connect').onclick = async () => {
                ws = new WebSocket('ws://localhost:8000/ws/voice');

                ws.onopen = () => {
                    status.textContent = 'Status: Connected';
                    document.getElementById('connect').disabled = true;
                    document.getElementById('disconnect').disabled = false;
                    document.getElementById('startMic').disabled = false;
                    document.getElementById('sendText').disabled = false;
                };

                ws.onmessage = async (event) => {
                    if (event.data instanceof Blob) {
                        // Audio data - play it
                        playAudio(await event.data.arrayBuffer());
                    } else {
                        const data = JSON.parse(event.data);
                        if (data.type === 'transcript') {
                            const prefix = data.is_final ? 'AI: ' : 'AI (partial): ';
                            transcript.innerHTML += prefix + data.text + '<br>';
                        } else if (data.type === 'connected') {
                            status.textContent = 'Status: Connected to ' + data.provider;
                        }
                    }
                };

                ws.onclose = () => {
                    status.textContent = 'Status: Disconnected';
                    document.getElementById('connect').disabled = false;
                    document.getElementById('disconnect').disabled = true;
                    document.getElementById('startMic').disabled = true;
                    document.getElementById('sendText').disabled = true;
                };
            };

            document.getElementById('disconnect').onclick = () => {
                ws?.close();
            };

            document.getElementById('startMic').onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                audioContext = new AudioContext({ sampleRate: 24000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);

                source.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = (e) => {
                    if (ws?.readyState === WebSocket.OPEN) {
                        const input = e.inputBuffer.getChannelData(0);
                        const pcm16 = new Int16Array(input.length);
                        for (let i = 0; i < input.length; i++) {
                            pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                        }
                        ws.send(pcm16.buffer);
                    }
                };

                document.getElementById('startMic').disabled = true;
                document.getElementById('stopMic').disabled = false;
            };

            document.getElementById('stopMic').onclick = () => {
                audioContext?.close();
                document.getElementById('startMic').disabled = false;
                document.getElementById('stopMic').disabled = true;
            };

            document.getElementById('sendText').onclick = () => {
                const input = document.getElementById('textInput');
                if (input.value && ws?.readyState === WebSocket.OPEN) {
                    transcript.innerHTML += 'You: ' + input.value + '<br>';
                    ws.send(JSON.stringify({ type: 'text', text: input.value }));
                    input.value = '';
                }
            };

            async function playAudio(arrayBuffer) {
                if (!audioContext) {
                    audioContext = new AudioContext({ sampleRate: 24000 });
                }
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("FastAPI Realtime Voice Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /                          - Test page")
    print("  WS   /ws/voice                  - Basic voice WebSocket")
    print("  WS   /ws/voice/configured       - Configurable WebSocket")
    print("  WS   /ws/voice/assistant        - Voice assistant with tools")
    print("  POST /api/voice/sessions        - Create session")
    print("  WS   /ws/voice/session/{id}     - Connect to session")
    print("  GET  /api/voice/providers       - List providers")
    print("  GET  /api/voice/models          - List models")
    print("  GET  /api/voice/voices          - List voices")
    print("\n" + "=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
