"""
CLI commands for interactive chat with LLMs.

Usage:
    ai-infra chat                    # Interactive REPL (auto-detects provider)
    ai-infra chat --provider openai  # Use specific provider
    ai-infra chat --model gpt-4o     # Use specific model
    ai-infra chat -m "Hello"         # One-shot message

Sessions & Persistence:
    ai-infra chat --session my-chat  # Resume or create named session
    ai-infra chat --new              # Start fresh (don't resume last session)
    ai-infra chat sessions           # List saved sessions
    ai-infra chat session-delete <n> # Delete a session

    Within chat REPL:
    /sessions                        # List saved sessions
    /save [name]                     # Save current session
    /load <name>                     # Load a saved session
    /new                             # Start new session (clears memory)
    /delete <name>                   # Delete a saved session
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

# Use Typer group for subcommands: ai-infra chat, ai-infra chat sessions, etc.
app = typer.Typer(
    help="Interactive chat with LLMs",
    invoke_without_command=True,  # Allow `ai-infra chat` to run the default command
)


# =============================================================================
# Chat Session Storage
# =============================================================================


class ChatStorage:
    """Local file-based storage for chat sessions.

    Stores chat sessions as JSON files in ~/.ai-infra/chat_sessions/
    Each session contains:
    - messages: The conversation history
    - metadata: Provider, model, system prompt, etc.
    - timestamps: Created and updated times
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize chat storage.

        Args:
            base_dir: Base directory for storage. Defaults to ~/.ai-infra/chat_sessions/
        """
        if base_dir is None:
            base_dir = Path.home() / ".ai-infra" / "chat_sessions"
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        # Sanitize session_id to be filename-safe
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._base_dir / f"{safe_id}.json"

    def save(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a chat session.

        Args:
            session_id: Unique identifier for the session
            messages: List of message dicts with 'role' and 'content'
            metadata: Optional metadata (provider, model, system, etc.)
        """
        path = self._session_path(session_id)
        existing = self._load_raw(session_id)

        data = {
            "session_id": session_id,
            "messages": messages,
            "metadata": metadata or {},
            "created_at": existing.get("created_at") if existing else datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_raw(self, session_id: str) -> dict[str, Any] | None:
        """Load raw session data."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            result = json.load(f)
            return dict(result) if isinstance(result, dict) else None

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a chat session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Session data dict or None if not found
        """
        return self._load_raw(session_id)

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._session_path(session_id).exists()

    def delete(self, session_id: str) -> bool:
        """Delete a chat session.

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with metadata.

        Returns:
            List of session info dicts sorted by updated_at (newest first)
        """
        sessions = []
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    sessions.append(
                        {
                            "session_id": data.get("session_id", path.stem),
                            "message_count": len(data.get("messages", [])),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "provider": data.get("metadata", {}).get("provider"),
                            "model": data.get("metadata", {}).get("model"),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_last_session_id(self) -> str | None:
        """Get the ID of the most recently updated session."""
        sessions = self.list_sessions()
        return sessions[0]["session_id"] if sessions else None

    def save_last_session_id(self, session_id: str) -> None:
        """Save the last used session ID for auto-resume."""
        last_file = self._base_dir / ".last_session"
        with open(last_file, "w") as f:
            f.write(session_id)

    def get_auto_resume_session_id(self) -> str | None:
        """Get the session ID to auto-resume."""
        last_file = self._base_dir / ".last_session"
        if last_file.exists():
            session_id = last_file.read_text().strip()
            if self.exists(session_id):
                return session_id
        return None


# Global storage instance
_storage: ChatStorage | None = None


def get_storage() -> ChatStorage:
    """Get or create the chat storage instance."""
    global _storage
    if _storage is None:
        _storage = ChatStorage()
    return _storage


def _get_llm(provider: str | None, model: str | None):
    """Get LLM instance.

    Note: provider and model are not used here - they're passed to individual
    calls like .chat() or .set_model(). This factory just creates the LLM instance.
    """
    from ai_infra.llm import LLM

    return LLM()


def _get_default_provider() -> str:
    """Get the default provider that would be auto-selected."""
    from ai_infra.llm.providers.discovery import get_default_provider

    return get_default_provider() or "none"


def _extract_content(response) -> str:
    """Extract text content from LLM response (AIMessage, dict, or string)."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, dict):
        return str(response.get("content", str(response)))
    return str(response)


def _build_messages_with_history(
    user_input: str,
    conversation: list[dict[str, str]],
    system: str | None = None,
) -> list[Any]:
    """Build LangChain message list with conversation history.

    Args:
        user_input: Current user message
        conversation: Previous conversation history (list of {role, content} dicts)
        system: Optional system prompt

    Returns:
        List of LangChain message objects
    """
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

    messages: list[BaseMessage] = []

    # Add system message if provided
    if system:
        messages.append(SystemMessage(content=system))

    # Add conversation history (excluding current message if already added)
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            # Only add if not already added
            if not system:
                messages.append(SystemMessage(content=content))

    # Add current user message
    messages.append(HumanMessage(content=user_input))

    return messages


def _generate_session_id() -> str:
    """Generate a unique session ID."""
    from datetime import datetime

    return f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _format_time_ago(iso_str: str | None) -> str:
    """Format ISO timestamp as human-readable time ago."""
    if not iso_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    except (ValueError, TypeError):
        return "unknown"


def _print_welcome(provider: str, model: str, session_id: str, message_count: int = 0):
    """Print welcome message."""
    typer.echo()
    typer.secho("╭─────────────────────────────────────────╮", fg=typer.colors.CYAN)
    typer.secho("│         ai-infra Interactive Chat       │", fg=typer.colors.CYAN)
    typer.secho("╰─────────────────────────────────────────╯", fg=typer.colors.CYAN)
    typer.echo()
    typer.echo(f"  Provider: {provider}")
    typer.echo(f"  Model:    {model}")
    typer.echo(f"  Session:  {session_id}")
    if message_count > 0:
        typer.secho(f"  Memory:   {message_count} messages restored", fg=typer.colors.GREEN)
    typer.echo()
    typer.secho("  Commands:", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /help     Show all commands", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /sessions List saved sessions", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /clear    Clear conversation", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /quit     Save and exit", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


def _print_help():
    """Print help message."""
    typer.echo()
    typer.secho("Conversation Commands:", bold=True)
    typer.echo("  /help              Show this help message")
    typer.echo("  /clear             Clear conversation history")
    typer.echo("  /system <prompt>   Set or update system prompt")
    typer.echo("  /history           Show conversation history")
    typer.echo()
    typer.secho("Session Commands:", bold=True)
    typer.echo("  /sessions          List all saved sessions")
    typer.echo("  /save [name]       Save current session (auto-generates name if omitted)")
    typer.echo("  /load <name>       Load a saved session")
    typer.echo("  /new               Start a new session (current is auto-saved)")
    typer.echo("  /delete <name>     Delete a saved session")
    typer.echo("  /rename <name>     Rename current session")
    typer.echo()
    typer.secho("Model Commands:", bold=True)
    typer.echo("  /model <name>      Change model")
    typer.echo("  /provider <name>   Change provider")
    typer.echo("  /temp <value>      Set temperature (0.0-2.0)")
    typer.echo()
    typer.secho("Exit Commands:", bold=True)
    typer.echo("  /quit, /exit       Save session and exit")
    typer.echo()
    typer.secho("Tips:", bold=True)
    typer.echo("  • Sessions auto-save on exit and auto-resume on start")
    typer.echo("  • Multi-line input: end line with \\ to continue")
    typer.echo("  • Ctrl+C to cancel current generation")
    typer.echo("  • Ctrl+D to exit")
    typer.echo()


def _run_repl(
    llm,
    provider: str | None,
    model: str | None,
    system: str | None = None,
    temperature: float = 0.7,
    stream: bool = True,
    session_id: str | None = None,
    no_persist: bool = False,
):
    """Run interactive REPL with session persistence."""
    import asyncio

    storage = get_storage()

    # Session management
    current_session_id = session_id or _generate_session_id()
    conversation: list[dict[str, str]] = []
    current_system = system
    current_temp = temperature
    current_provider = provider
    current_model = model

    # Try to load existing session
    restored_count = 0
    if session_id and storage.exists(session_id):
        session_data = storage.load(session_id)
        if session_data:
            conversation = session_data.get("messages", [])
            restored_count = len(conversation)
            # Restore metadata if not overridden
            metadata = session_data.get("metadata", {})
            if not system and metadata.get("system"):
                current_system = metadata["system"]
            if not provider and metadata.get("provider"):
                current_provider = metadata["provider"]
            if not model and metadata.get("model"):
                current_model = metadata["model"]

    # Display provider/model for welcome (resolve auto to actual)
    display_provider = current_provider or _get_default_provider()
    display_model = current_model or "default"

    _print_welcome(display_provider, display_model, current_session_id, restored_count)

    def _save_session():
        """Save current session to storage."""
        if no_persist:
            return
        metadata = {
            "provider": current_provider,
            "model": current_model,
            "system": current_system,
            "temperature": current_temp,
        }
        storage.save(current_session_id, conversation, metadata)
        storage.save_last_session_id(current_session_id)

    while True:
        try:
            # Prompt
            typer.secho("You: ", fg=typer.colors.GREEN, nl=False)
            user_input = input()

            # Handle empty input
            if not user_input.strip():
                continue

            # Handle multi-line input
            while user_input.endswith("\\"):
                user_input = user_input[:-1] + "\n"
                continuation = input("... ")
                user_input += continuation

            user_input = user_input.strip()

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd in ("quit", "exit", "q"):
                    _save_session()
                    typer.secho("✓ Session saved", fg=typer.colors.GREEN)
                    typer.echo("\nGoodbye! 👋")
                    break

                elif cmd == "help":
                    _print_help()
                    continue

                elif cmd == "clear":
                    conversation = []
                    typer.secho("✓ Conversation cleared", fg=typer.colors.YELLOW)
                    continue

                elif cmd == "history":
                    if not conversation:
                        typer.echo("No conversation history yet.")
                    else:
                        typer.echo()
                        for msg in conversation:
                            role = msg["role"].capitalize()
                            content = (
                                msg["content"][:100] + "..."
                                if len(msg["content"]) > 100
                                else msg["content"]
                            )
                            typer.echo(f"  [{role}] {content}")
                        typer.echo()
                    continue

                elif cmd == "system":
                    if arg:
                        current_system = arg
                        typer.secho(
                            f"✓ System prompt set: {arg[:50]}...",
                            fg=typer.colors.YELLOW,
                        )
                    else:
                        if current_system:
                            typer.echo(f"Current system prompt: {current_system}")
                        else:
                            typer.echo("No system prompt set. Use: /system <prompt>")
                    continue

                # Session commands
                elif cmd == "sessions":
                    sessions = storage.list_sessions()
                    if not sessions:
                        typer.echo("No saved sessions.")
                    else:
                        typer.echo()
                        typer.secho("Saved Sessions:", bold=True)
                        for s in sessions:
                            active = " (current)" if s["session_id"] == current_session_id else ""
                            provider_info = s.get("provider") or "auto"
                            time_ago = _format_time_ago(s.get("updated_at"))
                            typer.echo(
                                f"  • {s['session_id']}{active} "
                                f"- {s['message_count']} msgs, {provider_info}, {time_ago}"
                            )
                        typer.echo()
                    continue

                elif cmd == "save":
                    new_id = arg.strip() if arg else current_session_id
                    current_session_id = new_id
                    _save_session()
                    typer.secho(
                        f"✓ Session saved as: {current_session_id}",
                        fg=typer.colors.GREEN,
                    )
                    continue

                elif cmd == "load":
                    if not arg:
                        typer.secho("Usage: /load <session_name>", fg=typer.colors.RED)
                        continue
                    load_id = arg.strip()
                    if not storage.exists(load_id):
                        typer.secho(f"✗ Session not found: {load_id}", fg=typer.colors.RED)
                        typer.echo("Use /sessions to list available sessions")
                        continue
                    # Save current session before switching
                    _save_session()
                    # Load new session
                    session_data = storage.load(load_id)
                    if session_data:
                        conversation = session_data.get("messages", [])
                        current_session_id = load_id
                        metadata = session_data.get("metadata", {})
                        if metadata.get("system"):
                            current_system = metadata["system"]
                        if metadata.get("provider"):
                            current_provider = metadata["provider"]
                            llm = _get_llm(current_provider, current_model)
                        if metadata.get("model"):
                            current_model = metadata["model"]
                        if metadata.get("temperature"):
                            current_temp = metadata["temperature"]
                        typer.secho(
                            f"✓ Loaded session: {load_id} ({len(conversation)} messages)",
                            fg=typer.colors.GREEN,
                        )
                    continue

                elif cmd == "new":
                    # Save current session
                    _save_session()
                    # Start fresh
                    current_session_id = arg.strip() if arg else _generate_session_id()
                    conversation = []
                    current_system = system  # Reset to original system prompt
                    typer.secho(f"✓ New session: {current_session_id}", fg=typer.colors.GREEN)
                    continue

                elif cmd == "delete":
                    if not arg:
                        typer.secho("Usage: /delete <session_name>", fg=typer.colors.RED)
                        continue
                    delete_id = arg.strip()
                    if delete_id == current_session_id:
                        typer.secho(
                            "✗ Cannot delete current session. Use /new first.",
                            fg=typer.colors.RED,
                        )
                        continue
                    if storage.delete(delete_id):
                        typer.secho(f"✓ Deleted session: {delete_id}", fg=typer.colors.GREEN)
                    else:
                        typer.secho(f"✗ Session not found: {delete_id}", fg=typer.colors.RED)
                    continue

                elif cmd == "rename":
                    if not arg:
                        typer.secho("Usage: /rename <new_name>", fg=typer.colors.RED)
                        continue
                    new_name = arg.strip()
                    if storage.exists(new_name):
                        typer.secho(
                            f"✗ Session already exists: {new_name}",
                            fg=typer.colors.RED,
                        )
                        continue
                    old_id = current_session_id
                    current_session_id = new_name
                    _save_session()
                    storage.delete(old_id)
                    typer.secho(
                        f"✓ Renamed: {old_id} → {new_name}",
                        fg=typer.colors.GREEN,
                    )
                    continue

                elif cmd == "model":
                    if arg:
                        current_model = arg
                        typer.secho(f"✓ Model changed to: {arg}", fg=typer.colors.YELLOW)
                    else:
                        display = current_model or "default (auto)"
                        typer.echo(f"Current model: {display}")
                    continue

                elif cmd == "provider":
                    if arg:
                        current_provider = arg
                        try:
                            llm = _get_llm(current_provider, current_model)
                            typer.secho(f"✓ Provider changed to: {arg}", fg=typer.colors.YELLOW)
                        except Exception as e:
                            typer.secho(f"✗ Failed to change provider: {e}", fg=typer.colors.RED)
                    else:
                        display = current_provider or _get_default_provider() + " (auto)"
                        typer.echo(f"Current provider: {display}")
                    continue

                elif cmd == "temp":
                    if arg:
                        try:
                            current_temp = float(arg)
                            typer.secho(
                                f"✓ Temperature set to: {current_temp}",
                                fg=typer.colors.YELLOW,
                            )
                        except ValueError:
                            typer.secho(
                                "✗ Invalid temperature. Use a number 0.0-2.0",
                                fg=typer.colors.RED,
                            )
                    else:
                        typer.echo(f"Current temperature: {current_temp}")
                    continue

                else:
                    typer.secho(
                        f"Unknown command: /{cmd}. Type /help for commands.",
                        fg=typer.colors.RED,
                    )
                    continue

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Generate response
            typer.secho("AI: ", fg=typer.colors.BLUE, nl=False)

            try:
                # Build messages with full conversation history
                messages = _build_messages_with_history(
                    user_input,
                    conversation[:-1],  # Exclude current message (already in messages)
                    system=current_system,
                )

                # Resolve provider/model (handles None -> auto-detect)
                resolved_provider, resolved_model = llm._resolve_provider_and_model(
                    current_provider, current_model
                )

                # Get model with temperature
                chat_model: Any = llm.set_model(
                    resolved_provider,
                    resolved_model,
                    temperature=current_temp,
                )

                if stream:
                    # Streaming response
                    response_text = ""

                    async def stream_response():
                        nonlocal response_text
                        async for event in chat_model.astream(messages):
                            text = getattr(event, "content", None)
                            if text:
                                print(text, end="", flush=True)
                                response_text += text

                    asyncio.run(stream_response())
                    typer.echo()  # Newline after streaming
                else:
                    # Non-streaming response
                    response = chat_model.invoke(messages)
                    response_text = _extract_content(response)
                    typer.echo(response_text)

                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response_text})

                # Auto-save after each exchange
                _save_session()

            except KeyboardInterrupt:
                typer.echo("\n[Interrupted]")
                # Remove the user message since we didn't get a response
                conversation.pop()
                continue

            except Exception as e:
                typer.secho(f"\n✗ Error: {e}", fg=typer.colors.RED)
                conversation.pop()
                continue

            typer.echo()  # Extra newline for readability

        except EOFError:
            _save_session()
            typer.echo("\nGoodbye! 👋")
            break

        except KeyboardInterrupt:
            _save_session()
            typer.echo("\nGoodbye! 👋")
            break


@app.callback(invoke_without_command=True)
def chat_cmd(
    ctx: typer.Context,
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="One-shot message (non-interactive)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (default: auto-detect)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name",
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Temperature (0.0-2.0)",
    ),
    no_stream: bool = typer.Option(
        False,
        "--no-stream",
        help="Disable streaming output",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON (one-shot mode only)",
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        help="Session name to resume or create",
    ),
    new_session: bool = typer.Option(
        False,
        "--new",
        "-n",
        help="Start a new session (don't auto-resume)",
    ),
    no_persist: bool = typer.Option(
        False,
        "--no-persist",
        help="Disable session persistence",
    ),
):
    """
    Interactive chat with LLMs with session persistence.

    Start an interactive REPL:

        ai-infra chat

    Resume last session or create new one:

        ai-infra chat                   # Auto-resumes last session
        ai-infra chat --new             # Start fresh
        ai-infra chat --session mywork  # Resume/create named session

    Or send a one-shot message:

        ai-infra chat -m "What is the capital of France?"

    Examples:

        # Interactive with specific provider
        ai-infra chat --provider openai --model gpt-4o

        # Resume a named session
        ai-infra chat --session project-discussion

        # One-shot with system prompt
        ai-infra chat -m "Explain Python" -s "You are a teacher"

        # JSON output for scripting
        ai-infra chat -m "Hello" --json

    Session management:
        ai-infra chat sessions              # List saved sessions
        ai-infra chat session-delete <name> # Delete a session

    Within the REPL, use /help to see all commands including:
        /sessions   - List saved sessions
        /save       - Save current session
        /load       - Load a saved session
        /new        - Start a new session
    """
    # If a subcommand is being invoked, don't run the default chat
    if ctx.invoked_subcommand is not None:
        return

    # Get LLM
    try:
        llm = _get_llm(provider, model)
    except Exception as e:
        typer.secho(f"Error initializing LLM: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # For display purposes only
    display_provider = provider or _get_default_provider()
    display_model = model or "default"

    # One-shot mode (no persistence)
    if message:
        try:
            response = llm.chat(
                user_msg=message,
                system=system,
                provider=provider,  # Pass actual value (None for auto)
                model_name=model,  # Pass actual value (None for auto)
                model_kwargs={"temperature": temperature},
            )

            # Extract content from response (handles AIMessage, dict, or string)
            response_text = _extract_content(response)

            if output_json:
                result = {
                    "provider": display_provider,
                    "model": display_model,
                    "message": message,
                    "response": response_text,
                }
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(response_text)

        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        return

    # Interactive mode - determine session ID
    session_id = session  # Explicit session name if provided

    if not session_id and not new_session and not no_persist:
        # Auto-resume: try to get last session
        storage = get_storage()
        session_id = storage.get_auto_resume_session_id()

    _run_repl(
        llm=llm,
        provider=provider,  # None means auto-detect
        model=model,  # None means use default
        system=system,
        temperature=temperature,
        stream=not no_stream,
        session_id=session_id,
        no_persist=no_persist,
    )


@app.command("sessions")
def sessions_cmd(
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """
    List all saved chat sessions.

    Example:
        ai-infra chat sessions
        ai-infra chat sessions --json
    """
    storage = get_storage()
    sessions = storage.list_sessions()

    if not sessions:
        typer.echo("No saved sessions.")
        return

    if output_json:
        typer.echo(json.dumps(sessions, indent=2))
    else:
        typer.echo()
        typer.secho("Saved Chat Sessions:", bold=True)
        typer.echo()
        for s in sessions:
            provider_info = s.get("provider") or "auto"
            time_ago = _format_time_ago(s.get("updated_at"))
            typer.echo(
                f"  • {s['session_id']} - {s['message_count']} msgs, {provider_info}, {time_ago}"
            )
        typer.echo()
        typer.echo(f"Storage: {storage._base_dir}")
        typer.echo()


@app.command("session-delete")
def session_delete_cmd(
    session_name: str = typer.Argument(
        ...,
        help="Session name to delete (or 'all' to delete all sessions)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """
    Delete a chat session or all sessions.

    Examples:
        ai-infra chat session-delete chat-20251201-182558
        ai-infra chat session-delete all
        ai-infra chat session-delete all --force
    """
    storage = get_storage()

    if session_name.lower() == "all":
        sessions = storage.list_sessions()
        if not sessions:
            typer.echo("No sessions to delete.")
            return

        if not force:
            typer.echo(f"This will delete {len(sessions)} session(s):")
            for s in sessions:
                typer.echo(f"  • {s['session_id']}")
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                typer.echo("Cancelled.")
                return

        deleted = 0
        for s in sessions:
            if storage.delete(s["session_id"]):
                deleted += 1
        typer.secho(f"✓ Deleted {deleted} session(s)", fg=typer.colors.GREEN)

    else:
        if not storage.exists(session_name):
            typer.secho(f"✗ Session not found: {session_name}", fg=typer.colors.RED)
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Delete session '{session_name}'?")
            if not confirm:
                typer.echo("Cancelled.")
                return

        if storage.delete(session_name):
            typer.secho(f"✓ Deleted: {session_name}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"✗ Failed to delete: {session_name}", fg=typer.colors.RED)
            raise typer.Exit(1)


def register(main_app: typer.Typer):
    """Register chat command group to main app."""
    # Add the chat app as a subcommand group
    # This enables: ai-infra chat, ai-infra chat sessions, ai-infra chat session-delete
    main_app.add_typer(app, name="chat")
