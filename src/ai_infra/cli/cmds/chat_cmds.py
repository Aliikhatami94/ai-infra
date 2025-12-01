"""
CLI commands for interactive chat with LLMs.

Usage:
    ai-infra chat                    # Interactive REPL (auto-detects provider)
    ai-infra chat --provider openai  # Use specific provider
    ai-infra chat --model gpt-4o     # Use specific model
    ai-infra chat -m "Hello"         # One-shot message
"""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(help="Interactive chat with LLMs")


def _get_llm(provider: Optional[str], model: Optional[str]):
    """Get LLM instance with specified or auto-detected provider."""
    from ai_infra.llm import LLM

    kwargs = {}
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model_name"] = model

    return LLM(**kwargs)


def _get_default_provider() -> str:
    """Get the default provider that would be auto-selected."""
    from ai_infra.llm.providers.discovery import get_default_provider

    return get_default_provider() or "none"


def _extract_content(response) -> str:
    """Extract text content from LLM response (AIMessage, dict, or string)."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, dict):
        return response.get("content", str(response))
    return str(response)


def _print_welcome(provider: str, model: str):
    """Print welcome message."""
    typer.echo()
    typer.secho("╭─────────────────────────────────────────╮", fg=typer.colors.CYAN)
    typer.secho("│         ai-infra Interactive Chat       │", fg=typer.colors.CYAN)
    typer.secho("╰─────────────────────────────────────────╯", fg=typer.colors.CYAN)
    typer.echo()
    typer.echo(f"  Provider: {provider}")
    typer.echo(f"  Model:    {model}")
    typer.echo()
    typer.secho("  Commands:", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /help     Show commands", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /clear    Clear conversation", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /system   Set system prompt", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("    /quit     Exit chat", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


def _print_help():
    """Print help message."""
    typer.echo()
    typer.secho("Available Commands:", bold=True)
    typer.echo("  /help              Show this help message")
    typer.echo("  /clear             Clear conversation history")
    typer.echo("  /system <prompt>   Set or update system prompt")
    typer.echo("  /history           Show conversation history")
    typer.echo("  /model <name>      Change model")
    typer.echo("  /provider <name>   Change provider")
    typer.echo("  /temp <value>      Set temperature (0.0-2.0)")
    typer.echo("  /quit, /exit       Exit the chat")
    typer.echo()
    typer.secho("Tips:", bold=True)
    typer.echo("  • Multi-line input: end line with \\ to continue")
    typer.echo("  • Ctrl+C to cancel current generation")
    typer.echo("  • Ctrl+D to exit")
    typer.echo()


def _run_repl(
    llm,
    provider: Optional[str],
    model: Optional[str],
    system: Optional[str] = None,
    temperature: float = 0.7,
    stream: bool = True,
):
    """Run interactive REPL."""
    import asyncio

    conversation = []
    current_system = system
    current_temp = temperature
    # Store actual values (None means auto-detect)
    current_provider = provider
    current_model = model

    # Display provider/model for welcome (resolve auto to actual)
    display_provider = provider or _get_default_provider()
    display_model = model or "default"

    _print_welcome(display_provider, display_model)

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
                        typer.secho(f"✓ System prompt set: {arg[:50]}...", fg=typer.colors.YELLOW)
                    else:
                        if current_system:
                            typer.echo(f"Current system prompt: {current_system}")
                        else:
                            typer.echo("No system prompt set. Use: /system <prompt>")
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
                                f"✓ Temperature set to: {current_temp}", fg=typer.colors.YELLOW
                            )
                        except ValueError:
                            typer.secho(
                                "✗ Invalid temperature. Use a number 0.0-2.0", fg=typer.colors.RED
                            )
                    else:
                        typer.echo(f"Current temperature: {current_temp}")
                    continue

                else:
                    typer.secho(
                        f"Unknown command: /{cmd}. Type /help for commands.", fg=typer.colors.RED
                    )
                    continue

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Generate response
            typer.secho("AI: ", fg=typer.colors.BLUE, nl=False)

            try:
                if stream:
                    # Streaming response
                    response_text = ""

                    async def stream_response():
                        nonlocal response_text
                        async for token, _ in llm.stream_tokens(
                            user_input,
                            system=current_system,
                            provider=current_provider,
                            model_name=current_model,
                            model_kwargs={"temperature": current_temp},
                            messages=conversation[:-1],  # Exclude current message
                        ):
                            print(token, end="", flush=True)
                            response_text += token

                    asyncio.run(stream_response())
                    typer.echo()  # Newline after streaming
                else:
                    # Non-streaming response
                    response = llm.chat(
                        user_msg=user_input,
                        system=current_system,
                        provider=current_provider,
                        model_name=current_model,
                        model_kwargs={"temperature": current_temp},
                        messages=conversation[:-1],
                    )
                    response_text = _extract_content(response)
                    typer.echo(response_text)

                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response_text})

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
            typer.echo("\nGoodbye! 👋")
            break

        except KeyboardInterrupt:
            typer.echo("\nGoodbye! 👋")
            break


@app.command("chat")
def chat_cmd(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="One-shot message (non-interactive)",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (default: auto-detect)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name",
    ),
    system: Optional[str] = typer.Option(
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
):
    """
    Interactive chat with LLMs.

    Start an interactive REPL:

        ai-infra chat

    Or send a one-shot message:

        ai-infra chat -m "What is the capital of France?"

    Examples:

        # Interactive with specific provider
        ai-infra chat --provider openai --model gpt-4o

        # One-shot with system prompt
        ai-infra chat -m "Explain Python" -s "You are a teacher"

        # JSON output for scripting
        ai-infra chat -m "Hello" --json
    """
    import json

    # Get LLM
    try:
        llm = _get_llm(provider, model)
    except Exception as e:
        typer.secho(f"Error initializing LLM: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # For display purposes only
    display_provider = provider or _get_default_provider()
    display_model = model or "default"

    # One-shot mode
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

    # Interactive mode - pass actual values (None for auto)
    _run_repl(
        llm=llm,
        provider=provider,  # None means auto-detect
        model=model,  # None means use default
        system=system,
        temperature=temperature,
        stream=not no_stream,
    )


def register(app: typer.Typer):
    """Register chat commands to main app."""
    app.command("chat")(chat_cmd)
