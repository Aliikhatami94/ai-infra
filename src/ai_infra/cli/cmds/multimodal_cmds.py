"""
CLI commands for multimodal discovery (TTS, STT, Audio).

Usage:
    ai-infra tts-providers          # List TTS providers
    ai-infra tts-voices --provider openai  # List voices
    ai-infra stt-providers          # List STT providers
    ai-infra stt-models --provider openai  # List STT models
    ai-infra audio-models --provider openai  # List audio LLM models
"""

from __future__ import annotations

import json

import typer

from ai_infra.llm.multimodal.discovery import (
    get_default_stt_provider,
    get_default_tts_provider,
    is_stt_configured,
    is_tts_configured,
    list_audio_input_models,
    list_audio_output_models,
    list_stt_models,
    list_stt_providers,
    list_tts_models,
    list_tts_providers,
    list_tts_voices,
)

app = typer.Typer(help="Multimodal discovery commands (TTS, STT, Audio)")


# =============================================================================
# TTS Commands
# =============================================================================


@app.command("tts-providers")
def tts_providers_cmd(
    configured: bool = typer.Option(
        False,
        "--configured",
        "-c",
        help="Only show providers with API keys configured",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List all supported TTS providers."""
    providers = list_tts_providers()

    if configured:
        providers = [p for p in providers if is_tts_configured(p)]

    if output_json:
        typer.echo(json.dumps(providers, indent=2))
    else:
        if not providers:
            typer.echo("No TTS providers configured. Set API key environment variables.")
            raise typer.Exit(1)

        typer.echo("\nTTS Providers:")
        for provider in providers:
            status = "✓" if is_tts_configured(provider) else "✗"
            typer.echo(f"  {status} {provider}")


@app.command("tts-voices")
def tts_voices_cmd(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to list voices for (default: first configured)",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List available TTS voices for a provider."""
    if not provider:
        provider = get_default_tts_provider()
        if not provider:
            typer.echo("Error: No TTS provider configured. Set an API key.")
            raise typer.Exit(1)

    try:
        voices = list_tts_voices(provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if output_json:
        typer.echo(json.dumps(voices, indent=2))
    else:
        typer.echo(f"\n{provider} voices ({len(voices)}):")
        for voice in voices:
            typer.echo(f"  • {voice}")


@app.command("tts-models")
def tts_models_cmd(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to list models for",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List available TTS models for a provider."""
    if not provider:
        provider = get_default_tts_provider()
        if not provider:
            typer.echo("Error: No TTS provider configured. Set an API key.")
            raise typer.Exit(1)

    try:
        models = list_tts_models(provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if output_json:
        typer.echo(json.dumps(models, indent=2))
    else:
        typer.echo(f"\n{provider} TTS models ({len(models)}):")
        for model in models:
            typer.echo(f"  • {model}")


# =============================================================================
# STT Commands
# =============================================================================


@app.command("stt-providers")
def stt_providers_cmd(
    configured: bool = typer.Option(
        False,
        "--configured",
        "-c",
        help="Only show providers with API keys configured",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List all supported STT providers."""
    providers = list_stt_providers()

    if configured:
        providers = [p for p in providers if is_stt_configured(p)]

    if output_json:
        typer.echo(json.dumps(providers, indent=2))
    else:
        if not providers:
            typer.echo("No STT providers configured. Set API key environment variables.")
            raise typer.Exit(1)

        typer.echo("\nSTT Providers:")
        for provider in providers:
            status = "✓" if is_stt_configured(provider) else "✗"
            typer.echo(f"  {status} {provider}")


@app.command("stt-models")
def stt_models_cmd(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to list models for",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List available STT models for a provider."""
    if not provider:
        provider = get_default_stt_provider()
        if not provider:
            typer.echo("Error: No STT provider configured. Set an API key.")
            raise typer.Exit(1)

    try:
        models = list_stt_models(provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if output_json:
        typer.echo(json.dumps(models, indent=2))
    else:
        typer.echo(f"\n{provider} STT models ({len(models)}):")
        for model in models:
            typer.echo(f"  • {model}")


# =============================================================================
# Audio LLM Commands
# =============================================================================


@app.command("audio-models")
def audio_models_cmd(
    provider: str = typer.Option(
        "openai",
        "--provider",
        "-p",
        help="Provider to list audio models for",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List audio-capable LLM models for a provider."""
    try:
        input_models = list_audio_input_models(provider)
        output_models = list_audio_output_models(provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if output_json:
        typer.echo(
            json.dumps(
                {
                    "provider": provider,
                    "audio_input": input_models,
                    "audio_output": output_models,
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"\n{provider} Audio Models:")
        typer.echo("\n  Audio Input (can understand audio):")
        for model in input_models:
            typer.echo(f"    • {model}")
        typer.echo("\n  Audio Output (can generate audio):")
        for model in output_models:
            typer.echo(f"    • {model}")


def register(parent: typer.Typer):
    """Register multimodal commands with the parent CLI app."""
    parent.add_typer(app, name="multimodal", help="Multimodal discovery (TTS, STT, Audio)")
    # Also add as top-level commands for convenience
    parent.command("tts-providers")(tts_providers_cmd)
    parent.command("tts-voices")(tts_voices_cmd)
    parent.command("tts-models")(tts_models_cmd)
    parent.command("stt-providers")(stt_providers_cmd)
    parent.command("stt-models")(stt_models_cmd)
    parent.command("audio-models")(audio_models_cmd)
