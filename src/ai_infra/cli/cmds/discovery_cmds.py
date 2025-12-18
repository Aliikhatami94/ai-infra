"""
CLI commands for provider and model discovery.

Usage:
    ai-infra providers           # List all supported providers
    ai-infra providers --configured  # Only configured providers
    ai-infra models --provider openai  # List models for a provider
    ai-infra models --all        # List models for all configured providers
"""

from __future__ import annotations

import json

import typer

from ai_infra.llm.providers.discovery import (
    SUPPORTED_PROVIDERS,
    is_provider_configured,
    list_all_models,
    list_models,
)

app = typer.Typer(help="Provider and model discovery commands")


@app.command("providers")
def providers_cmd(
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
    """List all supported AI providers."""
    if configured:
        result = [p for p in SUPPORTED_PROVIDERS if is_provider_configured(p)]
    else:
        result = SUPPORTED_PROVIDERS.copy()

    if output_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        if not result:
            typer.echo("No providers configured. Set API key environment variables.")
            raise typer.Exit(1)

        for provider in result:
            status = "✓" if is_provider_configured(provider) else "✗"
            typer.echo(f"  {status} {provider}")


@app.command("models")
def models_cmd(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to list models for",
    ),
    all_providers: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="List models for all configured providers",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        "-r",
        help="Force refresh from API (bypass cache)",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List available models from AI providers."""
    if not provider and not all_providers:
        typer.echo("Error: Specify --provider <name> or --all")
        raise typer.Exit(1)

    if all_providers:
        try:
            result = list_all_models(refresh=refresh)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        if output_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            if not result:
                typer.echo("No configured providers found.")
                raise typer.Exit(1)

            for prov, models in result.items():
                typer.echo(f"\n{prov} ({len(models)} models):")
                for model in models[:10]:  # Show first 10
                    typer.echo(f"  • {model}")
                if len(models) > 10:
                    typer.echo(f"  ... and {len(models) - 10} more")
    else:
        if provider not in SUPPORTED_PROVIDERS:
            typer.echo(f"Error: Unknown provider '{provider}'")
            typer.echo(f"Supported: {', '.join(SUPPORTED_PROVIDERS)}")
            raise typer.Exit(1)

        if not is_provider_configured(provider):
            typer.echo(f"Error: Provider '{provider}' is not configured (no API key)")
            raise typer.Exit(1)

        try:
            models = list_models(provider, refresh=refresh)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        if output_json:
            typer.echo(json.dumps(models, indent=2))
        else:
            typer.echo(f"\n{provider} ({len(models)} models):")
            for model in models:
                typer.echo(f"  • {model}")


def register(parent: typer.Typer):
    """Register discovery commands with the parent CLI app."""
    parent.add_typer(app, name="discover", help="Provider and model discovery")
    # Also add as top-level commands for convenience
    parent.command("providers")(providers_cmd)
    parent.command("models")(models_cmd)
