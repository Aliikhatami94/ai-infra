"""
CLI commands for image generation provider and model discovery.

Usage:
    ai-infra image-providers           # List all supported providers
    ai-infra image-providers --configured  # Only configured providers
    ai-infra image-models --provider google  # List models for a provider
    ai-infra image-models --all        # List models for all configured providers
    ai-infra image-models --live       # Fetch live models from APIs
"""

from __future__ import annotations

import json

import typer

from ai_infra.imagegen.discovery import (
    SUPPORTED_PROVIDERS,
    is_provider_configured,
    list_all_available_models,
    list_available_models,
    list_models,
)

app = typer.Typer(help="Image generation provider and model discovery")


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
    """List all supported image generation providers."""
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

        typer.echo("\nImage Generation Providers:\n")
        for provider in result:
            status = "✓" if is_provider_configured(provider) else "✗"
            typer.echo(f"  {status} {provider}")
        typer.echo()


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
    live: bool = typer.Option(
        False,
        "--live",
        "-l",
        help="Fetch live models from provider APIs",
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
    """List available image generation models."""
    if not provider and not all_providers:
        typer.echo("Error: Specify --provider <name> or --all")
        raise typer.Exit(1)

    if all_providers:
        if live:
            try:
                result = list_all_available_models(refresh=refresh)
            except Exception as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)
        else:
            # Use static models for all configured providers
            result = {}
            for prov in SUPPORTED_PROVIDERS:
                if is_provider_configured(prov):
                    result[prov] = list_models(prov)

        if output_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            if not result:
                typer.echo("No configured providers found.")
                raise typer.Exit(1)

            typer.echo("\nImage Generation Models:\n")
            for prov, models in result.items():
                typer.echo(f"{prov} ({len(models)} models):")
                for model in models[:15]:  # Show first 15
                    typer.echo(f"  • {model}")
                if len(models) > 15:
                    typer.echo(f"  ... and {len(models) - 15} more")
                typer.echo()
    else:
        if provider not in SUPPORTED_PROVIDERS:
            typer.echo(f"Error: Unknown provider '{provider}'")
            typer.echo(f"Supported: {', '.join(SUPPORTED_PROVIDERS)}")
            raise typer.Exit(1)

        if live:
            if not is_provider_configured(provider):
                typer.echo(f"Error: Provider '{provider}' is not configured (no API key)")
                raise typer.Exit(1)

            try:
                models = list_available_models(provider, refresh=refresh)
            except Exception as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)
        else:
            models = list_models(provider)

        if output_json:
            typer.echo(json.dumps(models, indent=2))
        else:
            source = "(live)" if live else "(static)"
            typer.echo(f"\n{provider} {source} - {len(models)} models:\n")
            for model in models:
                typer.echo(f"  • {model}")
            typer.echo()


def register(parent: typer.Typer):
    """Register imagegen commands with the parent CLI app."""
    parent.add_typer(app, name="imagegen", help="Image generation discovery")
    # Also add as top-level commands with 'image-' prefix
    parent.command("image-providers")(providers_cmd)
    parent.command("image-models")(models_cmd)
