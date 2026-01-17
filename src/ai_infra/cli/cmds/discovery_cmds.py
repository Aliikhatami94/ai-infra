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

from ai_infra.cli.output import (
    print_all_models,
    print_cli_error,
    print_models,
    print_providers,
)
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
            print_cli_error(
                "No providers configured",
                hint="Set API key environment variables (e.g., OPENAI_API_KEY)",
            )
            raise typer.Exit(1)

        configured_list = [p for p in result if is_provider_configured(p)]
        print_providers(result, configured=configured_list)


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
        print_cli_error(
            "No provider specified",
            hint="Use --provider <name> or --all to list models",
        )
        raise typer.Exit(1)

    if all_providers:
        try:
            result = list_all_models(refresh=refresh)
        except Exception as e:
            print_cli_error(f"Failed to fetch models: {e}")
            raise typer.Exit(1)

        if output_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            if not result:
                print_cli_error(
                    "No configured providers found",
                    hint="Set API key environment variables",
                )
                raise typer.Exit(1)

            print_all_models(result)
    else:
        if provider not in SUPPORTED_PROVIDERS:
            print_cli_error(
                f"Unknown provider: {provider}",
                hint=f"Supported: {', '.join(SUPPORTED_PROVIDERS)}",
            )
            raise typer.Exit(1)

        if not is_provider_configured(provider):
            print_cli_error(
                f"Provider '{provider}' is not configured",
                hint=f"Set the {provider.upper()}_API_KEY environment variable",
            )
            raise typer.Exit(1)

        try:
            models = list_models(provider, refresh=refresh)
        except Exception as e:
            print_cli_error(f"Failed to fetch models: {e}")
            raise typer.Exit(1)

        if output_json:
            typer.echo(json.dumps(models, indent=2))
        else:
            print_models(provider, models)


def register(parent: typer.Typer):
    """Register discovery commands with the parent CLI app."""
    # Register as top-level commands only (no sub-app to avoid duplication)
    parent.command("providers", rich_help_panel="Discovery")(providers_cmd)
    parent.command("models", rich_help_panel="Discovery")(models_cmd)
