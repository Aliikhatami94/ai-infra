from __future__ import annotations
from pathlib import Path
from typing import Any, Union
import json
import yaml

from pydantic import BaseModel
from fastapi import FastAPI

from ai_infra.mcp.hosting.models import HostedMcp
from .runtime import mount_mcps, make_lifespan


def _parse_text_to_mapping(text: str) -> dict:
    """Accept JSON or YAML content; try JSON first, then YAML if available."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not yaml:
            raise ValueError("Input is not valid JSON, and PyYAML is not installed to parse YAML.")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping for HostedMcp.")
        return data


def _coerce_hosted_config(config: Any) -> HostedMcp:
    """
    Accept:
      - HostedMcp instance
      - any Pydantic BaseModel (will .model_dump())
      - dict mapping
      - str|Path -> treated as a file path (JSON or YAML)
    Return a validated HostedMcp.
    """
    if isinstance(config, HostedMcp):
        return config

    if isinstance(config, BaseModel):
        return HostedMcp.model_validate(config.model_dump())

    if isinstance(config, dict):
        return HostedMcp.model_validate(config)

    if isinstance(config, (str, Path)):
        path = Path(config)
        text = path.read_text(encoding="utf-8")
        data = _parse_text_to_mapping(text)
        return HostedMcp.model_validate(data)

    raise TypeError(
        f"Unsupported HostedMcp config type: {type(config)!r}; "
        "expected HostedMcp | BaseModel | dict | str | Path."
    )


def add_mcp_to_fastapi(app: FastAPI, config: Union[HostedMcp, BaseModel, dict, str, Path]) -> HostedMcp:
    """
    Mount hosted MCP servers into a FastAPI app.
    Returns the validated HostedMcp used.
    """
    cfg = _coerce_hosted_config(config)
    app.router.lifespan_context = make_lifespan(cfg.servers)
    mount_mcps(app, cfg.servers)
    return cfg