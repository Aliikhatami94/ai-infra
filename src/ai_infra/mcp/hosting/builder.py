from __future__ import annotations
from pathlib import Path
from typing import Any, Union
import json
import yaml

from pydantic import BaseModel
from fastapi import FastAPI

from ai_infra.mcp.hosting.models import HostedMcp
from .runtime import make_lifespan_manager, mount_hosted_servers

# ---------- parsing ----------

def _parse_config_text(text: str) -> dict:
    """Parse JSON or YAML into a dict."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not yaml:
            raise ValueError("Input is not valid JSON, and PyYAML is not installed.")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping for HostedMcp.")
        return data

def load_hosted_config(config: Union[HostedMcp, BaseModel, dict, str, Path]) -> HostedMcp:
    """
    Normalize user input (model/dict/filepath) into a validated HostedMcp.
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
        data = _parse_config_text(text)
        return HostedMcp.model_validate(data)
    raise TypeError(
        f"Unsupported HostedMcp config type: {type(config)!r}; "
        "expected HostedMcp | BaseModel | dict | str | Path."
    )

# ---------- public entrypoint ----------

def add_mcp_to_fastapi(app: FastAPI, config: Union[HostedMcp, BaseModel, dict, str, Path]) -> HostedMcp:
    """
    Validate and attach all hosted MCP servers to a FastAPI app.
    Returns the validated config used.
    """
    cfg = load_hosted_config(config)
    app.router.lifespan_context = make_lifespan_manager(cfg.servers)
    mount_hosted_servers(app, cfg.servers)
    return cfg