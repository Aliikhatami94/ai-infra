"""Tests for BYOK authentication helpers."""

import os

import pytest

from ai_infra.llm.auth import PROVIDER_ENV_VARS, atemporary_api_key, temporary_api_key


def test_temporary_api_key_restores_original(monkeypatch):
    env_var = PROVIDER_ENV_VARS["openai"]
    monkeypatch.setenv(env_var, "original-key")

    with temporary_api_key("openai", "temp-key"):
        assert os.environ[env_var] == "temp-key"

    assert os.environ[env_var] == "original-key"


@pytest.mark.asyncio
async def test_atemporary_api_key_removes_when_unset(monkeypatch):
    env_var = PROVIDER_ENV_VARS["anthropic"]
    monkeypatch.delenv(env_var, raising=False)

    async with atemporary_api_key("anthropic", "temp-key"):
        assert os.environ[env_var] == "temp-key"

    assert env_var not in os.environ


def test_temporary_api_key_unknown_provider():
    with pytest.raises(ValueError):
        with temporary_api_key("unknown", "temp-key"):
            pass
