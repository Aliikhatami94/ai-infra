"""Tests for MCP security auto-detection."""

import os

import pytest

from ai_infra.mcp import MCPSecuritySettings, mcp_from_functions


def test_default_auto_detection():
    """Test default auto-detection includes localhost."""
    security = MCPSecuritySettings()

    assert security.enabled is True
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "localhost:*" in security.allowed_hosts
    assert "[::1]:*" in security.allowed_hosts

    assert "http://127.0.0.1:*" in security.allowed_origins
    assert "http://localhost:*" in security.allowed_origins


def test_disabled_security():
    """Test disabling security."""
    security = MCPSecuritySettings(enable_security=False)

    assert security.enabled is False
    assert security.allowed_hosts == []
    assert security.allowed_origins == []

    # Should return TransportSecuritySettings with disabled flag
    transport_settings = security.to_transport_settings()
    assert transport_settings is not None
    assert transport_settings.enable_dns_rebinding_protection is False


def test_custom_domains():
    """Test custom domains."""
    security = MCPSecuritySettings(domains=["api.example.com", "example.com"])

    assert security.enabled is True
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "api.example.com:*" in security.allowed_hosts
    assert "example.com:*" in security.allowed_hosts

    assert "https://api.example.com" in security.allowed_origins
    assert "https://example.com" in security.allowed_origins


def test_railway_detection(monkeypatch):
    """Test Railway environment detection."""
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "myapp.railway.app")

    security = MCPSecuritySettings()

    assert "myapp.railway.app:*" in security.allowed_hosts
    assert "https://myapp.railway.app" in security.allowed_origins


def test_render_detection(monkeypatch):
    """Test Render environment detection."""
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "myapp.onrender.com")

    security = MCPSecuritySettings()

    assert "myapp.onrender.com:*" in security.allowed_hosts
    assert "https://myapp.onrender.com" in security.allowed_origins


def test_fly_detection(monkeypatch):
    """Test Fly.io environment detection."""
    monkeypatch.setenv("FLY_APP_NAME", "myapp")

    security = MCPSecuritySettings()

    assert "myapp.fly.dev:*" in security.allowed_hosts


def test_heroku_detection(monkeypatch):
    """Test Heroku environment detection."""
    monkeypatch.setenv("HEROKU_APP_NAME", "myapp")

    security = MCPSecuritySettings()

    assert "myapp.herokuapp.com:*" in security.allowed_hosts


def test_vercel_detection(monkeypatch):
    """Test Vercel environment detection."""
    monkeypatch.setenv("VERCEL_URL", "myapp.vercel.app")

    security = MCPSecuritySettings()

    assert "myapp.vercel.app:*" in security.allowed_hosts
    assert "https://myapp.vercel.app" in security.allowed_origins


def test_multiple_environments(monkeypatch):
    """Test detection with multiple environment variables."""
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "myapp.railway.app")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "myapp.onrender.com")

    security = MCPSecuritySettings()

    assert "myapp.railway.app:*" in security.allowed_hosts
    assert "myapp.onrender.com:*" in security.allowed_hosts


def test_mcp_from_functions_with_auto_security():
    """Test mcp_from_functions uses auto-detection by default."""

    def my_tool() -> str:
        return "test"

    mcp = mcp_from_functions(name="test", functions=[my_tool])

    # Should create successfully without explicit security
    assert mcp is not None


def test_mcp_from_functions_with_custom_security():
    """Test mcp_from_functions with custom security."""

    def my_tool() -> str:
        return "test"

    security = MCPSecuritySettings(domains=["api.example.com"])
    mcp = mcp_from_functions(name="test", functions=[my_tool], security=security)

    assert mcp is not None


def test_mcp_from_functions_with_disabled_security():
    """Test mcp_from_functions with disabled security."""

    def my_tool() -> str:
        return "test"

    security = MCPSecuritySettings(enable_security=False)
    mcp = mcp_from_functions(name="test", functions=[my_tool], security=security)

    assert mcp is not None

    # Verify transport security is properly configured
    # When security is disabled, transport_security should have enable_dns_rebinding_protection=False
    if hasattr(mcp, "_transport_security"):
        transport_sec = mcp._transport_security
        if transport_sec is not None:
            assert transport_sec.enable_dns_rebinding_protection is False
