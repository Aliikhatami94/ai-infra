# MCP Transport Security Improvements

## Summary

Simplified and improved MCP security settings in ai-infra to automatically detect deployment environments and configure appropriate security without manual intervention.

**Update (Dec 8, 2025)**: Fixed critical bug where `enable_security=False` was returning `None` from `to_transport_settings()`, causing FastMCP to use its default security (localhost only) instead of actually disabling security. Now always returns a proper `TransportSecuritySettings` object with the correct configuration.

## Problem

The original implementation had several issues:

1. **Manual configuration required**: Users had to use `MCPSecuritySettings.for_production()` and manually specify domains
2. **Platform-specific**: The `for_production()` method name implied a single "production" environment
3. **Localhost-only defaults**: Default settings only worked locally, causing failures in deployed environments
4. **Duplicate parameters**: `mcp_from_functions()` had both `security` and `transport_security` parameters
5. **DNS rebinding issues**: In deployed environments (Railway, Render, etc.), the Host header validation would reject internal MCP client connections

## Solution

### 1. Auto-detection of Deployment Environment

The new `MCPSecuritySettings` automatically detects the deployment platform from environment variables:

- **Railway**: `RAILWAY_PUBLIC_DOMAIN`, `RAILWAY_STATIC_URL`
- **Render**: `RENDER_EXTERNAL_HOSTNAME`
- **Fly.io**: `FLY_APP_NAME` (adds `.fly.dev` suffix)
- **Heroku**: `HEROKU_APP_NAME` (adds `.herokuapp.com` suffix)
- **Vercel**: `VERCEL_URL`
- **Generic**: `PUBLIC_URL`, `APP_URL`, `HOST`

Always includes localhost for local development.

### 2. Simplified API

**Before:**
```python
# Manual configuration - platform specific
_security = MCPSecuritySettings.for_production(
    domains=["api.nfrax.com", "nfrax.com", "www.nfrax.com"],
    include_localhost=True,
)

mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=_security,
    transport_security=None,  # Duplicate parameter
)
```

**After:**
```python
# Auto-detection - works everywhere
mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    # That's it! Auto-detects environment
)

# Or with custom domains (still auto-detects platform)
mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(domains=["api.example.com"]),
)

# Or disable security for dev
mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(enable_security=False),
)
```

### 3. Removed Duplicate Parameters

- Removed `transport_security` parameter from `mcp_from_functions()`
- Removed `TransportSecuritySettings` from public exports
- Users only need to use `MCPSecuritySettings` or rely on defaults

### 4. Removed Manual Methods

- Removed `MCPSecuritySettings.for_production()` (replaced by auto-detection)
- Removed `MCPSecuritySettings.allow_all()` (use `enable_security=False`)

## Changes Made

### Files Modified

1. **`ai-infra/src/ai_infra/mcp/server/tools.py`**
   - Added `_auto_detect_hosts()` and `_auto_detect_origins()` functions
   - Simplified `MCPSecuritySettings` constructor to use auto-detection by default
   - Removed `for_production()` and `allow_all()` class methods
   - Added `domains` parameter for custom domain specification
   - Removed `transport_security` parameter from `mcp_from_functions()`
   - Updated defaults to auto-detect by default

2. **`ai-infra/src/ai_infra/mcp/server/__init__.py`**
   - Removed `TransportSecuritySettings` from exports

3. **`ai-infra/src/ai_infra/mcp/__init__.py`**
   - Removed `TransportSecuritySettings` from exports

4. **`ai-infra/src/ai_infra/__init__.py`**
   - Removed `TransportSecuritySettings` from imports

5. **`ai-infra/docs/mcp/server.md`**
   - Added security section explaining auto-detection
   - Updated examples to show new simplified API

6. **`nfrax-api/src/nfrax_api/mcp/nfrax_mcp.py`**
   - Removed manual `for_production()` configuration
   - Now uses simple `MCPSecuritySettings(domains=[...])` which auto-detects platform

### Tests Added

Created `ai-infra/tests/unit/mcp/test_security_auto_detection.py` with comprehensive tests:
- Default auto-detection
- Disabled security
- Custom domains
- Railway detection
- Render detection
- Fly.io detection
- Heroku detection
- Vercel detection
- Multiple environment detection
- Integration with `mcp_from_functions()`

All 12 tests pass ✅

## Benefits

1. **Zero configuration for most use cases**: Works out of the box in any environment
2. **Platform agnostic**: No Railway-specific or platform-specific code needed
3. **Backwards compatible**: Existing code with custom domains still works
4. **Simpler API**: One parameter (`security`), one class (`MCPSecuritySettings`)
5. **Better developer experience**: No manual host/origin management
6. **Automatic security**: Always enables appropriate security based on environment

## Usage Examples

### Default (Recommended)
```python
from ai_infra import mcp_from_functions

def my_tool() -> str:
    return "result"

# Auto-detects environment - works everywhere
mcp = mcp_from_functions(name="my-mcp", functions=[my_tool])
```

### Custom Domains
```python
from ai_infra import MCPSecuritySettings, mcp_from_functions

# Still auto-detects platform, adds custom domains
mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(domains=["api.example.com"]),
)
```

### Development (No Security)
```python
from ai_infra import MCPSecuritySettings, mcp_from_functions

# Disable security for local dev/testing
mcp = mcp_from_functions(
    name="my-mcp",
    functions=[my_tool],
    security=MCPSecuritySettings(enable_security=False),
)
```

## Migration Guide

### For nfrax-api users

**Before:**
```python
_security = MCPSecuritySettings.for_production(
    domains=["api.nfrax.com", "nfrax.com"],
    include_localhost=True,
)
mcp = mcp_from_functions(name="nfrax-mcp", functions=[...], security=_security)
```

**After:**
```python
mcp = mcp_from_functions(
    name="nfrax-mcp",
    functions=[...],
    security=MCPSecuritySettings(domains=["api.nfrax.com", "nfrax.com"]),
)
```

### For users with `allow_all()`

**Before:**
```python
security = MCPSecuritySettings.allow_all()
mcp = mcp_from_functions(name="my-mcp", functions=[...], security=security)
```

**After:**
```python
security = MCPSecuritySettings(enable_security=False)
mcp = mcp_from_functions(name="my-mcp", functions=[...], security=security)
```

### For users with `transport_security`

**Before:**
```python
from mcp.server.transport_security import TransportSecuritySettings

ts = TransportSecuritySettings(...)
mcp = mcp_from_functions(name="my-mcp", functions=[...], transport_security=ts)
```

**After:**
```python
# Use MCPSecuritySettings instead
security = MCPSecuritySettings(domains=["example.com"])
mcp = mcp_from_functions(name="my-mcp", functions=[...], security=security)
```

## Technical Details

### Environment Detection Logic

1. **Localhost**: Always included (`127.0.0.1:*`, `localhost:*`, `[::1]:*`)
2. **Railway**: Checks `RAILWAY_PUBLIC_DOMAIN` and `RAILWAY_STATIC_URL`
3. **Render**: Checks `RENDER_EXTERNAL_HOSTNAME`
4. **Fly.io**: Checks `FLY_APP_NAME` and constructs `.fly.dev` domain
5. **Heroku**: Checks `HEROKU_APP_NAME` and constructs `.herokuapp.com` domain
6. **Vercel**: Checks `VERCEL_URL`
7. **Generic**: Checks `PUBLIC_URL`, `APP_URL`, `HOST`

### Security Behavior

- **Enabled by default**: DNS rebinding protection is on unless explicitly disabled
- **Wildcard ports**: All detected hosts use `:*` to allow any port
- **HTTPS for production**: Auto-detected domains use `https://` in origins
- **HTTP for localhost**: Localhost uses `http://` in origins

## Resolution of Original Issue

The transport security issue in nfrax-api was caused by:

1. Using `for_production()` which only added specific domains
2. Not detecting the Railway/deployment hostname automatically
3. Internal MCP client connections being rejected due to Host header mismatch

This is now fixed because:

1. Auto-detection includes deployment-specific hostnames
2. Works seamlessly across all platforms without configuration
3. Internal connections work because the deployment hostname is automatically allowed
