"""Tests for FastAPI chat endpoint integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ai_infra import Agent, ChatRequest, ChatResponse, add_agent_endpoint


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.arun = AsyncMock(return_value="Test response")
    agent._provider = "openai"
    agent._model_name = "gpt-4o-mini"
    return agent


def test_add_agent_endpoint_basic():
    """Test basic endpoint creation."""
    router = APIRouter()
    agent = Agent()
    add_agent_endpoint(router, agent)

    # Check route was added
    routes = [r.path for r in router.routes]
    assert "/chat" in routes


def test_add_agent_endpoint_custom_path():
    """Test custom path."""
    router = APIRouter()
    agent = Agent()
    add_agent_endpoint(router, agent, path="/custom")

    routes = [r.path for r in router.routes]
    assert "/custom" in routes


@pytest.mark.asyncio
async def test_endpoint_non_streaming(mock_agent):
    """Test non-streaming response."""
    app = FastAPI()
    router = APIRouter()
    add_agent_endpoint(router, mock_agent)
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] == "Test response"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"


def test_custom_request_model():
    """Test custom request model."""

    class MyRequest(ChatRequest):
        tenant_id: str

    router = APIRouter()
    agent = Agent()
    add_agent_endpoint(router, agent, request_model=MyRequest)

    # Check endpoint was created
    routes = [r.path for r in router.routes]
    assert "/chat" in routes


def test_get_agent_callable():
    """Test dynamic agent creation."""
    router = APIRouter()

    def get_agent():
        return Agent()

    add_agent_endpoint(router, agent=None, get_agent=get_agent)

    # Verify endpoint was created
    routes = [r.path for r in router.routes]
    assert "/chat" in routes


def test_no_agent_provided_raises():
    """Test that missing agent raises error."""
    router = APIRouter()
    add_agent_endpoint(router, agent=None, get_agent=None)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=True)

    # Error should happen when endpoint is called
    with pytest.raises(ValueError, match="No agent provided"):
        client.post("/chat", json={"message": "Hi", "stream": False})
