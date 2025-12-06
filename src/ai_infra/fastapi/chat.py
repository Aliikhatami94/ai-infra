"""FastAPI chat endpoint for Agent."""

import asyncio
import json
from typing import AsyncIterator, Callable, Optional, Type

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_infra.fastapi.models import ChatRequest, ChatResponse
from ai_infra.llm import Agent
from ai_infra.streaming import SSECallbacks


def add_agent_endpoint(
    router: APIRouter,
    agent: Optional[Agent] = None,
    *,
    path: str = "/chat",
    get_agent: Optional[Callable[[], Agent]] = None,
    request_model: Type[ChatRequest] = ChatRequest,
    response_model: Type[ChatResponse] = ChatResponse,
) -> None:
    """
    Add a chat endpoint to any FastAPI router.

    Ultra simple - just bring your router and agent:
        router = APIRouter()
        agent = Agent(tools=[...])
        add_agent_endpoint(router, agent)

    Works with svc-infra routers:
        from svc_infra.api.fastapi.routers import public_router
        add_agent_endpoint(public_router, agent)

    Dynamic agent per request:
        def get_agent_for_request():
            # Create agent with request-specific config
            return Agent(tools=[...])

        add_agent_endpoint(router, agent=None, get_agent=get_agent_for_request)

    Custom request/response models:
        class MyRequest(ChatRequest):
            tenant_id: str

        class MyResponse(ChatResponse):
            session_id: str

        add_agent_endpoint(
            router,
            agent,
            request_model=MyRequest,
            response_model=MyResponse,
        )

    Args:
        router: FastAPI APIRouter (or svc-infra public_router/user_router)
        agent: Agent instance (or None if using get_agent)
        path: Endpoint path (default: /chat)
        get_agent: Optional function to create agent per request
        request_model: Pydantic model for requests (extend ChatRequest)
        response_model: Pydantic model for responses (extend ChatResponse)
    """

    @router.post(path, response_model=response_model)
    async def chat_endpoint(request: request_model):
        # Get agent (static or dynamic)
        current_agent = get_agent() if get_agent else agent
        if not current_agent:
            raise ValueError("No agent provided. Pass agent= or get_agent=")

        # Non-streaming: simple response
        if not request.stream:
            result = await current_agent.arun(
                request.message,
                history=request.history,
                provider=request.provider,
                model_name=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            return response_model(
                response=result,
                provider=current_agent._provider,
                model=current_agent._model_name,
            )

        # Streaming: SSE response
        async def sse_generator() -> AsyncIterator[str]:
            callbacks = SSECallbacks(visibility="standard")

            # Run agent with streaming callbacks
            task = asyncio.create_task(
                current_agent.arun(
                    request.message,
                    history=request.history,
                    callbacks=callbacks,
                    provider=request.provider,
                    model_name=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            )

            # Stream events
            async for event in callbacks.stream():
                yield f"event: {event.event}\n"
                yield f"data: {json.dumps(event.data)}\n\n"

            # Wait for agent to finish
            await task
            callbacks.mark_done()

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
        )
