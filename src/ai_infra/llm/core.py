import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

from ai_infra.llm.settings import ModelSettings
from ai_infra.llm.providers import Providers
from ai_infra.llm.models import Models
from ai_infra.llm.utils import validate_provider_and_model, build_model_key, initialize_model

load_dotenv()

class CoreLLM:
    """
    Dynamic LLM setup and agent runner supporting multiple providers, models, and tools.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Use Any for models
        self.tools: List[Any] = []  # Use Any for tools

    def set_model(self, provider: str, model_name: str, **kwargs):
        validate_provider_and_model(provider, model_name)
        key = build_model_key(provider, model_name)
        if key not in self.models:
            self.models[key] = initialize_model(key, provider, **kwargs)
        return self.models[key]

    def _select_model(self, _state: Any, runtime: Runtime[ModelSettings]) -> Any:
        """
        Select and return a model bound with tools, based on the runtime context.
        """
        ctx = runtime.context
        key = f"{ctx.provider}:{ctx.model_name}"
        if key not in self.models:
            self.set_model(ctx.provider, ctx.model_name)
        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        return model.bind_tools(tools)

    def create_agent(self, tools: Optional[List[Any]] = None) -> Any:
        """
        Create a react agent with the given or default tools.
        """
        agent_tools = tools if tools is not None else self.tools
        return create_react_agent(model=self._select_model, tools=agent_tools)

    def run_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        # No need to validate again, set_model already validates
        model_kwargs = model_kwargs or {}
        self.set_model(provider, model_name, **model_kwargs)
        ctx = ModelSettings(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra=extra
        )
        agent = self.create_agent(tools)
        response = agent.invoke({"messages": messages}, context=ctx)
        return response