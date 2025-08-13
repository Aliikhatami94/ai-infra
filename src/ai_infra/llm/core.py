from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

from ai_infra.llm.settings import ModelSettings
from ai_infra.llm.utils import validate_provider_and_model, build_model_key, initialize_model

load_dotenv()

class CoreLLM:
    """
    Dynamic LLM setup and agent runner supporting multiple providers, models, and tools.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Use Any for models
        self.tools: List[Any] = []  # Use Any for tools

    # Model Management Methods
    def set_model(self, provider: str, model_name: str, **kwargs):
        """Set up and cache a model for the given provider and model name."""
        validate_provider_and_model(provider, model_name)
        key = build_model_key(provider, model_name)
        if key not in self.models:
            self.models[key] = initialize_model(key, provider, **kwargs)
        return self.models[key]

    def _select_model(self, _state: Any, runtime: Runtime[ModelSettings]) -> Any:
        """Select and return a model bound with tools, based on the runtime context."""
        ctx = runtime.context
        key = build_model_key(ctx.provider, ctx.model_name)
        if key not in self.models:
            self.set_model(ctx.provider, ctx.model_name)
        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        return model.bind_tools(tools)

    # Agent Management Methods
    def create_agent(self, tools: Optional[List[Any]] = None) -> Any:
        """Create a react agent with the given or default tools."""
        agent_tools = tools if tools is not None else self.tools
        return create_react_agent(model=self._select_model, tools=agent_tools)

    def _prepare_agent(self, provider, model_name, tools=None, extra=None, model_kwargs=None):
        """
        Internal helper to set up model, context, and agent instance.
        """
        model_kwargs = model_kwargs or {}
        self.set_model(provider, model_name, **model_kwargs)
        context = ModelSettings(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra=extra
        )
        agent = self.create_agent(tools)
        return agent, context

    async def arun_agent(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Async version of run_agent. Runs an agent with the specified configuration using ainvoke.
        """
        agent, context = self._prepare_agent(provider, model_name, tools, extra, model_kwargs)
        return await agent.ainvoke({"messages": messages}, context=context)

    def run_agent(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Synchronous version of run_agent_async. Runs an agent with the specified configuration using invoke.
        """
        agent, context = self._prepare_agent(provider, model_name, tools, extra, model_kwargs)
        return agent.invoke({"messages": messages}, context=context)
