import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

from ai_infra.llm.core.settings import ModelSettings
from ai_infra.llm.core.providers import Providers
from ai_infra.llm.core.models import Models

load_dotenv()

class CoreLLM:
    """
    Dynamic LLM setup and agent runner supporting multiple providers, models, and tools.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Use Any for models
        self.tools: List[Any] = []  # Use Any for tools

    def set_model(self, provider: str, model_name: str, **kwargs):
        # Accept provider as Providers.<provider> (e.g., Providers.openai)
        # Accept model_name as Models.<provider>.<model>.value (e.g., Models.openai.gpt_4o.value)
        provider_names = [v for k, v in Providers.__dict__.items() if not k.startswith('__') and not callable(v)]
        if provider not in provider_names:
            raise ValueError(f"Unknown provider: {provider}")
        valid_models = getattr(Models, provider)
        if model_name not in [m.value for m in valid_models]:
            raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")
        key = f"{provider}:{model_name}"
        if key not in self.models:
            self.models[key] = init_chat_model(
                key,
                api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
                **kwargs
            )
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
        provider_names = [v for k, v in Providers.__dict__.items() if not k.startswith('__') and not callable(v)]
        if provider not in provider_names:
            raise ValueError(f"Unknown provider: {provider}")
        valid_models = getattr(Models, provider)
        if model_name not in [m.value for m in valid_models]:
            raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")
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