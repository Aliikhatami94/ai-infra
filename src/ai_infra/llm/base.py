import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

from ai_infra.llm.context import LLMContext
from ai_infra.llm.settings import get_llm_settings

load_dotenv()

class BaseLLM:
    """
    Dynamic LLM setup and agent runner supporting multiple providers, models, and tools.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Use Any for models
        self.tools: List[Any] = []  # Use Any for tools

    def register_tool(self, tool_obj: Any):
        self.tools.append(tool_obj)

    def register_model(self, provider: str, model_name: str, **kwargs):
        key = f"{provider}:{model_name}"
        if key not in self.models:
            self.models[key] = init_chat_model(key, api_key=os.environ.get(f"{provider.upper()}_API_KEY"), **kwargs)
        return self.models[key]

    def select_model(self, _state: Any, runtime: Runtime[LLMContext]) -> Any:
        """
        Select and return a model bound with tools, based on the runtime context.
        """
        ctx = runtime.context
        key = f"{ctx.provider}:{ctx.model_name}"
        if key not in self.models:
            self.register_model(ctx.provider, ctx.model_name)
        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        return model.bind_tools(tools)

    def create_agent(self, tools: Optional[List[Any]] = None) -> Any:
        """
        Create a react agent with the given or default tools.
        """
        agent_tools = tools if tools is not None else self.tools
        return create_react_agent(self.select_model, tools=agent_tools)

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
        Run the agent with the specified provider, model, tools, and extra model kwargs.
        Returns the agent's response.
        """
        model_kwargs = model_kwargs or {}
        self.register_model(provider, model_name, **model_kwargs)
        ctx = LLMContext(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra=extra
        )
        agent = self.create_agent(tools)
        response = agent.invoke({"messages": messages}, context=ctx)
        return response

if __name__ == '__main__':
    llm = BaseLLM()
    settings = get_llm_settings()
    response = llm.run_agent(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        provider="openai",
        model_name="gpt-5-mini",
    )
    print(response)