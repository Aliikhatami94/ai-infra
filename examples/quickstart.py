from ai_infra.llm.core import BaseLLM
from ai_infra.llm.core.providers import Providers
from ai_infra.llm.core.models import Models

if __name__ == "__main__":
    llm = BaseLLM()
    res = llm.run_agent(
        messages=[{"role": "user", "content": "Hello there"}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    print(res)