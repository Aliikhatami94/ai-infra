from ai_infra.llm import CoreLLM, Providers, Models

if __name__ == "__main__":
    llm = CoreLLM()
    res = llm.run_agent(
        messages=[{"role": "user", "content": "Hello there"}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 100,
        }
    )
    print(res)