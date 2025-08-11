from ai_infra.llm import CoreLLM, Providers, Models

def test_agent():
    core = CoreLLM()
    res = core.run_agent(
        messages=[{"role": "user", "content": "Hello there"}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 100,
        }
    )
    print(res)

def test_llm():
    core = CoreLLM()
    llm = core.set_model(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    res = llm.invoke("Hello, how are you?")
    print(res)

if __name__ == "__main__":
    test_llm()