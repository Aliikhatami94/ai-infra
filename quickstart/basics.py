from ai_infra.llm import CoreLLM, Providers, Models

core = CoreLLM()

def test_agent():
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
    llm = core.set_model(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    res = llm.invoke("Hello, how are you?")
    print(res)

def test_sys_msg():
    model = core.set_model(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    res = model.invoke([
        {"role": "system", "content": "Your name is Alex"},
        {"role": "user", "content": "Hey what is your name?"}
    ])
    print(res)

if __name__ == "__main__":
    test_sys_msg()