from ai_infra.llm import CoreLLM, Providers, Models

core = CoreLLM()

def test_agent():
    res = core.run_agent(
        messages=[{"role": "user", "content": "What is your namea?"}],
        provider=Providers.google_genai,
        model_name=Models.google_genai.gemini_2_5_flash.value,
        model_kwargs={
            "temperature": 0.7,
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
    test_agent()