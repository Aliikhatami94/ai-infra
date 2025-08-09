from ai_infra.llm.base import BaseLLM

if __name__ == "__main__":
    llm = BaseLLM()
    res = llm.run_agent(
        messages=[{"role":"user","content":"Hello there"}],
        provider="openai",
        model_name="gpt-4o",
    )
    print(res)