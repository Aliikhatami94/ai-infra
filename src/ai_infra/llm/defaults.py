from ai_infra.llm.providers.providers import Providers

# Default models per provider - used when model_name is None
# These are sensible, cost-effective defaults that work well for most use cases
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google_genai": "gemini-2.0-flash",
    "xai": "grok-3-mini",
}

# Legacy defaults (for backward compat)
MODEL = DEFAULT_MODELS["openai"]
PROVIDER = str(Providers.openai)
