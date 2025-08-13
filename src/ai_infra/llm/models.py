from enum import Enum

class OpenAIModels(Enum):
    gpt_4o = "gpt-4o"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_5 = "gpt-5"
    gpt_5_nano = "gpt-5-nano"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_chat = "gpt-5-chat"

class AnthropicModels(Enum):
    claude_3_5_sonnet_latest = "claude-3-5-sonnet-latest"
    claude_3_7_sonnet_latest = "claude-3-7-sonnet-latest"
    claude_3_5_haiku_latest = "claude-3-5-haiku-latest"

class GoogleGenAIModels(Enum):
    gemini_2_5_flash = "gemini-2.5-flash"

class XAIModels(Enum):
    grok_3 = "grok-3"
    grok_3_mini = "grok-3-mini"

class Models:
    openai = OpenAIModels
    anthropic = AnthropicModels
    google_genai = GoogleGenAIModels
    xai = XAIModels
