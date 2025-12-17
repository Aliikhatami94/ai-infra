"""OpenAI provider configuration.

OpenAI supports:
- Chat: GPT-4o, GPT-4, GPT-3.5-turbo, o1 models
- Embeddings: text-embedding-3-small/large
- TTS: tts-1, tts-1-hd
- STT: whisper-1
- ImageGen: DALL-E 2/3
- Realtime: gpt-4o-realtime-preview
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

OPENAI = ProviderConfig(
    name="openai",
    display_name="OpenAI",
    env_var="OPENAI_API_KEY",
    capabilities={
        ProviderCapability.CHAT: CapabilityConfig(
            models=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4o-2024-11-20",
                "gpt-4o-2024-08-06",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1",
                "o1-mini",
                "o1-preview",
            ],
            default_model="gpt-5.1",
            features=[
                "streaming",
                "function_calling",
                "vision",
                "json_mode",
                "structured_output",
            ],
        ),
        ProviderCapability.EMBEDDINGS: CapabilityConfig(
            models=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            default_model="text-embedding-3-small",
            extra={
                "dimensions": {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536,
                }
            },
        ),
        ProviderCapability.TTS: CapabilityConfig(
            models=["tts-1", "tts-1-hd"],
            default_model="tts-1",
            voices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            default_voice="alloy",
            features=["streaming"],
        ),
        ProviderCapability.STT: CapabilityConfig(
            models=["whisper-1"],
            default_model="whisper-1",
            features=["timestamps", "word_timestamps", "language_detection"],
        ),
        ProviderCapability.IMAGEGEN: CapabilityConfig(
            models=["dall-e-2", "dall-e-3"],
            default_model="dall-e-3",
            features=["revised_prompt", "hd_quality", "style"],
            extra={
                "sizes": {
                    "dall-e-2": ["256x256", "512x512", "1024x1024"],
                    "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
                }
            },
        ),
        ProviderCapability.REALTIME: CapabilityConfig(
            models=[
                "gpt-4o-realtime-preview",
                "gpt-4o-realtime-preview-2024-10-01",
                "gpt-4o-realtime-preview-2024-12-17",
            ],
            default_model="gpt-4o-realtime-preview",
            voices=[
                "alloy",
                "ash",
                "ballad",
                "coral",
                "echo",
                "sage",
                "shimmer",
                "verse",
            ],
            default_voice="alloy",
            features=["vad", "function_calling", "transcription", "interruption"],
            extra={
                "websocket_url": "wss://api.openai.com/v1/realtime",
                "audio_format": "pcm16",
                "sample_rate": 24000,
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(OPENAI)
