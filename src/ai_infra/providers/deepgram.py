"""Deepgram provider configuration.

Deepgram supports:
- STT: Fast, accurate speech-to-text with streaming support

Known for real-time transcription and speaker diarization.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

DEEPGRAM = ProviderConfig(
    name="deepgram",
    display_name="Deepgram",
    env_var="DEEPGRAM_API_KEY",
    capabilities={
        ProviderCapability.STT: CapabilityConfig(
            models=[
                "nova-2",
                "nova-2-general",
                "nova-2-meeting",
                "nova-2-phonecall",
                "nova-2-voicemail",
                "nova-2-finance",
                "nova-2-conversationalai",
                "nova-2-video",
                "nova-2-medical",
                "nova-2-drivethru",
                "nova-2-automotive",
                "nova",
                "enhanced",
                "base",
            ],
            default_model="nova-2",
            features=[
                "streaming",
                "timestamps",
                "diarization",
                "smart_formatting",
                "punctuation",
                "profanity_filter",
                "redaction",
                "topics",
                "sentiment",
                "summarization",
            ],
            extra={
                "languages": [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "nl",
                    "ja",
                    "ko",
                    "zh",
                ],
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(DEEPGRAM)
