"""ElevenLabs provider configuration.

ElevenLabs supports:
- TTS: High-quality text-to-speech with voice cloning

Known for natural-sounding voices and voice cloning capabilities.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

ELEVENLABS = ProviderConfig(
    name="elevenlabs",
    display_name="ElevenLabs",
    env_var="ELEVENLABS_API_KEY",
    capabilities={
        ProviderCapability.TTS: CapabilityConfig(
            models=[
                "eleven_multilingual_v2",
                "eleven_turbo_v2",
                "eleven_turbo_v2_5",
                "eleven_monolingual_v1",
            ],
            default_model="eleven_multilingual_v2",
            voices=[
                # Default/built-in voices (more available via API)
                "Rachel",
                "Drew",
                "Clyde",
                "Paul",
                "Domi",
                "Dave",
                "Fin",
                "Sarah",
                "Antoni",
                "Thomas",
                "Charlie",
                "George",
                "Emily",
                "Elli",
                "Callum",
                "Patrick",
                "Harry",
                "Liam",
                "Dorothy",
                "Josh",
                "Arnold",
                "Charlotte",
                "Matilda",
                "Matthew",
                "James",
                "Joseph",
                "Jeremy",
                "Michael",
                "Ethan",
                "Gigi",
                "Freya",
                "Grace",
                "Daniel",
                "Lily",
                "Serena",
                "Adam",
                "Nicole",
                "Bill",
                "Jessie",
                "Sam",
                "Glinda",
                "Giovanni",
                "Mimi",
            ],
            default_voice="Rachel",
            features=["streaming", "voice_cloning", "multilingual", "emotion"],
            extra={
                "voice_fetch_endpoint": "/v1/voices",
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(ELEVENLABS)
