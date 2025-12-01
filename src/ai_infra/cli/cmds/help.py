_HELP = """\
ai-infra — Production-ready SDK for building AI applications

QUICK START:

  Start an interactive chat:
    $ ai-infra chat

  Send a one-shot message:
    $ ai-infra chat -m "What is Python?"

  List available providers:
    $ ai-infra providers

COMMANDS:

  Chat & Generation:
    chat              Interactive chat REPL or one-shot messages

  Discovery:
    providers         List all LLM providers
    models            List models for a provider

  Image Generation:
    image-providers   List image generation providers
    image-models      List available image models

  Multimodal:
    tts-providers     List text-to-speech providers
    tts-voices        List available voices
    tts-models        List TTS models
    stt-providers     List speech-to-text providers
    stt-models        List STT models
    audio-models      List audio LLM models

  MCP Tools:
    add-shim          Add MCP stdio publisher shim
    remove-shim       Remove MCP stdio publisher shim
    chmod             Make shim executable

EXAMPLES:

  # Interactive chat with OpenAI
  $ ai-infra chat --provider openai --model gpt-4o

  # One-shot with system prompt
  $ ai-infra chat -m "Explain Docker" -s "Be concise"

  # List configured providers
  $ ai-infra providers --configured

  # List OpenAI models
  $ ai-infra models --provider openai

  # Generate image (coming soon)
  $ ai-infra generate-image "A sunset over mountains"

ENVIRONMENT:

  Set API keys for the providers you want to use:
    OPENAI_API_KEY      OpenAI
    ANTHROPIC_API_KEY   Anthropic
    GOOGLE_API_KEY      Google
    XAI_API_KEY         xAI
    ELEVENLABS_API_KEY  ElevenLabs (TTS)
    DEEPGRAM_API_KEY    Deepgram (STT)

DOCUMENTATION:

  Full docs: https://github.com/Aliikhatami94/ai-infra/tree/main/docs
"""
