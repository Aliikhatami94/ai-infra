# CLI Reference

> Command-line interface for ai-infra.

## Installation

The CLI is installed automatically with the package:

```bash
pip install ai-infra
```

---

## Commands

### `ai-infra version`

Display version information:

```bash
ai-infra version
```

Output:
```
ai-infra 0.1.0
Python 3.11.5
```

---

### `ai-infra providers`

List available providers and their status:

```bash
ai-infra providers
```

Output:
```
Provider        Status      Capabilities
────────────────────────────────────────────
openai          ✓ ready     CHAT, EMBEDDINGS, TTS, IMAGEGEN
anthropic       ✓ ready     CHAT
google_genai    ✗ not set   CHAT, EMBEDDINGS
xai             ✗ not set   CHAT
elevenlabs      ✓ ready     TTS
deepgram        ✗ not set   STT
stability       ✗ not set   IMAGEGEN
replicate       ✗ not set   IMAGEGEN
voyage          ✓ ready     EMBEDDINGS
cohere          ✗ not set   EMBEDDINGS
```

Options:
```bash
# Show only configured providers
ai-infra providers --configured

# Filter by capability
ai-infra providers --capability CHAT

# JSON output
ai-infra providers --json
```

---

### `ai-infra check`

Verify provider configuration:

```bash
ai-infra check openai
```

Output:
```
Checking openai...
  ✓ API key found (OPENAI_API_KEY)
  ✓ Connection successful
  ✓ Model access: gpt-4o, gpt-4o-mini, dall-e-3

Provider openai is ready.
```

Check all providers:
```bash
ai-infra check --all
```

---

### `ai-infra generate`

Generate text using configured LLM:

```bash
ai-infra generate "Write a haiku about coding"
```

Output:
```
Lines of code unfold,
Logic dances, bugs emerge,
Debug, repeat, grow.
```

Options:
```bash
# Specify provider and model
ai-infra generate "Hello" --provider openai --model gpt-4o-mini

# Set temperature
ai-infra generate "Tell me a joke" --temperature 0.9

# Stream output
ai-infra generate "Tell me a story" --stream

# Read from file
ai-infra generate --file prompt.txt

# Output to file
ai-infra generate "Generate code" --output result.txt
```

---

### `ai-infra embed`

Generate embeddings:

```bash
ai-infra embed "Hello world"
```

Output:
```
Embedding (1536 dimensions):
[0.0123, -0.0456, 0.0789, ...]

Provider: openai
Model: text-embedding-3-small
```

Options:
```bash
# Specify provider
ai-infra embed "text" --provider voyage

# Output as JSON
ai-infra embed "text" --json

# Multiple texts
ai-infra embed "text1" "text2" "text3"
```

---

### `ai-infra image`

Generate images:

```bash
ai-infra image "A sunset over mountains"
```

Output:
```
Image generated successfully!
Saved to: sunset_20240115_103000.png

Provider: openai
Model: dall-e-3
Size: 1024x1024
```

Options:
```bash
# Specify provider and model
ai-infra image "prompt" --provider stability --model stable-diffusion-xl

# Set size
ai-infra image "prompt" --size 1792x1024

# Output path
ai-infra image "prompt" --output my_image.png

# Generate multiple
ai-infra image "prompt" --n 4
```

---

### `ai-infra speak`

Text-to-speech generation:

```bash
ai-infra speak "Hello, how are you today?"
```

Output:
```
Audio generated successfully!
Saved to: speech_20240115_103000.mp3

Provider: openai
Model: tts-1
Voice: alloy
Duration: 2.5s
```

Options:
```bash
# Specify voice
ai-infra speak "Hello" --voice nova

# Output path
ai-infra speak "Hello" --output greeting.mp3

# Provider
ai-infra speak "Hello" --provider elevenlabs
```

---

### `ai-infra transcribe`

Speech-to-text transcription:

```bash
ai-infra transcribe audio.mp3
```

Output:
```
Transcription:
"Hello, this is a test recording."

Provider: openai
Model: whisper-1
Duration: 5.2s
```

Options:
```bash
# Specify language
ai-infra transcribe audio.mp3 --language en

# Output to file
ai-infra transcribe audio.mp3 --output transcript.txt

# JSON output with timestamps
ai-infra transcribe audio.mp3 --json --timestamps
```

---

### `ai-infra mcp`

MCP server commands:

```bash
# List MCP servers
ai-infra mcp list

# Start MCP server
ai-infra mcp serve --port 8080

# Generate MCP from OpenAPI
ai-infra mcp from-openapi api.yaml --output mcp_tools.py
```

---

### `ai-infra config`

Manage configuration:

```bash
# Show current config
ai-infra config show

# Set a value
ai-infra config set default_provider openai

# Get a value
ai-infra config get default_provider

# List all settings
ai-infra config list
```

---

## Global Options

Available for all commands:

```bash
# Verbose output
ai-infra --verbose generate "Hello"

# Quiet mode (errors only)
ai-infra --quiet generate "Hello"

# Specify config file
ai-infra --config custom.yaml generate "Hello"

# JSON output (where supported)
ai-infra --json providers

# Help
ai-infra --help
ai-infra generate --help
```

---

## Environment Variables

The CLI respects these environment variables:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Defaults
AI_INFRA_DEFAULT_PROVIDER=openai
AI_INFRA_DEFAULT_MODEL=gpt-4o
AI_INFRA_DEFAULT_EMBEDDING_PROVIDER=openai

# Configuration
AI_INFRA_CONFIG_PATH=~/.ai-infra/config.yaml
AI_INFRA_LOG_LEVEL=INFO

# Output
AI_INFRA_OUTPUT_FORMAT=text  # or "json"
```

---

## Configuration File

Create `~/.ai-infra/config.yaml`:

```yaml
# Default providers
default_provider: openai
default_model: gpt-4o
default_embedding_provider: voyage

# Generation defaults
temperature: 0.7
max_tokens: 4096

# Image generation defaults
image_size: 1024x1024
image_quality: standard

# Output preferences
output_format: text
color: true
```

---

## Examples

### Quick Chat

```bash
# Simple generation
ai-infra generate "Explain quantum computing in one sentence"

# With specific model
ai-infra generate "Hello" -p anthropic -m claude-sonnet-4-20250514
```

### Batch Processing

```bash
# Generate embeddings for multiple files
for file in *.txt; do
  ai-infra embed --file "$file" --json >> embeddings.jsonl
done
```

### Image Generation Pipeline

```bash
# Generate and resize
ai-infra image "A logo for a tech company" --output logo.png
convert logo.png -resize 512x512 logo_small.png
```

### Transcription Workflow

```bash
# Transcribe and summarize
ai-infra transcribe meeting.mp3 --output transcript.txt
ai-infra generate --file transcript.txt "Summarize this meeting transcript"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Provider not configured |
| 4 | Rate limit exceeded |
| 5 | Validation error |

---

## See Also

- [Getting Started](getting-started.md) - Installation and setup
- [Providers](core/providers.md) - Provider configuration
- [LLM](core/llm.md) - Programmatic usage
