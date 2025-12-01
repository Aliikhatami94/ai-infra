# Vision

> Add images to chat for visual understanding and analysis.

## Quick Start

```python
from ai_infra import LLM

llm = LLM()
response = llm.chat(
    "What's in this image?",
    images=["photo.jpg"]
)
print(response)
```

---

## Supported Providers

| Provider | Models | Max Images |
|----------|--------|------------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo | Multiple |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku | Multiple |
| Google | gemini-2.0-flash, gemini-1.5-pro | Multiple |

---

## Basic Usage

### Single Image

```python
from ai_infra import LLM

llm = LLM()

# From file path
response = llm.chat(
    "Describe this image",
    images=["photo.jpg"]
)

# From URL
response = llm.chat(
    "What's in this image?",
    images=["https://example.com/image.jpg"]
)
```

### Multiple Images

```python
response = llm.chat(
    "Compare these two images",
    images=["image1.jpg", "image2.jpg"]
)
```

---

## Image Sources

### Local Files

```python
# Supports: jpg, jpeg, png, gif, webp
response = llm.chat("Describe", images=["local/path/image.png"])
```

### URLs

```python
response = llm.chat(
    "Describe",
    images=["https://example.com/image.jpg"]
)
```

### Base64 Encoded

```python
import base64

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = llm.chat(
    "Describe",
    images=[f"data:image/jpeg;base64,{b64}"]
)
```

### Bytes

```python
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = llm.chat(
    "Describe",
    images=[image_bytes]
)
```

---

## Provider-Specific

### OpenAI Vision

```python
llm = LLM(provider="openai", model="gpt-4o")

response = llm.chat(
    "What's in this image?",
    images=["photo.jpg"]
)
```

### Anthropic Vision

```python
llm = LLM(provider="anthropic", model="claude-sonnet-4-20250514")

response = llm.chat(
    "Analyze this image",
    images=["photo.jpg"]
)
```

### Google Vision

```python
llm = LLM(provider="google_genai", model="gemini-2.0-flash")

response = llm.chat(
    "Describe what you see",
    images=["photo.jpg"]
)
```

---

## Advanced Usage

### With System Prompt

```python
response = llm.chat(
    "Extract text from this image",
    images=["document.jpg"],
    system="You are an OCR assistant. Extract all visible text."
)
```

### Structured Output

```python
from pydantic import BaseModel

class ImageAnalysis(BaseModel):
    objects: list[str]
    scene: str
    mood: str

response = llm.chat(
    "Analyze this image",
    images=["photo.jpg"],
    response_model=ImageAnalysis
)
print(response.objects)  # ['tree', 'car', 'person']
```

### Multi-Turn with Images

```python
messages = [
    {"role": "user", "content": "What's in this image?", "images": ["img1.jpg"]},
    {"role": "assistant", "content": "This is a photo of a cat."},
    {"role": "user", "content": "What color is the cat?"},
]
response = llm.chat(messages)
```

---

## Image Encoding Utilities

```python
from ai_infra.llm.multimodal.vision import encode_image, ImageSource

# Auto-detect and encode
encoded = encode_image("photo.jpg")

# With specific format
encoded = encode_image("photo.jpg", format="jpeg")

# From bytes
encoded = encode_image(image_bytes, format="png")
```

---

## Error Handling

```python
from ai_infra import LLM
from ai_infra.errors import AIInfraError, ValidationError

try:
    response = llm.chat("Describe", images=["photo.jpg"])
except ValidationError as e:
    print(f"Invalid image: {e}")
except AIInfraError as e:
    print(f"Error: {e}")
```

---

## Best Practices

1. **Resize large images** - Providers have size limits
2. **Use JPEG for photos** - Better compression
3. **Use PNG for screenshots** - Better quality for text
4. **Include context** - Tell the model what to look for
5. **Check provider limits** - Max images per request varies

---

## See Also

- [LLM](../core/llm.md) - Chat completions
- [ImageGen](../imagegen/imagegen.md) - Generate images
- [Providers](../core/providers.md) - Provider capabilities
