# Image Generation

> Generate images with DALL-E, Imagen, Stability AI, and Replicate.

## Quick Start

```python
from ai_infra import generate_image

result = await generate_image(
    prompt="A serene mountain landscape at sunset",
    provider="openai",
    size="1024x1024",
)

# Save the image
result.save("landscape.png")
```

---

## Overview

ai-infra provides a unified interface for image generation across multiple providers:
- **OpenAI DALL-E** - DALL-E 2, DALL-E 3
- **Google Imagen** - Imagen 2, Imagen 3
- **Stability AI** - Stable Diffusion, SDXL
- **Replicate** - Various open-source models

---

## Basic Generation

### Simple Prompt

```python
from ai_infra import generate_image

result = await generate_image(
    prompt="A futuristic city with flying cars",
    provider="openai",
)

# Access generated image
image_bytes = result.data
result.save("city.png")
```

### With Parameters

```python
result = await generate_image(
    prompt="A cute robot assistant",
    provider="openai",
    model_name="dall-e-3",
    size="1792x1024",  # Widescreen
    quality="hd",
    style="vivid",
)
```

---

## Provider Configuration

### OpenAI DALL-E

```python
result = await generate_image(
    prompt="...",
    provider="openai",
    model_name="dall-e-3",  # or "dall-e-2"
    size="1024x1024",  # 1024x1024, 1792x1024, 1024x1792
    quality="standard",  # or "hd"
    style="vivid",  # or "natural"
)
```

### Stability AI

```python
result = await generate_image(
    prompt="...",
    provider="stability",
    model_name="stable-diffusion-xl",
    size="1024x1024",
    steps=30,
    cfg_scale=7.0,
    seed=12345,  # For reproducibility
)
```

### Google Imagen

```python
result = await generate_image(
    prompt="...",
    provider="google_genai",
    model_name="imagen-3",
    size="1024x1024",
    aspect_ratio="1:1",  # or "16:9", "9:16", etc.
)
```

### Replicate

```python
result = await generate_image(
    prompt="...",
    provider="replicate",
    model_name="stability-ai/sdxl",
    size="1024x1024",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

---

## Multiple Images

Generate multiple variations:

```python
results = await generate_image(
    prompt="A magical forest",
    provider="openai",
    n=4,  # Generate 4 images
)

for i, result in enumerate(results):
    result.save(f"forest_{i}.png")
```

---

## Image Editing

### Inpainting (OpenAI)

```python
from ai_infra import edit_image

result = await edit_image(
    image="input.png",
    mask="mask.png",  # White areas = edit region
    prompt="Add a rainbow in the sky",
    provider="openai",
)
```

### Image Variations

```python
from ai_infra import create_variation

result = await create_variation(
    image="input.png",
    provider="openai",
    n=3,  # Generate 3 variations
)
```

---

## Advanced Options

### Negative Prompts

```python
result = await generate_image(
    prompt="A beautiful portrait",
    negative_prompt="blurry, low quality, distorted",
    provider="stability",
)
```

### Seed for Reproducibility

```python
result = await generate_image(
    prompt="A red apple",
    provider="stability",
    seed=42,  # Same seed = same image
)
```

### Style Transfer

```python
result = await generate_image(
    prompt="A landscape",
    style="anime",  # Provider-specific styles
    provider="stability",
)
```

---

## Working with Results

### ImageResult Object

```python
result = await generate_image(prompt="...")

# Image data
image_bytes = result.data
image_base64 = result.base64

# Metadata
print(f"Provider: {result.provider}")
print(f"Model: {result.model}")
print(f"Size: {result.size}")
print(f"Seed: {result.seed}")  # If available

# Save to file
result.save("output.png")
result.save("output.jpg", format="jpeg", quality=90)

# Convert to PIL Image (if pillow installed)
pil_image = result.to_pil()
pil_image.show()
```

---

## Streaming Generation

For providers that support progressive generation:

```python
async for progress in generate_image_stream(
    prompt="...",
    provider="replicate",
):
    print(f"Progress: {progress.percentage}%")

    if progress.preview:
        progress.preview.save("preview.png")

    if progress.is_complete:
        progress.result.save("final.png")
```

---

## Provider Discovery

### List Available Providers

```python
from ai_infra import list_imagegen_providers

providers = list_imagegen_providers()
for provider in providers:
    print(f"{provider.name}: {provider.models}")
```

### Check Provider Configuration

```python
from ai_infra import is_provider_configured

if is_provider_configured("openai", "IMAGEGEN"):
    result = await generate_image(prompt="...", provider="openai")
else:
    print("OpenAI not configured for image generation")
```

---

## Error Handling

```python
from ai_infra import (
    generate_image,
    ImageGenError,
    ContentFilterError,
    RateLimitError,
)

try:
    result = await generate_image(prompt="...")
except ContentFilterError:
    print("Prompt was rejected by content filter")
except RateLimitError:
    print("Rate limit exceeded, try again later")
except ImageGenError as e:
    print(f"Generation failed: {e}")
```

---

## Cost Tracking

```python
result = await generate_image(prompt="...")

if result.cost:
    print(f"Cost: ${result.cost:.4f}")
    print(f"Model: {result.model}")
```

---

## Example: Product Images

```python
async def generate_product_images(
    product_name: str,
    styles: list[str],
) -> list:
    results = []

    for style in styles:
        result = await generate_image(
            prompt=f"Professional product photo of {product_name}, {style} style, white background, studio lighting",
            provider="openai",
            model_name="dall-e-3",
            size="1024x1024",
            quality="hd",
        )
        results.append(result)

    return results

# Generate in multiple styles
images = await generate_product_images(
    "wireless headphones",
    styles=["minimalist", "lifestyle", "technical"],
)
```

---

## Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Stability AI
STABILITY_API_KEY=sk-...

# Google
GOOGLE_API_KEY=...

# Replicate
REPLICATE_API_TOKEN=r8_...
```

---

## See Also

- [Providers](../core/providers.md) - Provider configuration
- [Vision](../multimodal/vision.md) - Image understanding
- [Validation](../infrastructure/validation.md) - Input validation
