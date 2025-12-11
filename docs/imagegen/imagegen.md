# Image Generation

> Generate images with DALL-E, Imagen, Stability AI, and Replicate.

## Quick Start

```python
from ai_infra import ImageGen

gen = ImageGen()  # Auto-detects provider from env
image = gen.generate("A serene mountain landscape at sunset")
image.save("landscape.png")
```

---

## Overview

ai-infra provides a unified interface for image generation across multiple providers:
- **OpenAI DALL-E** - DALL-E 2, DALL-E 3
- **Google Imagen** - Imagen 2, Imagen 3
- **Stability AI** - Stable Diffusion, SDXL
- **Replicate** - Various open-source models

---

## Basic Usage

### Simple Generation

```python
from ai_infra import ImageGen

gen = ImageGen()
image = gen.generate("A futuristic city with flying cars")
image.save("city.png")
```

### With Provider Selection

```python
from ai_infra import ImageGen

# OpenAI DALL-E
gen = ImageGen(provider="openai")
image = gen.generate("A cute robot assistant")

# Stability AI
gen = ImageGen(provider="stability")
image = gen.generate("A photorealistic portrait")

# Google Imagen
gen = ImageGen(provider="google_genai")
image = gen.generate("An abstract painting")
```

### With Parameters

```python
gen = ImageGen(provider="openai")
image = gen.generate(
    "A cute robot assistant",
    model="dall-e-3",
    size="1792x1024",  # Widescreen
    quality="hd",
    style="vivid",
)
```

---

## Async Generation

```python
from ai_infra import ImageGen

gen = ImageGen(provider="openai")
image = await gen.agenerate("A sunset over mountains")
image.save("sunset.png")
```

---

## Provider Configuration

### OpenAI DALL-E

```python
gen = ImageGen(provider="openai")
image = gen.generate(
    "Your prompt...",
    model="dall-e-3",  # or "dall-e-2"
    size="1024x1024",  # 1024x1024, 1792x1024, 1024x1792
    quality="standard",  # or "hd"
    style="vivid",  # or "natural"
)
```

### Stability AI

```python
gen = ImageGen(provider="stability")
image = gen.generate(
    "Your prompt...",
    model="stable-diffusion-xl",
    size="1024x1024",
    steps=30,
    cfg_scale=7.0,
    seed=12345,  # For reproducibility
)
```

### Google Imagen

```python
gen = ImageGen(provider="google_genai")
image = gen.generate(
    "Your prompt...",
    model="imagen-3",
    size="1024x1024",
    aspect_ratio="1:1",  # or "16:9", "9:16", etc.
)
```

### Replicate

```python
gen = ImageGen(provider="replicate")
image = gen.generate(
    "Your prompt...",
    model="stability-ai/sdxl",
    size="1024x1024",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

---

## Multiple Images

Generate multiple variations:

```python
gen = ImageGen(provider="openai")
images = gen.generate(
    "A magical forest",
    n=4,  # Generate 4 images
)

for i, image in enumerate(images):
    image.save(f"forest_{i}.png")
```

---

## Image Editing

### Inpainting (OpenAI)

```python
gen = ImageGen(provider="openai")
image = gen.edit(
    image="input.png",
    mask="mask.png",  # White areas = edit region
    prompt="Add a rainbow in the sky",
)
```

### Image Variations

```python
gen = ImageGen(provider="openai")
images = gen.create_variation(
    image="input.png",
    n=3,  # Generate 3 variations
)
```

---

## Advanced Options

### Negative Prompts

```python
gen = ImageGen(provider="stability")
image = gen.generate(
    "A beautiful portrait",
    negative_prompt="blurry, low quality, distorted",
)
```

### Seed for Reproducibility

```python
gen = ImageGen(provider="stability")
image = gen.generate(
    "A red apple",
    seed=42,  # Same seed = same image
)
```

### Style Transfer

```python
gen = ImageGen(provider="stability")
image = gen.generate(
    "A landscape",
    style="anime",  # Provider-specific styles
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
