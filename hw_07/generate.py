import base64
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found")

OUTPUT_DIR = Path(__file__).parent
MODEL = "google/gemini-3.1-flash-image"


def generate_image(prompt: str, filename: str) -> str:
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Generate an image: {prompt}",
                },
            ],
            "max_tokens": 2000,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    images = data["choices"][0]["message"].get("images", [])
    if not images:
        text = data["choices"][0]["message"].get("content", "")
        raise ValueError(f"No images returned. Response: {text[:200]}")

    image_url = images[0]["image_url"]["url"]
    _, base64_data = image_url.split(",", 1)
    image_bytes = base64.b64decode(base64_data)

    filepath = OUTPUT_DIR / filename
    filepath.write_bytes(image_bytes)
    return str(filepath)


if __name__ == "__main__":
    prompts = [
        (
            "a cute orange cat sitting on a wooden chair, simple cartoon style, "
            "flat colors, white background",
            "image_1_simple.png",
        ),
        (
            "a futuristic cyberpunk city street at night, neon signs in Japanese, "
            "rain reflecting on wet pavement, flying cars in the sky, cinematic "
            "lighting, highly detailed, digital art",
            "image_2_medium.png",
        ),
        (
            "ethereal fantasy landscape with floating crystal islands in a pastel "
            "sunset sky, waterfalls cascading into clouds, bioluminescent trees, "
            "intricate details, hyperrealistic, magical atmosphere, artstation "
            "style, unreal engine 5",
            "image_3_advanced.png",
        ),
    ]

    for prompt, filename in prompts:
        print(f"Generating: {filename}")
        print(f"  Prompt: {prompt[:80]}...")
        try:
            path = generate_image(prompt, filename)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

# uv run python hw_07/generate.py
