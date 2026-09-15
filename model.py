import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys from .env file.
load_dotenv()

# Text models (Groq)
groq_api_keys = [
    key for key in [
        os.getenv("groq-apikey"),
        os.getenv("groq-apikey2"),
    ] if key
]

# Text models to try in order.
text_model_names = [
    "groq:qwen/qwen3.8-27b",
    "groq:openai/gpt-oss-120b",
    "groq:meta-llama/llama-prompt-guard-2-22m",
]

# Build a list of all model variants (each model with each key)
text_models = []
for model_name in text_model_names:
    for key in groq_api_keys:
        text_models.append(
            init_chat_model(
                model=model_name,
                temperature=0,
                api_key=key,
            )
        )

# Chain all text models together: tries primary model with key1 -> key2 -> fallback model with key1 -> key2
model = text_models[0].with_fallbacks(text_models[1:]) if text_models else None

# Image models (Gemini) for generating blog illustrations.
# Supports multiple API keys and fallback models to handle quota limits.
gemini_api_keys = [
    key for key in [
        os.getenv("gemini_api_key"),
        os.getenv("gemini_api_key_2"),
        os.getenv("gemini_api_key_3"),
        os.getenv("gemini_api_key_4"),
    ] if key
]

# Image models to try in order: fast/cheap first, high-quality last.
image_model_names = [
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
]

# Build image models grouped by model name.
# Structure: list of (model_name, [model_with_key1, model_with_key2, ...])
# This lets the image agent round-robin keys first, then fall back to the next model.
image_model_groups = []
for model_name in image_model_names:
    key_variants = []
    for key in gemini_api_keys:
        key_variants.append(
            init_chat_model(
                model=model_name,
                model_provider="google_genai",
                temperature=0,
                google_api_key=key,
            )
        )
    if key_variants:
        image_model_groups.append((model_name, key_variants))

# Total number of API keys configured.
num_image_keys = len(gemini_api_keys)

# Primary image model (None if no keys are configured).
image_model = image_model_groups[0][1][0] if image_model_groups else None
