import os
from io import BytesIO
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from huggingface_hub import InferenceClient

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

# Hugging Face image-generation routes.
hf_api_keys = [
    key
    for key in [
        os.getenv("hf_token")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        *(
            os.getenv(f"hf_token_{index}")
            or os.getenv(f"HF_TOKEN_{index}")
            or os.getenv(f"HUGGINGFACEHUB_API_TOKEN_{index}")
            for index in range(2, 61)
        ),
    ]
    if key
]

image_model_names = [
    os.getenv("HF_IMAGE_MODEL", "Qwen/Qwen-Image")
]


class HuggingFaceImageModel:
    """Adapt Hugging Face text-to-image inference to LangChain's invoke API."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = InferenceClient(api_key=api_key)

    def invoke(self, messages):
        prompt = messages[-1].content if messages else "Generate a blog illustration."
        image = self.client.text_to_image(prompt=prompt, model=self.model_name)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return SimpleNamespace(content=buffer.getvalue())

# Build image models grouped by model name.
# Structure: list of (model_name, [model_with_key1, model_with_key2, ...])
# This lets the image agent round-robin keys first, then fall back to the next model.
image_model_groups = []
for model_name in image_model_names:
    key_variants = []
    for key in hf_api_keys:
        key_variants.append(HuggingFaceImageModel(model_name, key))
    if key_variants:
        image_model_groups.append((model_name, key_variants))

# Total number of API keys configured.
num_image_keys = len(hf_api_keys)

# Primary image model (None if no keys are configured).
image_model = image_model_groups[0][1][0] if image_model_groups else None
