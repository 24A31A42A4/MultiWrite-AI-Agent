import os
import time
import random
import threading
import base64
import mimetypes
from urllib.parse import urlparse

import requests

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from model import image_model, image_model_groups, num_image_keys
from state import ImageState

# Create the image graph.
image_graph = StateGraph(ImageState)
image_request_lock = threading.Lock()
image_key_cursor = {}
image_key_cooldowns = {}
image_key_in_use = set()
image_route_cursor = 0
IMAGE_RATE_LIMIT_COOLDOWN_SECONDS = 30


# Image agent generates a blog illustration for a given section.
def image_node(state: ImageState):
    print("\nRunning image_node...")
    # Stagger parallel execution to avoid thundering herd on the API
    time.sleep(random.uniform(0.5, 3.0))
    
    task = state["task"]
    prompt = state.get("image_prompt") or f"Create a high-quality blog illustration for: {task.title}"

    # Skip if no Hugging Face image models are configured.
    if not image_model_groups:
        print("Finished image_node.")
        return {
            "image_outputs": [{
                "task_id": task.id,
                "image_path": "",
                "image_data": "Hugging Face image model is not configured. Add hf_token to your .env file.",
            }],
        }

    messages = [
        SystemMessage(
            content="""You are the Image Agent for AgentWriter AI.

Create one publication-ready editorial illustration for a technical blog.
Convert the assigned section into a single clear visual story, not a poster,
slide, infographic, dashboard, or document. Choose the strongest visual form
for the subject: a cinematic systems scene, a clean architectural concept,
an expressive technical still life, or a simple visual metaphor.

Use a sophisticated modern editorial style, accurate and believable objects,
clear depth, restrained detail, strong focal hierarchy, balanced negative space,
consistent perspective, and soft professional lighting. Use a wide 16:9
composition suitable for a blog header. Make the image readable at thumbnail
size and avoid visual clutter.

ABSOLUTE NEGATIVE REQUIREMENTS: no words, letters, numbers, symbols, labels,
headings, logos, watermarks, captions, tables, comparison matrices, charts,
graphs, dashboards, UI panels, screenshots, code, or fake typography. Do not
render language-like marks. Never attempt to spell anything. Generate actual
image content, not a written description."""
        ),
        HumanMessage(
            content=f"""Task Title:
{task.title}

Task Goal:
{task.goal}

Prompt idea:
{prompt}

Create the final 16:9 blog illustration for this section.

Section title: {task.title}
Section goal: {task.goal}
Visual direction: {prompt}

First identify the central subject and the relationship the reader should
understand. Then depict that relationship visually using coherent objects,
scale, lighting, color contrast, and composition. If the topic compares ideas,
show contrast through two distinct visual environments or opposing systems,
never through text, tables, or labels. If the topic describes architecture or
workflow, show connected components as physical or abstract forms without
annotation. Keep every important element fully inside the frame.

Return only the generated image. Do not return a prompt, explanation,
description, text, symbols, or a second concept."""
        ),
    ]

    # Rotate globally across every model/key pair so parallel tasks spread load.
    routes = [
        (model_name, key_idx, current_model)
        for model_name, key_variants in image_model_groups
        for key_idx, current_model in enumerate(key_variants)
    ]
    attempted_routes = set()
    global image_route_cursor

    while len(attempted_routes) < len(routes):
        selected_route = None
        with image_request_lock:
            now = time.monotonic()
            for offset in range(len(routes)):
                route_idx = (image_route_cursor + offset) % len(routes)
                if route_idx in attempted_routes:
                    continue
                model_name, key_idx, current_model = routes[route_idx]
                if image_key_cooldowns.get((model_name, key_idx), 0) > now:
                    continue
                if (model_name, key_idx) in image_key_in_use:
                    continue
                image_route_cursor = (route_idx + 1) % len(routes)
                image_key_in_use.add((model_name, key_idx))
                selected_route = (route_idx, model_name, key_idx, current_model)
                break

        if selected_route is None:
            break

        route_idx, model_name, key_idx, current_model = selected_route
        attempted_routes.add(route_idx)
        print(
            f"  🎨 Trying {model_name} with rotated key "
            f"{key_idx + 1}/{num_image_keys}"
        )
        try:
            response = current_model.invoke(messages)
            print(f"  ✅ Success with {model_name} (key {key_idx + 1}/{num_image_keys})")
            return _save_image_result(response, task)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimitError" in type(e).__name__:
                with image_request_lock:
                    image_key_cooldowns[(model_name, key_idx)] = (
                        time.monotonic() + IMAGE_RATE_LIMIT_COOLDOWN_SECONDS
                    )
                print(
                    f"  ⚠️  Rate limited on {model_name} "
                    f"(key {key_idx + 1}/{num_image_keys}). "
                    f"Switching to the next API key..."
                )
            else:
                print(
                    f"  ❌ Error on {model_name} "
                    f"(key {key_idx + 1}/{num_image_keys}): {error_str[:120]}"
                )
        finally:
            with image_request_lock:
                image_key_in_use.discard((model_name, key_idx))

    # All models and keys exhausted, skip gracefully.
    print("  ⚠️  Image generation skipped (all models and API keys exhausted).")
    print("Finished image_node.")
    return {
        "image_outputs": [{
            "task_id": task.id,
            "image_path": "",
            "image_data": "Image generation skipped due to API quota limits.",
        }],
    }


# Helper to save the image result to a file and return state.
def _save_image_result(response, task):
    image_result = response.content if hasattr(response, "content") else response
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    image_bytes, extension, image_url = _extract_image(image_result)
    if image_url:
        try:
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            image_bytes = image_response.content
            extension = _extension_for_mime(
                image_response.headers.get("content-type")
            )
        except requests.RequestException as error:
            print(f"Could not download generated image URL: {error}")
            image_bytes = None

    if image_bytes:
        file_name = f"{task.id or 'section'}_image.{extension}"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        image_data = "Generated image asset"
        relative_path = os.path.relpath(
            file_path, os.path.dirname(os.path.abspath(__file__))
        ).replace(os.sep, "/")
    else:
        description = _content_as_text(image_result)
        file_name = f"{task.id or 'section'}_image.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(description)
        image_data = description
        relative_path = ""

    print("Finished image_node.")
    return {
        "image_outputs": [{
            "task_id": task.id,
            "image_path": relative_path,
            "image_data": image_data,
        }],
    }


def _content_as_text(content) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _extract_image(content):
    """Extract bytes or a downloadable URL from common image response formats."""
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        if isinstance(block, bytes):
            return block, "png", None

        if isinstance(block, dict):
            get_value = block.get
        else:
            get_value = lambda key, default=None: getattr(block, key, default)

        mime_type = (
            get_value("mime_type")
            or get_value("mimeType")
            or get_value("media_type")
        )
        data = (
            get_value("data")
            or get_value("image")
            or get_value("base64")
            or get_value("base64_data")
            or get_value("base64Data")
        )
        inline_data = get_value("inline_data") or get_value("inlineData")
        if inline_data:
            if isinstance(inline_data, dict):
                data = inline_data.get("data") or data
                mime_type = (
                    inline_data.get("mime_type")
                    or inline_data.get("mimeType")
                    or mime_type
                )
            else:
                data = getattr(inline_data, "data", None) or data
                mime_type = (
                    getattr(inline_data, "mime_type", None)
                    or getattr(inline_data, "mimeType", None)
                    or mime_type
                )

        image_url = get_value("image_url") or get_value("imageUrl")
        if isinstance(image_url, dict):
            data = image_url.get("url") or data
        elif isinstance(image_url, str):
            data = image_url

        if get_value("type") in {"image", "image_data", "image_url"}:
            data = data or get_value("url")
        data = data or get_value("url")

        if isinstance(data, bytes):
            return data, _extension_for_mime(mime_type), None
        if isinstance(data, str):
            if data.startswith(("http://", "https://")):
                return None, _extension_for_mime(mime_type), data
            if data.startswith("data:") and "," in data:
                header, encoded = data.split(",", 1)
                try:
                    return base64.b64decode(encoded), _extension_for_mime(header), None
                except (ValueError, base64.binascii.Error):
                    continue
            try:
                return (
                    base64.b64decode(data, validate=True),
                    _extension_for_mime(mime_type),
                    None,
                )
            except (ValueError, base64.binascii.Error):
                continue
    return None, None, None


def _extension_for_mime(mime_type) -> str:
    if mime_type and "jpeg" in mime_type:
        return "jpg"
    if mime_type and "webp" in mime_type:
        return "webp"
    return "png"


# Connect the image graph.
image_graph.add_node("image_agent", image_node)
image_graph.add_edge(START, "image_agent")
image_graph.add_edge("image_agent", END)

