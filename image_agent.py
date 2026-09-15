import os
import time
import random
import threading

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


# Image agent generates a blog illustration for a given section.
def image_node(state: ImageState):
    print("\nRunning image_node...")
    # Stagger parallel execution to avoid thundering herd on the API
    time.sleep(random.uniform(0.5, 3.0))
    
    task = state["task"]
    prompt = state.get("image_prompt") or f"Create a high-quality blog illustration for: {task.title}"

    # Skip if no Gemini image models are configured.
    if not image_model_groups:
        print("Finished image_node.")
        return {
            "image_path": "",
            "image_data": "Gemini image model is not configured. Add gemini_api_key to your .env file.",
        }

    messages = [
        SystemMessage(
            content="You are the Image Agent for AgentWriter AI. Create a high-quality image prompt and generate a blog illustration concept for the section."
        ),
        HumanMessage(
            content=f"""Task Title:
{task.title}

Task Goal:
{task.goal}

Prompt idea:
{prompt}

Create a polished, vivid image concept and a short visual description suitable for blog generation."""
        ),
    ]

    # Rotate keys globally so parallel image nodes do not all begin with key 1.
    max_retries_per_key = 1
    base_delay = 10

    total_models = len(image_model_groups)

    for model_idx, (model_name, key_variants) in enumerate(image_model_groups):
        print(f"  🎨 Trying model {model_idx + 1}/{total_models}: {model_name}")

        attempted_keys = set()
        while len(attempted_keys) < len(key_variants):
            with image_request_lock:
                now = time.monotonic()
                start_idx = image_key_cursor.get(model_name, 0)
                available_keys = [
                    key_idx
                    for offset in range(len(key_variants))
                    for key_idx in [(start_idx + offset) % len(key_variants)]
                    if key_idx not in attempted_keys
                    and (model_name, key_idx) not in image_key_in_use
                    and image_key_cooldowns.get((model_name, key_idx), 0) <= now
                ]
                if available_keys:
                    key_idx = available_keys[0]
                    image_key_cursor[model_name] = (key_idx + 1) % len(key_variants)
                    image_key_in_use.add((model_name, key_idx))
                    wait_until = None
                else:
                    remaining_keys = [
                        key_idx
                        for key_idx in range(len(key_variants))
                        if key_idx not in attempted_keys
                    ]
                    wait_until = min(
                        image_key_cooldowns.get((model_name, key_idx), now)
                        for key_idx in remaining_keys
                    )

            if wait_until is not None:
                time.sleep(max(0.05, wait_until - time.monotonic()))
                continue

            attempted_keys.add(key_idx)
            current_model = key_variants[key_idx]
            try:
                response = current_model.invoke(messages)
                print(f"  ✅ Success with {model_name} (key {key_idx + 1}/{num_image_keys})")
                return _save_image_result(response, task)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimitError" in type(e).__name__:
                    delay = base_delay + random.uniform(0.1, 2.0)
                    with image_request_lock:
                        image_key_cooldowns[(model_name, key_idx)] = time.monotonic() + delay
                    print(
                        f"  ⚠️  Rate limited on {model_name} "
                        f"(key {key_idx + 1}/{num_image_keys}). "
                        f"Skipping it for {delay:.2f}s..."
                    )
                else:
                    print(f"  ❌ Error on {model_name} (key {key_idx + 1}/{num_image_keys}): {error_str[:120]}")
            finally:
                with image_request_lock:
                    image_key_in_use.discard((model_name, key_idx))

            if len(attempted_keys) < len(key_variants):
                print(f"  🔄 Switching to another key for {model_name}...")

        # All keys exhausted for this model.
        if model_idx < total_models - 1:
            print(f"  🔄 Falling back to next model...")

    # All models and keys exhausted, skip gracefully.
    print("  ⚠️  Image generation skipped (all models and API keys exhausted).")
    print("Finished image_node.")
    return {
        "image_path": "",
        "image_data": "Image generation skipped due to API quota limits.",
    }


# Helper to save the image result to a file and return state.
def _save_image_result(response, task):
    image_result = response.content if hasattr(response, "content") else str(response)

    # Save the generated image description to a file.
    output_dir = os.path.join(os.getcwd(), "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{task.id or 'section'}_image.txt"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(image_result)

    print("Finished image_node.")
    return {
        "image_path": file_path,
        "image_data": image_result,
    }


# Connect the image graph.
image_graph.add_node("image_node", image_node)
image_graph.add_edge(START, "image_node")
image_graph.add_edge("image_node", END)

