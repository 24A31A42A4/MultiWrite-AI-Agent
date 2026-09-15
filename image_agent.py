import os
import time
import random

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from model import image_model, image_model_groups, num_image_keys
from state import ImageState

# Create the image graph.
image_graph = StateGraph(ImageState)


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

    # Strategy: for each model, round-robin through API keys first.
    # This escapes rate-limited keys quickly before falling back to heavier models.
    max_retries_per_key = 2
    base_delay = 10  # seconds

    total_models = len(image_model_groups)

    for model_idx, (model_name, key_variants) in enumerate(image_model_groups):
        print(f"  🎨 Trying model {model_idx + 1}/{total_models}: {model_name}")

        for key_idx, current_model in enumerate(key_variants):
            for attempt in range(max_retries_per_key):
                try:
                    response = current_model.invoke(messages)
                    print(f"  ✅ Success with {model_name} (key {key_idx + 1}/{num_image_keys})")
                    return _save_image_result(response, task)
                except Exception as e:
                    error_str = str(e)
                    # Check for rate limit errors (429) from Gemini.
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimitError" in type(e).__name__:
                        delay = base_delay * (2 ** attempt) + random.uniform(0.1, 2.0)
                        print(
                            f"  ⚠️  Rate limited on {model_name} "
                            f"(key {key_idx + 1}/{num_image_keys}, "
                            f"attempt {attempt + 1}/{max_retries_per_key}). "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Non-rate-limit error: log and try next key/model.
                        print(f"  ❌ Error on {model_name} (key {key_idx + 1}/{num_image_keys}): {error_str[:120]}")
                        break  # Skip remaining retries for this key, try next key.

            # All retries for this key exhausted.
            if key_idx < len(key_variants) - 1:
                print(f"  🔄 Switching to key {key_idx + 2}/{num_image_keys} for {model_name}...")

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

