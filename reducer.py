from langgraph.graph import StateGraph, START, END

from state import ReducerState
import re

# Create the reducer graph.
reducer_graph = StateGraph(ReducerState)


def _remove_duplicate_section_heading(content: str, title: str) -> str:
    """Remove a leading worker heading because the reducer owns section headings."""
    lines = content.strip().splitlines()
    if lines and re.match(r"^#{1,6}\s+", lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


def _image_markdown(image_output: dict, title: str) -> str:
    """Build the Markdown image block for a successfully saved image."""
    image_path = str(image_output.get("image_path") or "").strip()
    if not image_path:
        return ""
    return f"![{title}]({image_path})"


# Reducer agent combines all written sections into one final blog.
def reducer_node(state: ReducerState):
    print("\nRunning reducer_node...")
    plan = state["plan"]
    section_outputs = state["section_outputs"]
    image_outputs = state.get("image_outputs", [])
    print(
        f"Reducer received {len(section_outputs)}/{len(plan.tasks)} worker sections "
        f"and {len(image_outputs)} image outputs."
    )
    sections_by_id = {
        section.get("task_id"): section
        for section in section_outputs
        if section.get("task_id")
    }
    images_by_id = {
        image.get("task_id"): image
        for image in image_outputs
        if image.get("task_id")
    }

    markdown_parts = [f"# {plan.blog_title}"]
    merged_image_count = 0
    for task in plan.tasks:
        section = sections_by_id.get(task.id)
        if not section:
            raise ValueError(f"Missing worker output for task {task.id}: {task.title}")

        markdown_parts.append(f"## {task.title}")
        image_markdown = _image_markdown(images_by_id.get(task.id, {}), task.title)
        if image_markdown:
            markdown_parts.append(image_markdown)
            merged_image_count += 1
        markdown_parts.append(
            _remove_duplicate_section_heading(section["content"], task.title)
        )

    final_blog = "\n\n".join(markdown_parts).strip() + "\n"
    print(f"Merged {merged_image_count} image(s) into final Markdown.")
    print("Finished reducer_node.")
    return {
        "final_blog": final_blog
    }


# Connect the reducer graph.
reducer_graph.add_node("reducer_node", reducer_node)

reducer_graph.add_edge(START, "reducer_node")

reducer_graph.add_edge("reducer_node", END)