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


# Reducer agent combines all written sections into one final blog.
def reducer_node(state: ReducerState):
    print("\nRunning reducer_node...")
    plan = state["plan"]
    section_outputs = state["section_outputs"]
    print(
        f"Reducer received {len(section_outputs)}/{len(plan.tasks)} worker sections "
        "without image outputs."
    )
    sections_by_id = {
        section.get("task_id"): section
        for section in section_outputs
        if section.get("task_id")
    }
    markdown_parts = [f"# {plan.blog_title}"]
    for task in plan.tasks:
        section = sections_by_id.get(task.id)
        if not section:
            raise ValueError(f"Missing worker output for task {task.id}: {task.title}")

        markdown_parts.append(f"## {task.title}")
        markdown_parts.append(
            _remove_duplicate_section_heading(section["content"], task.title)
        )

    final_blog = "\n\n".join(markdown_parts).strip() + "\n"
    print("Finished reducer_node.")
    return {
        "final_blog": final_blog
    }


# Connect the reducer graph.
reducer_graph.add_node("reducer_node", reducer_node)

reducer_graph.add_edge(START, "reducer_node")

reducer_graph.add_edge("reducer_node", END)