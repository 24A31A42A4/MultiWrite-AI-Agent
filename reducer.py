from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from model import model
from state import ReducerState

# Create the reducer graph.
reducer_graph = StateGraph(ReducerState)


# Reducer agent combines all written sections into one final blog.
def reducer_node(state: ReducerState):
    print("\nRunning reducer_node...")
    plan = state["plan"]
    section_outputs = state["section_outputs"]

    messages = [
        SystemMessage(
            content="""You are the Reducer Agent for AgentWriter AI.

Your job is to combine multiple independently written blog sections
into one complete, polished blog.

Requirements:
- Follow the order of tasks in the Plan.
- Combine all section outputs.
- Do not remove important information.
- Remove unnecessary repetition between sections.
- Maintain a consistent tone and writing style.
- Make transitions between sections natural.
- Preserve useful code examples and citations.
- Do not invent new facts.
- Keep the structure of the original Plan.
- Produce a clean, publication-ready Markdown blog.

Return only the final blog."""
        ),

        HumanMessage(
            content=f"""BLOG PLAN:

{plan}

WORKER SECTION OUTPUTS:

{section_outputs}

Combine these sections into the final blog."""
        ),
    ]

    # Ask the LLM to merge all sections into the final blog.
    response = model.invoke(messages)

    print("Finished reducer_node.")
    return {
        "final_blog": response.content
    }


# Connect the reducer graph.
reducer_graph.add_node("reducer_node", reducer_node)

reducer_graph.add_edge(START, "reducer_node")

reducer_graph.add_edge("reducer_node", END)