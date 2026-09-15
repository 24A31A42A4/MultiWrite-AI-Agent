import time
import random

from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from model import model
from state import WorkerState

# Create the worker graph.
worker_graph = StateGraph(WorkerState)


# Worker agent writes one blog section based on its assigned task.
def worker_node(state: WorkerState):
    print("\nRunning worker_node...")
    # Stagger parallel execution to avoid thundering herd on the API
    time.sleep(random.uniform(0.5, 3.0))

    task = state["task"]
    research_report = state["research_report"]

    messages = [
        SystemMessage(
            content="""You are a Blog Section Writer Agent for AgentWriter AI.

Write ONE blog section based on the given task.

Follow all task requirements carefully.

Requirements:
- Write only the assigned section.
- Do not write the entire blog.
- Follow the task title and goal.
- Cover all important bullet points.
- Stay close to the target word count.
- Use the required tone and style.
- Use the research report when research is required.
- Do not invent facts.
- Include citations when required.
- Include code only when required.
- Avoid repetition.
- Make the section clear, useful, and well structured.
- Use Markdown formatting where appropriate.

Return only the completed section."""
        ),
        HumanMessage(
            content=f"""
TASK

ID:
{task.id}

TITLE:
{task.title}

GOAL:
{task.goal}

BULLET POINTS:
{task.bullets}

TARGET WORDS:
{task.target_words}

TAGS:
{task.tags}

REQUIRES RESEARCH:
{task.requires_research}

REQUIRES CITATIONS:
{task.requires_citations}

REQUIRES CODE:
{task.requires_code}


RESEARCH REPORT:
{research_report}
"""
        ),
    ]

    # Retry with exponential backoff if Groq rate limits the request.
    max_retries = 10
    base_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)
            break
        except Exception as e:
            error_str = str(e)
            # Check for rate limit errors (429) from any provider.
            if "429" in error_str or "rate_limit" in error_str.lower() or "RateLimitError" in type(e).__name__:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 2.0)
                print(f"  ⚠️  Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.2f}s...")
                time.sleep(delay)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    print("Finished worker_node.")
    return {
        "section_outputs": [response.content]
    }


# Connect the worker graph.
worker_graph.add_node("worker_node", worker_node)

worker_graph.add_edge(START, "worker_node")
worker_graph.add_edge("worker_node", END)