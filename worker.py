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
    audience = state["audience"]
    tone = state["tone"]

    messages = [
        SystemMessage(
            content="""You are a Blog Section Writer Agent for AgentWriter AI.
                     You are a senior technical writer and developer advocate for AgentWriter AI.

                     Write exactly one high-quality blog section from the assigned Task.

                     Writing requirements:
                     - Write only the assigned section; do not write the whole blog or unrelated
                         sections. If the assigned task is the conclusion, write that conclusion.
                     - Make the section directly fulfill the task goal and cover every bullet point
                         without repeating or merging distinct points.
                     - Keep the section close to the requested target word count.
                     - Write a fully developed section, not a summary: use at least three
                         substantive paragraphs for a normal section and at least two for an
                         introduction or conclusion.
                     - Separate distinct ideas into separate paragraphs; do not combine the whole
                         section into one paragraph or a short list of claims.
                     - Assume the stated audience and use the requested tone consistently.
                     - Start with a useful explanation, then use clear subheadings, numbered steps,
                         tables, or bullets when they improve scanning and comprehension.
                     - Explain technical terms before relying on them, and prefer concrete examples,
                         trade-offs, and implementation guidance over generic claims.
                     - Use the research report only for relevant support. Do not invent facts,
                         statistics, product capabilities, citations, or source details.
                     - When research is required, cite claims using the available source URL or
                         source name. When citations are not required, do not add unsupported claims.
                     - Include code only when required. Code must be complete enough to understand,
                         use the correct language fence, and be followed by a concise explanation.
                     - Avoid filler introductions, repetition, marketing language, and references to
                         the writing process or other agents.
                     - Do not shorten the section below the requested target word count merely to
                         be concise; include the concrete explanations needed by the bullets.
                     - Use valid Markdown and make the result ready to publish.

                     Return only the completed section."""
        ),
        HumanMessage(
            content=f"""
TASK

AUDIENCE:
{audience}

TONE:
{tone}

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
        "section_outputs": [{
            "task_id": task.id,
            "title": task.title,
            "content": response.content,
        }]
    }


# Connect the worker graph.
worker_graph.add_node("worker_node", worker_node)

worker_graph.add_edge(START, "worker_node")
worker_graph.add_edge("worker_node", END)