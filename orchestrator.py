from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph

from model import model
from state import OrchestratorState, Plan

# Create the orchestrator graph.
Orchestrator_graph = StateGraph(OrchestratorState)

# Ensure the LLM outputs a structured Plan object.
structured_llm = model.with_structured_output(Plan)


# Planner agent turns the topic and research into a structured blog plan.
def planner_agent(state: OrchestratorState):
    print("\nRunning planner_agent...")
    user_topic = state["user_topic"]
    research_report = state["research_report"]

    messages = [
        SystemMessage(
            content="""You are the Orchestrator Agent for AgentWriter AI.

You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Create a complete Plan object with:
- blog_title
- audience
- tone
- blog_kind
- constraints
- tasks

The tasks should be a list of Task objects and each task should include:
- id
- title
- goal
- bullets
- target_words
- tags
- requires_research
- requires_citations
- requires_code
- requires_image (always set to false)

Hard requirements:
- Create 5 to 9 tasks suitable for the topic and audience.
- Include an introduction task first and a conclusion task last; use the
    remaining tasks for the main topic sections.
- Do not plan image generation or image assets for any section.
- The goal must be exactly one sentence.
- Each task must contain 3 to 6 concrete, specific, non-overlapping bullets.
- Set target_words between 120 and 550 for every task.

Use the topic and research report as the main inputs.
Return a clear and practical blog plan."""
        ),
        HumanMessage(
            content=f"""User Topic:
{user_topic}

Research Report:
{research_report}"""
        ),
    ]

    # Ask the LLM to generate the blog plan.
    response = structured_llm.invoke(messages)
    response = response.model_copy(
        update={
            "tasks": [
                task.model_copy(update={"requires_image": False})
                for task in response.tasks
            ]
        }
    )
    print("Finished planner_agent.")
    return {"plan": response}


# Connect the orchestrator graph.
Orchestrator_graph.add_node("planner_agent", planner_agent)
Orchestrator_graph.add_edge(START, "planner_agent")
