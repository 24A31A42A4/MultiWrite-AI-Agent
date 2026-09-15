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

Your job is to turn the user's topic and research report into a structured blog plan.

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
- requires_image (set to true for sections that would benefit from a visual illustration)

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
    print("Finished planner_agent.")
    return {"plan": response}


# Connect the orchestrator graph.
Orchestrator_graph.add_node("planner_agent", planner_agent)
Orchestrator_graph.add_edge(START, "planner_agent")
