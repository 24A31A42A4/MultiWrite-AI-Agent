from langgraph.graph import START, StateGraph
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from model import model
from state import RouterState

# Create the router graph.
router_graph = StateGraph(RouterState)


# Define the structured output format for the router response.
class StructuredResponse(BaseModel):
    research_required: bool = Field(description="Whether external research is required")
    queries: list[str] = Field(description="A list of research queries")


# Ensure the LLM outputs the required structure.
structured_llm = model.with_structured_output(StructuredResponse)


# Router agent decides if research is needed and generates search queries.
def router_agent(state: RouterState):
    print("\nRunning router_agent...")
    topic = state["user_topic"]

    messages = [
        SystemMessage(
            content="""You are the Router Agent for AgentWriter AI, a multi-agent blog generation system.

Decide whether web research is needed before planning the blog.

For evergreen topics where correctness does not depend on recent facts, such as concepts and fundamentals, set research_required to false.
For mostly evergreen topics that need up-to-date examples, tools, or models, set research_required to true.
For volatile topics, including weekly roundups, "this week", "latest", rankings, pricing, or policy and regulation, set research_required to true.
For topics comparing AI models, providers, benchmarks, capabilities, or current
technology choices, always set research_required to true because these facts
change over time.

When research_required is false, return an empty list of queries.

When research_required is true, return 3 to 10 high-signal queries. Make each query scoped and specific to the user's topic; do not use generic queries such as just "AI" or "LLM". Include current-year and latest-information wording for time-sensitive technology topics. If the user asks for "last week", "this week", or "latest", include that time constraint in the queries.

Do not perform the research yourself or write the blog. Only make the routing decision and generate research queries.

Return the result using the required structured output format."""
        ),
        HumanMessage(content=topic),
    ]

    # Ask the LLM to analyze the topic and return routing decision.
    response = structured_llm.invoke(messages)

    print("Finished router_agent.")
    return {
        "user_topic": topic,
        "research_required": response.research_required,
        "queries": response.queries,
    }


# Connect the router graph.
router_graph.add_node("router_agent", router_agent)
router_graph.add_edge(START, "router_agent")
