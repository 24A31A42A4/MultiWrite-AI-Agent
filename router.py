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

Your job is to analyze the user's blog topic and decide whether external research is required before generating the blog.

If the topic can be answered reliably using general knowledge and does not require current, factual, or source-backed information, set research_required to false and return an empty list of queries.

If the topic requires factual verification, current information, specific statistics, recent events, technical details, historical facts, or source-backed information, set research_required to true.

When research is required, generate 7 to 10 focused and useful search queries that will help the Research Agent collect high-quality information for the blog.

The queries should:
- Cover different important aspects of the topic.
- Avoid unnecessary or duplicate searches.
- Be specific and useful for research.
- Focus on information that will help create a high-quality blog.
- Prefer authoritative and reliable information sources.

Do not perform the research yourself.
Do not write the blog.
Only make the routing decision and generate research queries.

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
