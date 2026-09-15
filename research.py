from langgraph.graph import START, StateGraph
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

import os
from tavily import TavilyClient

from model import model
from state import ResearchState


# Load environment variables.

load_dotenv()

tavily_api_key = os.getenv("tavily_search_api_key")

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# Create the research graph.

Research_graph = StateGraph(ResearchState)


def _compact_search_results(results: list[dict]) -> str:
    excerpts = []
    for result_group in results:
        for result in result_group.get("results", []):
            excerpts.append(
                "Title: {title}\nURL: {url}\nContent: {content}".format(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", "")[:1200],
                )
            )

    return "\n\n".join(excerpts)[:24000]


# Research agent searches for information.

def Research_node(state: ResearchState):
    print("\nRunning Research_node...")
    queries = state["queries"]

    results = []

    # Search Tavily for every query
    for query in queries:

        response = tavily_client.search(
            query,
            max_results=3
        )

        results.append(response)


    # Summarize the research results.

    compact_results = _compact_search_results(results)
    research_prompt = f"""
You are the Research Agent for AgentWriter AI.

Analyze the search results below and create a concise,
high-quality research report for the Orchestrator Agent.

Include:

- Important facts
- Key concepts
- Important statistics
- Relevant findings
- Useful technical details
- Source names
- Source URLs

Rules:

- Remove duplicate information.
- Remove irrelevant information.
- Do not include raw JSON structure.
- Do not invent facts.
- Keep the report concise but useful.
- Preserve important source information.
- Organize the information clearly.

SEARCH RESULTS:

{compact_results}
"""


    messages = [

        SystemMessage(
            content="""
You are a professional research summarization agent.

Your job is to convert raw web search results
into a concise and factual research report.
"""
        ),

        HumanMessage(
            content=research_prompt
        )

    ]


    # Ask LLM to summarize the research
    response = model.invoke(messages)


    # Return the research report.

    print("Finished Research_node.")
    return {
        "research_report": response.content
    }


# Connect the research graph.

Research_graph.add_node(
    "Research_node",
    Research_node
)

Research_graph.add_edge(
    START,
    "Research_node"
)