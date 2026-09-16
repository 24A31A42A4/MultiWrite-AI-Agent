from langgraph.graph import START, StateGraph
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import json
import os
from datetime import date
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


class EvidenceItem(BaseModel):
    title: str = Field(description="The source title")
    url: str = Field(description="The source URL")
    snippet: str = Field(description="A short factual evidence snippet")
    published_at: str | None = Field(
        default=None,
        description="Publication date as YYYY-MM-DD, or null when unavailable or unclear",
    )


class EvidenceResponse(BaseModel):
    evidence: list[EvidenceItem]


structured_llm = model.with_structured_output(EvidenceResponse)


def _compact_search_results(results: list[dict]) -> str:
    excerpts = []
    for result_group in results:
        for result in result_group.get("results", []):
            excerpts.append(
                "Title: {title}\nURL: {url}\nPublished date: {published_date}\nContent: {content}".format(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    published_date=result.get("published_date")
                    or result.get("published_at", ""),
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
        fresh_query = f"{query} latest information as of {date.today().isoformat()}"
        response = tavily_client.search(
            fresh_query,
            search_depth="advanced",
            max_results=5,
        )

        results.append(response)


    # Summarize the research results.

    compact_results = _compact_search_results(results)
    research_prompt = f"""
You are a research synthesizer for technical writing.

Given the raw web search results below, produce a concise, deduplicated list
of EvidenceItem objects for the Orchestrator Agent.

Each EvidenceItem should preserve the useful evidence from one source, including
the source title, URL, a short factual snippet, and published_at when available.

Rules:
- Only include items with a non-empty URL.
- Prefer relevant and authoritative sources, such as company blogs, official
    documentation, and reputable outlets.
- If a published date is explicitly present in the result payload, keep it as
    YYYY-MM-DD. If it is missing or unclear, set published_at to null. Do not
    guess dates.
- Keep snippets short and factual.
- Deduplicate items by URL.
- Do not invent facts or dates.
- Do not include raw search-result JSON.

SEARCH RESULTS:

{compact_results}
"""


    messages = [

        SystemMessage(
            content="""
You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty URL.
- Prefer relevant and authoritative sources, such as company blogs, official
    documentation, and reputable outlets.
- If a published date is explicitly present in the result payload, keep it as
    YYYY-MM-DD. If missing or unclear, set published_at to null. Do not guess.
- Keep snippets short.
- Deduplicate by URL.
"""
        ),

        HumanMessage(
            content=research_prompt
        )

    ]


    # Ask the LLM to produce validated evidence items.
    response = structured_llm.invoke(messages)

    evidence_by_url = {}
    for item in response.evidence:
        url = item.url.strip()
        if url and url not in evidence_by_url:
            evidence_by_url[url] = item.model_copy(update={"url": url})


    # Return the research report.

    print("Finished Research_node.")
    return {
        "research_report": json.dumps(
            [item.model_dump() for item in evidence_by_url.values()],
            ensure_ascii=True,
        )
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