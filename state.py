import langchain,langgraph

from langgraph.graph import StateGraph,state,END,START
import operator
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel,Field

# This file stores the data structure for the multi-agent blog workflow.
# Each class defines the information passed between agents.

# Router passes the user request and decides if research is needed.
class RouterState(TypedDict):
    user_topic: str
    research_required: bool
    queries: list[str]

# Research agent stores the search queries and final research results.
class ResearchState(TypedDict):
    queries: list[str]
    research_report: str

# Each task is one section or part of the blog plan.
class Task(BaseModel):
    id: str
    title: str
    goal: str
    bullets: list[str]
    target_words: int
    tags: list[str]
    requires_research: bool
    requires_citations: bool
    requires_code: bool
    requires_image: bool = False

# Full plan for the blog: title, audience, tone, and all tasks.
class Plan(BaseModel):
    blog_title: str = Field(description="Name of the blog title")
    audience: str = Field(description="Who are the audience")
    tone: str = Field(description="The tone of the blog")
    blog_kind: Literal["Explainer", "tutorial", "comparison", "system_design"]
    constraints: list[str]
    tasks: list[Task]

# Orchestrator combines the topic, research, and final plan.
class OrchestratorState(TypedDict):
    user_topic: str
    research_report: str
    plan: Plan

# Worker stores the section being written.
class WorkerState(TypedDict):
    task: Task
    research_report: str
    section_outputs: list[str]

# Image agent stores the generated image details.
class ImageState(TypedDict):
    task: Task
    image_prompt: str
    image_path: str
    image_data: str

# Reducer combines all section results into the final blog.
class ReducerState(TypedDict):
    plan: Plan
    section_outputs: list[str]
    final_blog: str

# Full app state used to connect all agents together.
# section_outputs uses Annotated + operator.add so parallel workers can merge results.
class AgentWriterState(TypedDict):
    user_topic: str
    research_required: bool
    queries: list[str]
    research_report: str
    plan: Plan
    section_outputs: Annotated[list[str], operator.add]
    image_prompt: str
    image_path: str
    final_blog: str