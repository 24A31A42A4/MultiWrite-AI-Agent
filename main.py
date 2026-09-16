from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from state import AgentWriterState
from router import router_agent
from research import Research_node
from orchestrator import planner_agent
from reducer import reducer_node
from worker import worker_node
from image_agent import image_node
from blogagent import blog_agent
import os
# Create the main agent writer graph.
AgentWriter = StateGraph(AgentWriterState)

# Register the actual runnable functions, not the sub-graphs.
AgentWriter.add_node("router_agent", router_agent)
AgentWriter.add_node("Research_agent", Research_node)
AgentWriter.add_node("Orchestrator_agent", planner_agent)
AgentWriter.add_node("reducer_agent", reducer_node)
AgentWriter.add_node("worker_agent", worker_node)
AgentWriter.add_node("image_agent", image_node)
AgentWriter.add_node("Blog_agent",blog_agent)


def save_blog_node(state: AgentWriterState):
    output_path = os.path.abspath(f'{state["user_topic"]}.md')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(state["final_blog"])
    print(f"Saved complete Markdown blog to {output_path}")
    return {"blog_file_path": output_path}


AgentWriter.add_node("save_blog", save_blog_node)

# Route to research or skip to orchestrator based on router decision.
def route_after_router(state):
    print("\nRunning route_after_router...")
    if state.get("research_required"):
        print("Finished route_after_router -> research_needed.")
        return "research_needed"
    print("Finished route_after_router -> research_not_needed.")
    return "research_not_needed"


# Connect the edges: start -> router -> research or orchestrator.
AgentWriter.add_edge(START, "router_agent")
AgentWriter.add_conditional_edges(
    "router_agent",
    route_after_router,
    {
        "research_needed": "Research_agent",
        "research_not_needed": "Orchestrator_agent",
    },
)

# After research, move to the orchestrator.
AgentWriter.add_edge("Research_agent", "Orchestrator_agent")


# Fan out tasks to worker and image agents in parallel.
def fanout_tasks(state: AgentWriterState):
    print("\nRunning fanout_tasks...")
    sends = []
    worker_count = len(state["plan"].tasks)
    image_count = sum(
        1 for task in state["plan"].tasks if getattr(task, "requires_image", False)
    )
    print(
        f"Planning {worker_count} blog sections and {image_count} image tasks "
        f"({worker_count + image_count} total dispatches)."
    )
    for task in state["plan"].tasks:
        # Send image tasks to the image agent if needed.
        if getattr(task, "requires_image", False):
            sends.append(
                Send(
                    "image_agent",
                    {
                        "task": task,
                        "image_prompt": f"Create a blog illustration for: {task.title}",
                    },
                )
            )
        # Send every task to a worker agent to write the section.
        sends.append(
            Send(
                "worker_agent",
                {
                    "task": task,
                    "research_report": state.get("research_report", ""),
                        "audience": state["plan"].audience,
                        "tone": state["plan"].tone,
                },
            )
        )
    print(
        f"✅ END fanout_tasks: {worker_count} worker sections, "
        f"{image_count} image tasks, {len(sends)} total dispatches"
    )
    return sends


# After orchestrator, fan out to workers and image agents.
AgentWriter.add_conditional_edges(
    "Orchestrator_agent",
    fanout_tasks,
    ["worker_agent", "image_agent"],
)

# After all workers and image agents finish, reduce into the final blog.
AgentWriter.add_edge("worker_agent", "reducer_agent")
AgentWriter.add_edge("image_agent", "reducer_agent")
AgentWriter.add_edge("reducer_agent", "save_blog")
AgentWriter.add_edge("save_blog", "Blog_agent")
AgentWriter.add_edge("Blog_agent", END)

# Compile the graph into a runnable app.
agent_writer_app = AgentWriter.compile()

# Save a visual diagram of the graph.
graph_image_png = agent_writer_app.get_graph().draw_mermaid_png()
with open("agent_writer.png", "wb") as f:
    f.write(graph_image_png)


# Run the blog generation pipeline.
response = agent_writer_app.invoke(
    {
        "user_topic": "How AI Agents Are Changing Software Development in 2026",
        "publish_requested": True,
    },
    config={"max_concurrency": 3}
)

# Save the final blog to a file.
final_blog = response.get("final_blog") or response.get("final_blog", "")

if final_blog:
    print(f"✅ Final blog generated: {response.get('blog_file_path', 'in memory')}")
else:
    print(response)
    print("No final blog content found in the response.")