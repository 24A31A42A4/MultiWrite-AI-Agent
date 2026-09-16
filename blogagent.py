from state import MediumAgent
from model import model
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
import requests
import os
import re
import mimetypes
from langchain.messages import HumanMessage,SystemMessage
from pathlib import Path

load_dotenv()
DEVTO_API_KEY = os.getenv("DEVTO_API_KEY")


def _upload_markdown_images(markdown: str, base_dir: Path) -> str:
    """Upload local Markdown images and replace paths with public DEV.to URLs."""
    image_pattern = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")

    def replace_image(match):
        alt_text, image_reference = match.groups()
        if image_reference.startswith(("http://", "https://", "data:")):
            return match.group(0)

        image_path = Path(image_reference)
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        if not image_path.exists() or not image_path.is_file():
            print(f"Image file not found, keeping reference: {image_reference}")
            return match.group(0)

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    "https://dev.to/api/images",
                    headers={"api-key": DEVTO_API_KEY},
                    files={
                        "image": (image_path.name, image_file, mime_type),
                    },
                    data={"alt_text": alt_text},
                    timeout=60,
                )
            if not response.ok:
                print(f"Image upload failed ({response.status_code}): {response.text[:300]}")
                return match.group(0)
            image_url = response.json().get("url")
            if not image_url:
                print(f"Image upload returned no URL: {response.text[:300]}")
                return match.group(0)
            print(f"Uploaded image: {image_path.name}")
            return f"![{alt_text}]({image_url})"
        except requests.RequestException as error:
            print(f"Image upload request failed: {error}")
            return match.group(0)

    return image_pattern.sub(replace_image, markdown)


def _remove_leading_title(markdown: str) -> str:
    """DEV.to renders the article title separately from body Markdown."""
    return re.sub(r"^\s*#\s+.+?\s*\n+", "", markdown, count=1)


def _local_image_references(markdown: str) -> list[str]:
    image_pattern = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
    return [
        reference
        for reference in image_pattern.findall(markdown)
        if not reference.startswith(("http://", "https://", "data:"))
    ]

@tool
def publish_to_devto(
    title: str,
    content: str,
    tags: list[str],
    published: bool = False
) -> str:
    """Publish an article to DEV Community."""

    if not DEVTO_API_KEY:
        return "DEV.to publishing failed: DEVTO_API_KEY is not configured."

    normalized_tags = [
        tag.strip().lower()
        for tag in tags
        if tag and tag.strip()
    ][:4]

    url = "https://dev.to/api/articles"

    payload = {
        "article": {
            "title": title,
            "body_markdown": content,
            "published": published,
            "tags": normalized_tags
        }
    }

    response = requests.post(
        url,
        json=payload,
        headers={
            "api-key": DEVTO_API_KEY,
            "Content-Type": "application/json"
        }
    )

    if not response.ok:
        try:
            details = response.json()
        except ValueError:
            details = response.text[:500]
        return f"DEV.to publishing failed ({response.status_code}): {details}"

    data = response.json()

    article_url = data.get("url")
    if not article_url:
        return f"DEV.to returned an unexpected response: {data}"
    return f"Article published: {article_url}"


def blog_agent(state:MediumAgent):
    print("\nRunning blog_agent...")
    blog_file_path = state.get("blog_file_path", "")
    blog_content = (
        Path(blog_file_path).read_text(encoding="utf-8")
        if blog_file_path
        else state["final_blog"]
    )
    publish_requested = state.get("publish_requested", False)

    if not publish_requested:
        print("Skipping DEV.to publishing because publish_requested is false.")
        print("Finished blog_agent.")
        return {
            "final_blog": blog_content,
            "publish_requested": False,
            "blog_file_path": blog_file_path,
        }

    print("DEV.to publishing requested. Uploading the saved Markdown directly...")
    if not DEVTO_API_KEY:
        print("DEV.to publishing skipped: DEVTO_API_KEY is not configured.")
        return {"final_blog": blog_content, "blog_file_path": blog_file_path}

    blog_content = _upload_markdown_images(
        blog_content,
        Path(blog_file_path).parent if blog_file_path else Path.cwd(),
    )
    unresolved_images = _local_image_references(blog_content)
    if unresolved_images:
        print(
            "DEV.to publishing skipped because image upload failed for: "
            + ", ".join(unresolved_images)
        )
        return {"final_blog": blog_content, "blog_file_path": blog_file_path}

    if blog_file_path:
        Path(blog_file_path).write_text(blog_content, encoding="utf-8")

    title_match = re.search(r"^#\s+(.+?)\s*$", blog_content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "AgentWriter Article"
    publish_content = _remove_leading_title(blog_content)
    tags = ["ai", "llm", "opensource", "engineering"]
    publish_result = publish_to_devto.invoke({
        "title": title,
        "content": publish_content,
        "tags": tags,
        "published": True,
    })
    print(publish_result)
    print("Finished blog_agent.")
    return {"final_blog": blog_content, "blog_file_path": blog_file_path}


blog_graph=StateGraph(MediumAgent)

blog_graph.add_node("Blog_agent",blog_agent)

blog_graph.add_edge(START,"Blog_agent")
blog_graph.add_edge("Blog_agent",END)
