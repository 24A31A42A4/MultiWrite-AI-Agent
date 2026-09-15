# **AgentWriter AI Requirements**

## **1. Project Goal**

AgentWriter AI is a multi-agent system that takes a user's topic/query and automatically generates a complete, well-structured blog/article.

The system should be able to:

Understand → Research → Plan → Write → Add Images/Citations → Combine → Review → Deliver

## **2. User Input**

**Required input:** Topic/query

## **3. System Output**

The final output should be a complete blog containing:

- Title
- Introduction
- Multiple sections
- Relevant content
- Conclusion
- Citations when research is used
- Images when required

## **4. Agents / Components**

### **Router Agent**

**Responsibility:**
Understand the user request and determine the required workflow.
It should decide things such as:
- Does this topic require research?
- Does it need current information?
- Is this a normal blog-generation request?

### **Research Agent**

**Responsibility:**

When research is required:
- Generate search queries
- Search relevant sources
- Collect information
- Extract useful facts
- Remove irrelevant information
- Produce a research report

### **Orchestrator**

**Responsibility:**
Create the execution plan for the blog.

It decides:
- How many sections?
- What is each section about?
- How many words?
- Does the section need research?
- Does it need citations?
- Does it need code?
- Does it need an image?

**Example:**

- Section 1: Introduction — Research: No, Image: No
- Section 2: How Agentic AI Works — Research: Yes, Image: Yes
- Section 3: LangGraph Architecture — Research: Yes, Code: Yes

### **Worker Agent**

**Responsibility:**
Write one individual section according to the Orchestrator's plan.

If there are 7 sections:

- Worker → Section 1
- Worker → Section 2
- Worker → Section 3
- Worker → Section 4
- Worker → Section 5
- Worker → Section 6
- Worker → Section 7

These can be executed in parallel.

### **Image Agent**

**Responsibility:**
Handle sections that require images.

It can:
- Determine image requirements
- Generate image prompts
- Generate images
- Associate images with sections

### **Reducer**

**Responsibility:**
Take all Worker outputs and create one complete blog.

It should:
- Put sections in the correct order
- Combine them
- Remove unnecessary duplication
- Maintain consistency
- Integrate citations/images

## **Observability**

You should be able to see all the steps.