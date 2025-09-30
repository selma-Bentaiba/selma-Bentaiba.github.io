---
layout: note
title: "Momentum Learning Day 9"
last_modified_at: 29-09-2025 
categories: [Momentum_Learning,Learning Journey]
---

# Day 9 | Momentum Learning Series

After four days away, I came back to the course today.
Pausing was a useful reminder: momentum is fragile, but once the concepts have been internalized, resuming feels less like starting over and more like reconnecting with a system already in place.  

The **Agents course** is still long, but I’m steadily moving through it , aiming to complete the **smolagents** section tomorrow inchallah.  


## Retrieval Agents  

Today’s focus was on **Retrieval-Augmented Generation (RAG)** and its *agentic* extension.  

- **Traditional RAG**: simply retrieval + generation.  
- **Agentic RAG**: retrieval becomes iterative and reflective.
    Agents can formulate queries, evaluate results, and loop until a satisfying outcome is reached.  

This shift made me see retrieval not as a static lookup, but as **a reasoning layer tightly integrated with the agent’s decision cycle**.  


## What I Implemented  

1. **Web Search Agent with DDGS**  
   - Used `CodeAgent` with `DuckDuckGoSearchTool`.  
   - Flow: analyze request → retrieve → process → store for reuse.  
   - This embedded retrieval directly inside the reasoning process, rather than treating it as a side operation.  

2. **Custom Knowledge Base with BM25Retriever**  
   - Built a small knowledge set (superhero party themes).  
   - Applied a text splitter, then designed `PartyPlanningRetrieverTool` with BM25 to return top 5 ranked results.  
   - Engineering perspective: **constructing a pipeline** — raw docs → embeddings/index → retriever → agent reasoning.  


## Embedded Reflections  

- Building tools felt less like “trying out features” and more like **designing interfaces for agents to reason over knowledge**.  
- BM25 gave precise ranking control, showing that retrieval quality is deeply tied to algorithmic choice, not just embeddings.  
- Compared to earlier exercises, today’s work had more of a **system-architecture feel**: retrieval pipelines as part of the reasoning flow, not isolated utilities.  

---

## Quiz Checkpoints  

- **Tool Creation** → lightweight functions via `@tool`; complex ones via `Tool` subclasses.  
- **CodeAgent & ReAct** → iterative cycle of reasoning, action, feedback, adjustment.  
- **Tool Sharing** → Hugging Face Hub makes custom tools reusable across projects.  
- **ToolCallingAgent** → emits JSON with tool + arguments.  
- **Default Toolbox** → provides baseline tools (search, Python, etc.) for prototyping.  


**Takeaway:** Retrieval isn’t passive storage or search; it’s an **active reasoning partner** in agent workflows. 
And even with long gaps, progress compounds: the deeper the system level understanding, the quicker I can pick up where I left off.

At this point, it’s less about learning a course and more about shaping the mindset of an agent systems architect, designing the reasoning flow itself, not just calling tools.
