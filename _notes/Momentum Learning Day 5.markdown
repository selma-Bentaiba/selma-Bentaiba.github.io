---
layout: note
title: "Momentum Learning Day 5 "
last_modified_at: 21-09-2025 
categories: [Momentum_Learning,Learning Journey]

---
 # Day 5| Momentum Learning Series

Today I wrapped up the **LLMs as Operating Systems** course. It feels like a milestone because now I see the bigger picture of how LLMs can actually work as the “core” of applications, not just a chatbot.


## Agentic RAG & External Memory

The main takeaway was about **giving agents memory and data sources**:

* One way is just copying data into their **archival memory** (like a built-in database the agent can look up).
* The other way is connecting the agent to an **external tool** that can query data on demand.

I tested both:

* Created a source (“employee handbook”), uploaded a file, attached it to the agent, and made sure embeddings matched. Once connected, the agent could reference the file like it had read it itself.
* Then I built a dummy “database” (just a dictionary) and plugged it into the agent via a tool. The agent could call this tool and fetch answers from it.

From an engineer’s perspective, this makes things more **modular**. Instead of cramming everything into the context window, we can design agents that **reach out for information**.


## Multi-Agent Orchestration

The second part was about **getting multiple agents to collaborate**. 
In Letta, agents are meant to run as services, so the question is: how do they talk to each other?

There are two ways:

1. **Message tools** → one agent can send a message to another.
2. **Shared memory blocks** → two agents share the same context window so they both “see” the same data.

I built two agents:

* One for outreach (like sending resumes).
* Another for evaluation (with a reject/approve tool).

They passed messages back and forth, and with shared memory, both had the same view of what was going on.

Finally, I tried the **multi-agent abstraction**: put both agents into a single group chat. That was simpler, but the idea is the same agents can coordinate either by tools or by a shared space.

For me, this part really shows how we can move beyond a single “all knowing” agent. 
We can design **specialized agents** that cooperate almost like microservices, but in natural language.

---

## Reflection

I’ll be honest: I rushed through today to keep up with the consistency streak. I didn’t touch the CV book again,
and I know I need to fix my timing so I’m not just checking boxes but actually **digesting** the material.

Still, I’m happy I finished the course, it gave me a clear technical intuition about how LLM-based systems are structured:

* Memory is not just a bigger context window, but a system of archives and tools.
* Agents can be extended and composed, almost like APIs calling each other.

That’s powerful to know as an engineer. Inchallah, tomorrow I’ll slow down a bit, return to the CV book, and balance the depth with consistency.
