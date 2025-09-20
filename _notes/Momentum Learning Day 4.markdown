---
layout: note
title: "Momentum Learning Day 4 "
last_modified_at: 19-09-2025 
categories: [Momentum_Learning,Learning Journey]

---
 # Day 4| Momentum Learning Series

Today I went deeper into how **memory works inside Letta agents**. 

I set up a simple agent with two blocks (one for human input and one for persona ) and played around with inspecting its system prompt, tools, and memory history. 
It was nice to actually see how the agent keeps track of things behind the scenes.  


The main idea was the difference between **core memory** (what the agent actively uses in context) 
and **archival memory** (stuff saved for later but not in the immediate window). That split makes a lot of sense once you see it in action.  


I also tried customizing memory: adding new blocks and tools, and even building a small **task queue memory** where the agent can push and pop tasks. 

That part was fun, it felt like giving the agent a basic to-do list that it can manage on its own.  

What I liked most is how *programmable* the memory system is.
It’s not just the model doing black-box reasoning; you can shape how it remembers and interacts with information. 
As an engineer, that feels powerful.. it opens up room for building agents that aren’t only “smart,” but also structured and adaptable.  

**PS:** I haven’t studied the CV book for two days. Tomorrow I need to catch up *inchallah* and also finish this course!  
