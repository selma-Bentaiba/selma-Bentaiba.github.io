---
layout: note
title: "Momentum Learning Day 7"
last_modified_at: 23-09-2025 
categories: [Momentum_Learning,Learning Journey]

---
 # Day 7| Momentum Learning Series

Today I got deeper into **smolagents**, specifically the `CodeAgent`.

The core insight: letting the agent write and execute **Python code** instead of just JSON unlocks flexibility and makes tool use feel natural.  


In practice, this means you don’t just get a black-box answer. You see:
- the Python code the model generated,
- the execution results,
- and the reasoning steps logged in memory.

For me, that transparency was the big shift. It felt less like guessing and more like debugging with a colleague.

---

## Building Alfred (with smolagents)
I assembled a demo agent "Alfred" (the butler) using **smolagents**.  

- **Custom tools I wrote:**
  - `suggest_menu()` → suggest menus depending on the occasion.  
  - `catering_service_tool()` → simulate picking the best catering service in Gotham.  
  - `SuperheroPartyThemeTool` → generate themed ideas .  

- **Prebuilt tools I plugged in:**
  - `DuckDuckGoSearchTool` (search),  
  - `VisitWebpageTool` (navigate),  
  - `FinalAnswerTool` (format output).  

With these wired into a `CodeAgent`, I could ask:  
 “Give me the best playlist for a party at Wayne’s mansion. Theme: villain masquerade.”  

Alfred went step by step: picked the theme, searched, browsed links, and finally returned a curated playlist.  
Watching each tool call and execution log made the whole process feel robust and traceable.

---

## Reflections as an Engineer
- **smolagents is pragmatic**: the `MultiStepAgent` + execution log design is exactly what makes debugging feasible.  
- **Small tools matter**: even trivial ones (`suggest_menu`) gave structure and extended capabilities.  
- **Observability is real**: I like that smolagents integrates with OpenTelemetry + Langfuse. Being able to replay a run or see why it failed is non-negotiable in production.  
- **Feels future-proof**: this setup makes agents composable, testable, and closer to real software systems rather than “magic prompts.”  

---

## Next Step
The party planner was fun, but the same pattern applies to serious workflows.  
Next, I want to try building a study assistant that schedules prep tasks with [datetime] and pushes runs to the Hugging Face Hub for reuse.  

End of Day 7. I feel like I’m not just learning AI concepts anymore, I’m actually starting to think like an **engineer of agents**.
