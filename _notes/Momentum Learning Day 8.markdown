---
layout: note
title: "Momentum Learning Day 8"
last_modified_at: 24-09-2025 
categories: [Momentum_Learning,Learning Journey]
---

# Day 8 | Momentum Learning Series

Today I dug into the difference between **`CodeAgent`** and **`ToolCallingAgent`** in *smolagents*.  

The key learning is that **`CodeAgent` generates Python code** while **`ToolCallingAgent` outputs JSON blobs** with tool names and arguments.
That shift in representation really matters:  
- With Python, I get flexibility and the ability to debug by reading the generated code.  
- With JSON, I get structure and predictability, which feels closer to API wiring.  

Even the traces highlight this difference, a CodeAgent shows “Executing parsed code …” while a ToolCallingAgent shows “Calling tool … with arguments …”.
Seeing that contrast made me realize how the **action format shapes both observability and debugging flow**.  

To practice, I extended my “party planner” agent Alfred using both methods.
With the `@tool` decorator, I built a quick `catering_service_tool()` to simulate picking the best catering in Gotham. 

With the subclass method, I wrote `SuperheroPartyThemeTool`, where I explicitly defined inputs, outputs, and a forward function. 

This hands-on contrast made it clear:  
- The decorator path is perfect for quick prototyping.  
- The subclass path forces more structure, which scales better for complex systems.  

Writing these tools also taught me that designing them is basically **API design**!! I had to be deliberate with names, argument types, and descriptions, otherwise the agent reasoning would get messy. 
That clicked as a very *software engineering* way of thinking about AI.  

By the end of today, I see the tradeoff clearly: **JSON calls give clean structure, Python execution gives expressive power.** 
Choosing one is less about “which is better” and more about “what the workflow needs.” 

That mindset shift felt like moving from “using AI” to actually **engineering orchestration layers between reasoning and execution**.  


End of Day 8. My biggest takeaway: *how actions are represented  code vs JSON changes everything about how the agent behaves and how I interact with it as an engineer*.  
