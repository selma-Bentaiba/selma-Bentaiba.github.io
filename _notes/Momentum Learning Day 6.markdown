---
layout: note
title: "Momentum Learning Day 6 "
last_modified_at: 22-09-2025 
categories: [Momentum_Learning,Learning Journey]

---
 # Day 6| Momentum Learning Series

After finishing the **LLMs as Operating Systems** course yesterday, I didn’t want to lose momentum, but I also didn’t want to just rush into something shiny and new.  
So today was about **reinforcing foundations** while continuing the bigger plan I set for myself around agents.  


## Revisiting CNNs  

I went back to the **deep learning for computer vision book** and re-read the CNN chapters.
I already know CNNs, but coming back to them with an engineer’s mindset feels different than just “studying the theory.”  

When you look at CNNs in the context of building CV systems, the design choices really stand out:  

- **Convolutional layers** → not just math, but a way to *force the network* to learn local patterns.  
- **Pooling layers** → a clever compression trick: *you don’t need every pixel, just the essence.*  
- **Fully connected layers** → the collapse point, where the network finally says: *alright, classify this thing already.*  

The beauty here is **constraints leading to elegance**.  
A kernel is tiny — just a `3×3` or `5×5` matrix of weights — but sliding it across an image extracts edges, textures, and higher-level features.
That minimal design is why CNNs became the backbone of modern vision, and why they still matter even when transformers dominate the headlines.  

So no, this wasn’t *new learning*.  
It was sharpening a tool I know I’ll need later in this CV book.  

---

## HuggingFace Agents and Picking Up the Plan  

The second thread today was getting back to the **AI Agents course from HuggingFace**, specifically the **Smolagents framework**.  
This is something I had started in the summer but left unfinished. 
Picking it up now iq part of the roadmap.  

What stood out about **Smolagents**:  

- **Code-first** → you don’t just prompt and hope; you define actions in code.  
- **Lightweight** → minimal abstraction, fast experiments.  
- **Flexible** → HuggingFace Hub + multiple LLMs supported out of the box.  

It supports:  

- **CodeAgents** (the core type).  
- **ToolCallingAgents** (via JSON).  
- **Multi-step workflows** → chain actions together.  

Compared to yesterday’s look at Letta and multi-agent orchestration, Smolagents feels like the **sandbox** where I can actually get my hands dirty, test ideas, and learn fast.  

---

## Reflection  

Day 6 wasn’t about breakthroughs, it was about **engineering discipline**:  

- Revisiting CNNs → reinforced why their design still matters in CV.  
- Smolagents → gave me a lightweight, practical entry point for agents.  

If Day 5 was about *seeing the architecture*,  
Day 6 was about *picking the right tools off the shelf and checking they’re sharp.*  
