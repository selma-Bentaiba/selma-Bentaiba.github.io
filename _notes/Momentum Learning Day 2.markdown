---
layout: note
title: "Momentum Learning Day 2 "
last_modified_at: 18-09-2025 23.55
categories: [Momentum_Learning,Learning Journey]

---
 # Day 2| Momentum Learning Series 


To strengthen and deepen my knowledge of agents, I started a new course called [LLMs as Operating Systems: Agent Memory](https://learn.deeplearning.ai/courses/llms-as-operating-systems-agent-memory).

It introduces the idea of self-editable memory in agents, with a focus on the MemGPT design.

The key concept is that the context input window can be thought of as a kind of virtual memory in a computer,
where the LLM agent also plays the role of an operating system deciding which information should go into the input context window.


As practice, I built a simple agent with editable memory from scratch, using only a Python dictionary as the memory.

I then designed a function to update this memory when needed, and instructed the LLM (OpenAI in this case) on how to use the tool (the “function”).

The most important part was creating an agentic loop, so the agent could perform multi-step reasoning as required.

This felt like working at a low-level foundation something I can build on later to create more complex agents with memory.

## Book part

After that, I studied from the Deep Learning for Computer Vision book. I reviewed how, in image classification, MLPs (fully connected layers) struggled with feature learning from images.

This is why CNNs replaced them for feature extraction, while fully connected layers remain useful for the classification stage.

The high-level CNN architecture looks like this:

* Input layer
* Convolutional layers (for feature extraction)
* Fully connected layer (for classification)
* Output prediction

The convolutional layers produce feature maps of the image. With each layer, the image dimensions shrink while the depth increases, until we end up with a long array of small features.
These are then fed into the fully connected layer for classification.

In terms of feature learning:

* Early layers detect low-level features such as lines and edges.
* Later layers detect patterns within patterns, gradually learning more complex features until the model captures the bigger picture.
