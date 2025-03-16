---
layout: page
title: Notes
permalink: /notes/
---

This is where I write down my random daily thoughts, my weekly learnings, goals, plans etc.

{% for note in site.notes %}
<div class="note-item">
  <!-- Display the image at the top -->
  {% if note.image %}
    <img src="{{ note.image | relative_url }}" alt="{{ note.title | escape }}" class="note-image">
  {% endif %}

  <!-- Toggle for title and content -->
  <details>
    <summary>{{ note.title | escape }}</summary>
    <p class="note-date">Last updated: {{ note.last_modified_at | date: "%B %-d, %Y" }}</p>
    {{ note.content }}
  </details>
</div>
{% endfor %}

<style>
  .note-item {
    margin-bottom: 2rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 1.5rem;
  }
  .note-image {
    max-width: 100%;
    height: auto;
    margin-bottom: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  .note-date {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
  }
  details summary {
    cursor: pointer;
    font-weight: bold;
    padding: 0.5rem 0;
    list-style: none;
  }
  details summary:hover {
    color: #0366d6;
  }
  details[open] summary {
    margin-bottom: 0.5rem;
  }
</style>
