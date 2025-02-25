---
layout: page
title: Notes
permalink: /notes/
---

This is where I write down my random daily thoughts, my weekly learnings, goals, plans etc.

{% for note in site.notes %}
<details class="note-item">
  <summary>{{ note.title | escape }}</summary>
  <p class="note-date">Last updated: {{ note.last_modified_at | date: "%B %-d, %Y" }}</p>
  {% if site.show_excerpts %} 
    {{ note.excerpt }} 
  {% else %}
    {{ note.content }}
  {% endif %}
</details>
{% endfor %}

<style>
  .note-item {
    margin-bottom: 1rem;
  }
  .note-date {
    font-size: 0.8rem;
    color: #666;
  }
  details summary {
    cursor: pointer;
    font-weight: bold;
    padding: 0.5rem 0;
  }
  details summary:hover {
    color: #0366d6;
  }
</style>
