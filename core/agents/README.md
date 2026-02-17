# Agents (Room 3 — Planned)

This folder is reserved for agent orchestration, planning policies, and LangGraph flows.

## Current Status

No agents implemented yet. Room 2 (External Knowledge / MCP filesystem) is the current priority.
Agents are planned for Room 3, after document ingestion is working.

## Planned Capabilities

- **Research Agent:** Multi-step research with tool use (search memory, search documents,
  query knowledge graph, generate diagram, propose memory)
- **Diagram Refinement Agent:** Iteratively improve diagrams with human-in-the-loop approval
- **Memory Curation Agent:** Find duplicates, contradictions, suggest merges across memory cards
- **Document Analysis Agent:** Summarize ingested documents, extract key entities, propose graph structures

## Framework

LangGraph for visible reasoning workflows — agent steps displayed as visual graphs
in the Graph Explorer tab, aligning with Noesis "artifacts over chat" philosophy.

## Dependencies

LangChain and LangGraph are installed in requirements.txt but not yet active.
These will be used in Room 3 after Room 2 document ingestion is complete.
