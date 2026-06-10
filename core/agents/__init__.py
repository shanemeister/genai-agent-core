"""Agent and MCP server implementations for Noesis.

Phase 5 territory. Currently houses:
  - mcp_servers/       Wire-protocol MCP servers (NOES-63)
                       wrapping the existing tool-use executors in
                       core/tools/. Same business logic; different
                       transport.

Planned (per project_development_plan.md Phase 5 + Phase 5 Spike):
  - case_reviewer.py   LangGraph multi-agent orchestrator. NOES-62
                       (spike) is on feature/p5-spike-case-reviewer-
                       langgraph; the full Phase 5 build is gated on
                       NOES-23/24/25 (safety + grouper + workflow).
"""
