"""MCP wire-protocol servers wrapping Noesis tool-use executors.

Each server is a thin FastMCP transport over the existing async
executors in core/tools/. Business logic is unchanged — these servers
add a wire protocol so the same tools are consumable by any MCP-aware
client (the chat pipeline keeps its in-process tool-use path; agents
on stdio/HTTP use these servers).

Servers:
  - rules_server.py   noesis-rules: 4 tools over the CDI rules engine
  - snomed_server.py  noesis-snomed: 4 tools over the SNOMED CT graph

Run a server in dev mode:
    mcp dev core/agents/mcp_servers/rules_server.py
    mcp dev core/agents/mcp_servers/snomed_server.py
"""
