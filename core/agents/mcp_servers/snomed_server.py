"""noesis-snomed MCP server.

Wraps the 4 SNOMED CT tools in core/tools/snomed_tools.py behind the
MCP wire protocol. Business logic lives in execute_tool() and the
underlying _snomed_* helpers; this file is transport only.

Run as a stdio server for an MCP client to connect:

    PYTHONPATH=. python -m core.agents.mcp_servers.snomed_server

Open the MCP Inspector against this server. See the launcher note
in rules_server.py for the `npx @modelcontextprotocol/inspector`
incantation — same pattern applies here.

Note: the existing execute_tool in core/tools/snomed_tools.py is a
combined dispatcher — it handles snomed_* tools directly and falls
through to rules_tools.execute_tool for rules_* names. This server
only exposes the snomed_* tools; the rules tools live in their own
server (noesis-rules) for clean per-domain scoping.

Requires Neo4j at NEO4J_URI (default bolt://192.168.4.25:7687) to be
reachable. Set via .env. NOES-64 tracks pre-existing log warnings
from MAPS_TO->ICD10Code on missing priority/advice properties; the
results still return correctly with those fields as null.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from core.tools.snomed_tools import execute_tool

mcp = FastMCP("noesis-snomed")


@mcp.tool()
async def snomed_search(query: str, limit: int = 10) -> str:
    """Search SNOMED CT concepts by name.

    Use this when the user asks about a clinical concept and you need
    to find its SNOMED CT code (SCTID), or when you need to look up what
    concepts exist for a given clinical term like 'heart failure' or
    'diabetes'.

    Args:
        query: Clinical term to search for (e.g., 'heart failure',
               'sepsis', 'diabetes mellitus').
        limit: Maximum number of results to return (default 10).
    """
    return await execute_tool("snomed_search", {"query": query, "limit": limit})


@mcp.tool()
async def snomed_get_concept(sctid: str) -> str:
    """Get full details for a SNOMED CT concept by its SCTID.

    Returns parent concepts (IS_A), clinical relationships (finding site,
    associated morphology, causative agent, etc.), child concepts, and
    ICD-10 cross-mappings. Use this when you already have a SCTID and
    need to understand the concept's clinical context.

    Args:
        sctid: SNOMED CT concept ID (e.g., '84114007' for heart failure).
    """
    return await execute_tool("snomed_get_concept", {"sctid": sctid})


@mcp.tool()
async def snomed_get_descendants(sctid: str, max_depth: int = 2) -> str:
    """Get all descendant concepts of a SNOMED CT concept.

    Use this when the user asks 'what types of X exist?' or 'what are
    the subtypes of X?' For example, descendants of 'heart failure'
    include systolic heart failure, diastolic heart failure, acute
    heart failure, etc.

    Args:
        sctid: SNOMED CT concept ID to get descendants for.
        max_depth: Maximum hierarchy depth to traverse (clamped to 1-5
                   by the executor; default 2). Limit 200 descendants
                   returned regardless of depth.
    """
    return await execute_tool(
        "snomed_get_descendants",
        {"sctid": sctid, "max_depth": max_depth},
    )


@mcp.tool()
async def snomed_get_ancestors(sctid: str, max_depth: int = 2) -> str:
    """Get all ancestor concepts of a SNOMED CT concept.

    Use this when the user asks 'what category does X belong to?' or
    'what is X a type of?' For example, ancestors of 'acute systolic
    heart failure' include 'heart failure', 'disorder of cardiovascular
    system', etc.

    Args:
        sctid: SNOMED CT concept ID to get ancestors for.
        max_depth: Maximum hierarchy depth to traverse (clamped to 1-5
                   by the executor; default 2).
    """
    return await execute_tool(
        "snomed_get_ancestors",
        {"sctid": sctid, "max_depth": max_depth},
    )


if __name__ == "__main__":
    mcp.run()
